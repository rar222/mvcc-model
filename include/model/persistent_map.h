#pragma once
//
// A minimal persistent hash map/set (HAMT: hash array mapped trie).
//
// Why this exists: the secondary index (external key -> Id) must be part of every
// published Root, and Roots are immutable and cheap to derive from one another. A
// std::unordered_map forces an O(n) copy per commit -- the exact size-proportional
// cost the chunked object store is designed to avoid. A HAMT shares structure
// between versions, so producing a new version path-copies only O(log32 n) nodes.
//
// This is a from-scratch, dependency-free implementation (no network to pull
// immer). It supports exactly what the model needs: set/insert, erase, get/
// contains, and cheap persistent derivation. It is NOT a general-purpose
// container.
//
// Layout: a 32-ary trie keyed on 5-bit slices of the key's hash. Each internal
// node holds a 32-bit bitmap and a densely packed array of children; a child is
// either another node or a leaf. A leaf holds ONE entry inline; collisions
// (full 64-bit hash equal, or deeper than the hash provides) chain leaves
// together and are compared by the actual key.
//
// Key type is a template parameter, hashed via `Hash` -- a stateless functor
// template parameter, no default, the same convention this project already uses
// for Id-keyed std::unordered_map/std::unordered_set (see model::IdHash). This
// keeps this header dependency-free of model::Id while still letting Model key a
// PersistentMap/PersistentSet directly on Id -- see StringHash below for the one
// instantiation this file provides itself, since every OTHER user of this header
// lives in model.h/model.cpp and supplies its own Hash (IdHash).
//
// PersistentMap<K,V,Hash> and PersistentSet<K,Hash> are both thin wrappers over
// one shared trie implementation, detail::TrieCore -- the two differ only in
// what a leaf entry holds (a (K,V) pair, or a bare K) and how a comparison key is
// extracted from one, via the KeyOf template parameter. Path-copying, collision
// push-down, and bucket shrink/drop-on-empty are tricky enough to want living in
// exactly one place rather than duplicated between a map and a set version.

#include <cassert>
#include <cstdint>
#include <memory>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace model::pmap {

/// Hasher for a std::string key. FNV-1a, 64-bit -- adequate for dispersing
/// string keys across the trie. A named, reusable functor (rather than a
/// private method baked into the trie) so it can be passed as the explicit
/// Hash argument wherever a PersistentMap is keyed by a real string
/// (Root::by_field, and the OUTER map of Root::by_cached_field) -- see
/// model.h's Root for those instantiations.
struct StringHash {
    std::uint64_t operator()(const std::string& k) const noexcept {
        std::uint64_t h = 1469598103934665603ull;
        for (unsigned char c : k) {
            h ^= c;
            h *= 1099511628211ull;
        }
        return h;
    }
};

namespace detail {

/// Entry = std::pair<K,V> (PersistentMap's shape): the comparison key is the
/// pair's first half.
template <class K, class V>
struct PairKeyOf {
    const K& operator()(const std::pair<K, V>& e) const noexcept { return e.first; }
};

/// Entry = K itself (PersistentSet's shape): the entry IS the comparison key,
/// so "replacing" a matched entry with itself (see TrieCore::set_entry's
/// collision-bucket step) is a correct, harmless no-op -- no special-casing
/// needed anywhere in TrieCore for the value-less case.
template <class K>
struct IdentityKeyOf {
    const K& operator()(const K& e) const noexcept { return e; }
};

/// Detects an opt-in `static constexpr bool is_perfect = true;` on a Hash
/// functor -- the same optional-static-member detection idiom model.h's
/// has_define_references<> etc. use, so a Hash type that never declares it
/// (StringHash, or any user-supplied Hash) defaults to false via SFINAE
/// rather than a hard error. `is_perfect` is a promise from the Hash's
/// author: DISTINCT keys always produce DISTINCT 64-bit hash values --
/// model::IdHash makes exactly this claim in its own doc comment (Id's
/// (gen,index) pair packed losslessly into 64 bits). TrieCore uses the
/// promise to drop collision-chain storage entirely for that Hash's Leaf
/// shape (see hash_is_perfect_v below and Leaf's own comment) -- getting
/// this wrong for a Hash that ISN'T actually collision-free silently
/// corrupts the trie (see chain_copy's assert), so only declare it where
/// the guarantee is real, not just "very likely."
template <class Hash, class = void>
struct hash_is_perfect : std::false_type {};
template <class Hash>
struct hash_is_perfect<Hash, std::void_t<decltype(Hash::is_perfect)>>
    : std::bool_constant<Hash::is_perfect> {};
template <class Hash>
inline constexpr bool hash_is_perfect_v = hash_is_perfect<Hash>::value;

/// The trie itself, parameterized on what a leaf actually stores (Entry) and
/// how to get a K back out of one (KeyOf). PersistentMap and PersistentSet
/// below are both ~30-line forwarding wrappers over this.
template <class K, class Entry, class Hash, class KeyOf>
class TrieCore {
    // One entry INLINE per Leaf, with a chain link for collisions, rather
    // than a vector of entries: a vector costs a second heap allocation
    // (its buffer) for every single-entry leaf -- i.e. for essentially
    // every entry in the trie, since a chain longer than one needs two keys
    // sharing a full 64-bit hash. Measured on the bulk-load memory
    // benchmark, that second allocation was a double-digit share of the
    // whole per-entry footprint.
    //
    // No stored `hash` field: every place that needs it (set_in's collision
    // check, the push-down split) has the entry's own key in hand, so
    // entry_hash() below recomputes it via Hash{} -- a few instructions for
    // IdHash, one rehash for StringHash, and only ever on the O(log32 n)
    // path of an insert/erase, never on a hot read (get_in/leaf_get compare
    // by KEY, not by hash, and never touch this field at all). Caching a
    // value this cheap to rederive just to save the recompute would be
    // paying 8 bytes on EVERY leaf in the trie for a savings that doesn't
    // exist on the paths that matter -- do not add it back.
    //
    // `next` is a unique_ptr, NOT a shared_ptr, on purpose: a shared_ptr
    // member is 16 bytes and would push the (entry, next) leaf into the
    // next glibc size class for every entry in the trie, all to enable
    // tail-sharing that only a length->=2 chain could ever use. Instead,
    // tails are exclusively OWNED by their head, and every chain edit
    // deep-copies the surviving links (see chain_copy) -- do not "optimize"
    // this back to sharing a tail out of an edited chain: without a
    // refcount, two versions pointing at one tail is a use-after-free the
    // moment either is destroyed. The copying is free in practice, for the
    // same reason the sharing was worthless: a chain longer than one link
    // needs a full 64-bit hash collision (impossible for distinct Ids under
    // model::IdHash -- a perfect hash -- and astronomically rare for
    // StringHash), so the chains being copied essentially never have more
    // than the one link that is being edited anyway. Heads stay shared_ptr
    // (in Node::leaves): heads ARE genuinely shared, across every node
    // clone and push-down that path-copying produces.
    //
    // For a Hash that DECLARES itself collision-free (hash_is_perfect_v --
    // e.g. model::IdHash), "impossible" above is exact, not just likely: a
    // chain longer than one link can never legitimately occur, so there is
    // nothing to link to in the first place. NoChain is that case's `next`:
    // zero-sized (via [[no_unique_address]]), and its get() always reports
    // "no tail" -- which is EXACTLY the value a real, empty
    // unique_ptr<const Leaf> would also report, so every generic walker
    // below (leaf_get, each_in, chain_copy's base case, ...) needs no
    // if-constexpr fork to handle both Leaf shapes; only the handful of
    // places that would otherwise WRITE a second link (chain_set/
    // chain_erase) need to know the difference, and they fail loudly
    // (assert) rather than silently drop data if that promise ever turns
    // out to be false at runtime.
    struct Leaf;  // forward declaration: NoChain::get()'s return type needs the name,
                  // not a complete type (it never dereferences it)

    struct NoChain {
        const Leaf* get() const noexcept { return nullptr; }
    };
    static constexpr bool kPerfectHash = hash_is_perfect_v<Hash>;
    using ChainLink = std::conditional_t<kPerfectHash, NoChain, std::unique_ptr<const Leaf>>;

    struct Leaf {
        Leaf(Entry e, ChainLink n) : entry(std::move(e)), next(std::move(n)) {}
        Entry entry;
        [[no_unique_address]] ChainLink next;  ///< collision chain; empty under a perfect Hash
    };

    struct Node {
        std::uint32_t bitmap = 0;
        // Each present bit has a child: either a Node or a Leaf.
        std::vector<std::shared_ptr<const Node>> children;
        std::vector<std::shared_ptr<const Leaf>> leaves;
        // children[k]/leaves[k] correspond positionally; exactly one is non-null
        // per occupied slot. Kept as two parallel vectors for simplicity.
    };

    static std::uint64_t hash_key(const K& k) noexcept {
        return static_cast<std::uint64_t>(Hash{}(k));
    }

    /// The hash a Leaf's entry WOULD have stored, recomputed from its key --
    /// see Leaf's own comment for why nothing caches this.
    static std::uint64_t entry_hash(const Entry& e) noexcept { return hash_key(KeyOf{}(e)); }

    static std::uint32_t slice(std::uint64_t hash, int shift) noexcept {
        return static_cast<std::uint32_t>((hash >> shift) & 0x1f);  // 5 bits
    }
    static std::uint32_t bit(std::uint32_t idx) noexcept { return 1u << idx; }
    static std::uint32_t popcount_below(std::uint32_t bitmap, std::uint32_t idx) noexcept {
        return static_cast<std::uint32_t>(__builtin_popcount(bitmap & (bit(idx) - 1)));
    }

    std::shared_ptr<const Node> root_;
    std::size_t size_ = 0;

    explicit TrieCore(std::shared_ptr<const Node> r, std::size_t n)
        : root_(std::move(r)), size_(n) {}

    // Return a new node equal to `n` but with slot `pos` replaced/inserted.
    static std::shared_ptr<Node> clone_node(const Node* n) {
        auto c = std::make_shared<Node>();
        if (n) {
            c->bitmap = n->bitmap;
            c->children = n->children;
            c->leaves = n->leaves;
        }
        return c;
    }

    static const Entry* leaf_get(const Leaf* lf, const K& key) {
        for (const Leaf* l = lf; l; l = l->next.get())
            if (KeyOf{}(l->entry) == key) return &l->entry;
        return nullptr;
    }

    // Deep-copies a tail chain. Every edited chain copies its surviving
    // links through this rather than sharing them -- see Leaf::next's
    // comment for why sharing is not an option (and why copying is free).
    //
    // Under a perfect Hash there IS no tail to copy -- `l` is structurally
    // always null here (every caller passes `something->next.get()`, and
    // NoChain::get() always returns null) -- so this returns an empty
    // ChainLink unconditionally. The assert is the one place that promise
    // gets checked: if `l` is somehow non-null anyway, hash_is_perfect_v was
    // declared for a Hash that collides in practice, and silently dropping
    // `l`'s entry here would corrupt the trie -- fail loudly instead of
    // doing that quietly (see CLAUDE.md's "prefer failing loudly").
    static ChainLink chain_copy(const Leaf* l) {
        if constexpr (kPerfectHash) {
            assert(!l && "Hash declared is_perfect but produced a real collision");
            return ChainLink{};
        } else {
            if (!l) return nullptr;
            return std::make_unique<Leaf>(l->entry, chain_copy(l->next.get()));
        }
    }

    // Tail-link half of chain_set: `entry` replaces the link whose key
    // matches, or lands in a fresh link at the end (reported via `added`).
    static std::unique_ptr<const Leaf> chain_set_links(const Leaf* l, const K& key,
                                                       const Entry& entry, bool& added) {
        if (!l) {
            added = true;
            return std::make_unique<Leaf>(entry, nullptr);
        }
        if (KeyOf{}(l->entry) == key)
            return std::make_unique<Leaf>(entry, chain_copy(l->next.get()));
        return std::make_unique<Leaf>(l->entry,
                                      chain_set_links(l->next.get(), key, entry, added));
    }

    // Returns `lf`'s chain, rebuilt, with `entry` replacing the link whose
    // key matches or appended as a fresh link if none does (reported via
    // `added`). The head is built with make_shared (Node holds heads by
    // shared_ptr, and the fused control block keeps it one allocation);
    // tail links go through the unique_ptr helpers above.
    //
    // Only ever called (see set_in) when lf_hash == hash. Under a perfect
    // Hash that equality IMPLIES KeyOf{}(lf->entry) == key -- distinct keys
    // can't share a hash -- so this is always a replace of the one entry
    // this slot can ever hold, never an append to a chain that (by
    // construction) never exists.
    static std::shared_ptr<const Leaf> chain_set(const Leaf* lf, const K& key,
                                                 const Entry& entry, bool& added) {
        if constexpr (kPerfectHash) {
            assert(lf && KeyOf{}(lf->entry) == key &&
                  "Hash declared is_perfect but produced a real collision");
            added = false;
            return std::make_shared<Leaf>(entry, NoChain{});
        } else {
            if (!lf) {
                added = true;
                return std::make_shared<Leaf>(entry, nullptr);
            }
            if (KeyOf{}(lf->entry) == key)
                return std::make_shared<Leaf>(entry, chain_copy(lf->next.get()));
            return std::make_shared<Leaf>(lf->entry,
                                          chain_set_links(lf->next.get(), key, entry, added));
        }
    }

    // Tail-link half of chain_erase. Precondition (caller checks via
    // leaf_get): `key` IS present in `l`'s chain -- which is what lets this
    // rebuild unconditionally instead of needing a "key absent, share
    // untouched" path that unique ownership couldn't express anyway.
    static std::unique_ptr<const Leaf> chain_erase_links(const Leaf* l, const K& key) {
        assert(l && "caller verified the key is present in this chain");
        if (KeyOf{}(l->entry) == key) return chain_copy(l->next.get());
        return std::make_unique<Leaf>(l->entry, chain_erase_links(l->next.get(), key));
    }

    // Returns `lf`'s chain, rebuilt without `key`'s link. Same presence
    // precondition as chain_erase_links. A null result means the chain
    // emptied -- the caller drops the slot.
    //
    // Under a perfect Hash, `lf` (found via the same hash slice as `key`)
    // IS the entry for `key` -- same reasoning as chain_set -- so erasing
    // it always empties the slot; there is no tail that could survive.
    static std::shared_ptr<const Leaf> chain_erase(const Leaf* lf, const K& key) {
        assert(lf && "caller verified the key is present in this chain");
        if constexpr (kPerfectHash) {
            assert(KeyOf{}(lf->entry) == key &&
                  "Hash declared is_perfect but produced a real collision");
            return nullptr;
        } else {
            if (KeyOf{}(lf->entry) == key) {
                const Leaf* t = lf->next.get();
                if (!t) return nullptr;  // the chain held only this key
                return std::make_shared<Leaf>(t->entry, chain_copy(t->next.get()));
            }
            return std::make_shared<Leaf>(lf->entry, chain_erase_links(lf->next.get(), key));
        }
    }

    // ---- set ----
    // Returns the new subtree root for the node at `shift`, and reports whether
    // the key was newly inserted (vs replaced) via `added`.
    static std::shared_ptr<const Node> set_in(const Node* n, std::uint64_t hash,
                                              int shift, const K& key,
                                              const Entry& entry, bool& added) {
        const std::uint32_t idx = slice(hash, shift);
        const std::uint32_t b = bit(idx);

        if (!n || !(n->bitmap & b)) {
            // Empty slot: insert a fresh single-entry leaf.
            auto nn = clone_node(n);
            const std::uint32_t pos = popcount_below(nn->bitmap, idx);
            // ChainLink{}, not nullptr: this code path is common to both
            // Leaf shapes (unlike chain_set/chain_erase, it never forks on
            // kPerfectHash), and nullptr has no conversion to NoChain.
            auto lf = std::make_shared<Leaf>(entry, ChainLink{});
            nn->bitmap |= b;
            nn->children.insert(nn->children.begin() + pos, nullptr);
            nn->leaves.insert(nn->leaves.begin() + pos, lf);
            added = true;
            return nn;
        }

        const std::uint32_t pos = popcount_below(n->bitmap, idx);
        auto nn = clone_node(n);

        if (n->children[pos]) {
            // Slot holds a subtree: recurse.
            nn->children[pos] = set_in(n->children[pos].get(), hash, shift + 5, key, entry, added);
            return nn;
        }

        // Slot holds a leaf. Its hash isn't stored -- rederive it from its
        // own entry (see Leaf's comment); this recompute happens at most
        // once per level of an O(log32 n) insert, never on a read path.
        const Leaf* lf = n->leaves[pos].get();
        const std::uint64_t lf_hash = entry_hash(lf->entry);
        if (lf_hash == hash) {
            // Same hash: replace-or-append within the chain (true collision or
            // same key).
            nn->leaves[pos] = chain_set(lf, key, entry, added);
            return nn;
        }

        // Different hash sharing this slot: push the existing leaf down into a
        // new subtree, then insert the new key beside it.
        if (shift + 5 >= 64) {
            // Ran out of hash bits (astronomically unlikely with distinct hashes,
            // but handle it): merge into one collision chain.
            nn->leaves[pos] = std::make_shared<Leaf>(entry, chain_copy(lf));
            added = true;
            return nn;
        }
        auto sub = std::make_shared<Node>();
        const std::uint32_t exist_idx = slice(lf_hash, shift + 5);
        sub->bitmap = bit(exist_idx);
        sub->children.push_back(nullptr);
        sub->leaves.push_back(n->leaves[pos]);
        auto sub2 = set_in(sub.get(), hash, shift + 5, key, entry, added);
        nn->children[pos] = sub2;
        nn->leaves[pos] = nullptr;
        return nn;
    }

    // ---- erase ----
    static std::shared_ptr<const Node> erase_in(const Node* n, std::uint64_t hash,
                                                int shift, const K& key,
                                                bool& removed) {
        if (!n) return nullptr;
        const std::uint32_t idx = slice(hash, shift);
        const std::uint32_t b = bit(idx);
        if (!(n->bitmap & b)) return copy_shared(n);  // not present

        const std::uint32_t pos = popcount_below(n->bitmap, idx);
        auto nn = clone_node(n);

        if (n->children[pos]) {
            auto sub = erase_in(n->children[pos].get(), hash, shift + 5, key, removed);
            if (sub && sub->bitmap != 0) {
                nn->children[pos] = sub;
                return nn;
            }
            // Subtree emptied: drop this slot.
            nn->bitmap &= ~b;
            nn->children.erase(nn->children.begin() + pos);
            nn->leaves.erase(nn->leaves.begin() + pos);
            return nn;
        }

        // Leaf slot.
        const Leaf* lf = n->leaves[pos].get();
        if (!leaf_get(lf, key)) return copy_shared(n);  // key absent

        removed = true;
        auto nl = chain_erase(lf, key);
        if (nl) {
            nn->leaves[pos] = std::move(nl);
            return nn;
        }
        // Chain emptied (it held only this key): remove the slot entirely.
        nn->bitmap &= ~b;
        nn->children.erase(nn->children.begin() + pos);
        nn->leaves.erase(nn->leaves.begin() + pos);
        return nn;
    }

    static std::shared_ptr<const Node> copy_shared(const Node* n) {
        // The node is unchanged; reuse it. We only have a raw pointer here, so we
        // must reconstruct a shared_ptr owner -- but every caller already holds
        // one, so instead of copying we clone. (Called only on the not-present
        // paths, which are rare; correctness over cleverness.)
        return clone_node(n);
    }

    static const Entry* get_in(const Node* n, std::uint64_t hash, int shift,
                               const K& key) {
        while (n) {
            const std::uint32_t idx = slice(hash, shift);
            const std::uint32_t b = bit(idx);
            if (!(n->bitmap & b)) return nullptr;
            const std::uint32_t pos = popcount_below(n->bitmap, idx);
            if (n->children[pos]) {
                n = n->children[pos].get();
                shift += 5;
                continue;
            }
            return leaf_get(n->leaves[pos].get(), key);
        }
        return nullptr;
    }

    template <class F>
    static void each_in(const Node* n, F& f) {
        if (!n) return;
        for (std::size_t i = 0; i < n->children.size(); ++i) {
            if (n->children[i])
                each_in(n->children[i].get(), f);
            else if (n->leaves[i])
                for (const Leaf* l = n->leaves[i].get(); l; l = l->next.get()) f(l->entry);
        }
    }

public:
    TrieCore() = default;

    std::size_t size() const noexcept { return size_; }
    bool empty() const noexcept { return size_ == 0; }

    /// Returns a new core with key -> entry. O(log32 n) node allocations;
    /// shares the rest of the structure with `*this`.
    TrieCore set_entry(const K& key, const Entry& entry) const {
        bool added = false;
        auto r = set_in(root_.get(), hash_key(key), 0, key, entry, added);
        return TrieCore(r, size_ + (added ? 1 : 0));
    }

    /// Returns a new core without `key` (or an equal core if absent).
    TrieCore erase_key(const K& key) const {
        bool removed = false;
        auto r = erase_in(root_.get(), hash_key(key), 0, key, removed);
        return TrieCore(r, size_ - (removed ? 1 : 0));
    }

    const Entry* get_entry(const K& key) const {
        return get_in(root_.get(), hash_key(key), 0, key);
    }

    template <class F>
    void each_entry(F&& f) const {
        each_in(root_.get(), f);
    }
};

}  // namespace detail

template <class K, class V, class Hash>
class PersistentMap {
    using Core = detail::TrieCore<K, std::pair<K, V>, Hash, detail::PairKeyOf<K, V>>;
    Core core_;
    explicit PersistentMap(Core c) : core_(std::move(c)) {}

public:
    PersistentMap() = default;

    std::size_t size() const noexcept { return core_.size(); }
    bool empty() const noexcept { return core_.empty(); }

    /// Returns a new map with key=val. O(log32 n) node allocations; shares the
    /// rest of the structure with `*this`.
    PersistentMap set(const K& key, const V& val) const {
        return PersistentMap(core_.set_entry(key, std::pair<K, V>(key, val)));
    }

    /// Returns a new map without `key` (or an equal map if absent).
    PersistentMap erase(const K& key) const { return PersistentMap(core_.erase_key(key)); }

    const V* get(const K& key) const {
        const std::pair<K, V>* e = core_.get_entry(key);
        return e ? &e->second : nullptr;
    }

    template <class F>
    void for_each(F&& f) const {
        core_.each_entry([&](const std::pair<K, V>& e) { f(e.first, e.second); });
    }
};

/// A persistent SET: like PersistentMap, but there is no value -- the key IS
/// the entry, so no per-entry storage is spent duplicating what the key
/// already encodes. Used wherever a PersistentMap's value was always just a
/// copy of its own key (Root::by_type, and the inner bucket of
/// Root::by_cached_field/Root::by_cached_reference -- see model.h's Root).
template <class K, class Hash>
class PersistentSet {
    using Core = detail::TrieCore<K, K, Hash, detail::IdentityKeyOf<K>>;
    Core core_;
    explicit PersistentSet(Core c) : core_(std::move(c)) {}

public:
    PersistentSet() = default;

    std::size_t size() const noexcept { return core_.size(); }
    bool empty() const noexcept { return core_.empty(); }

    /// Returns a new set with `key` present. O(log32 n) node allocations;
    /// shares the rest of the structure with `*this`.
    PersistentSet insert(const K& key) const { return PersistentSet(core_.set_entry(key, key)); }

    /// Returns a new set without `key` (or an equal set if absent).
    PersistentSet erase(const K& key) const { return PersistentSet(core_.erase_key(key)); }

    bool contains(const K& key) const { return core_.get_entry(key) != nullptr; }

    template <class F>
    void for_each(F&& f) const {
        core_.each_entry([&](const K& k) { f(k); });
    }
};

}  // namespace model::pmap
