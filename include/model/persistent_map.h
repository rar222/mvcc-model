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

#include <cstdint>
#include <memory>
#include <string>
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
    // whole per-entry footprint. The chain is immutable like everything
    // else here: "editing" one rebuilds the links up to the edit point and
    // shares the tail (see chain_set/chain_erase).
    struct Leaf {
        Leaf(std::uint64_t h, Entry e, std::shared_ptr<const Leaf> n)
            : hash(h), entry(std::move(e)), next(std::move(n)) {}
        std::uint64_t hash;
        Entry entry;
        std::shared_ptr<const Leaf> next;  ///< collision chain; almost always null
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

    // Returns `lf`'s chain with `entry` replacing the link whose key matches,
    // or appended as a fresh link if none does (reported via `added`). Links
    // before the edit point are copied; everything after it is shared.
    static std::shared_ptr<const Leaf> chain_set(const std::shared_ptr<const Leaf>& lf,
                                                 std::uint64_t hash, const K& key,
                                                 const Entry& entry, bool& added) {
        if (!lf) {
            added = true;
            return std::make_shared<Leaf>(hash, entry, nullptr);
        }
        if (KeyOf{}(lf->entry) == key)
            return std::make_shared<Leaf>(lf->hash, entry, lf->next);
        return std::make_shared<Leaf>(lf->hash, lf->entry,
                                      chain_set(lf->next, hash, key, entry, added));
    }

    // Returns `lf`'s chain with the link whose key matches removed (reported
    // via `removed`; the chain is shared untouched if the key is absent).
    // A null result means the chain emptied -- the caller drops the slot.
    static std::shared_ptr<const Leaf> chain_erase(const std::shared_ptr<const Leaf>& lf,
                                                   const K& key, bool& removed) {
        if (!lf) return nullptr;
        if (KeyOf{}(lf->entry) == key) {
            removed = true;
            return lf->next;
        }
        auto tail = chain_erase(lf->next, key, removed);
        if (!removed) return lf;  // key absent below: share the whole chain as-is
        return std::make_shared<Leaf>(lf->hash, lf->entry, std::move(tail));
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
            auto lf = std::make_shared<Leaf>(hash, entry, nullptr);
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

        // Slot holds a leaf.
        const Leaf* lf = n->leaves[pos].get();
        if (lf->hash == hash) {
            // Same hash: replace-or-append within the chain (true collision or
            // same key).
            nn->leaves[pos] = chain_set(n->leaves[pos], hash, key, entry, added);
            return nn;
        }

        // Different hash sharing this slot: push the existing leaf down into a
        // new subtree, then insert the new key beside it.
        if (shift + 5 >= 64) {
            // Ran out of hash bits (astronomically unlikely with distinct hashes,
            // but handle it): merge into one collision chain.
            nn->leaves[pos] = std::make_shared<Leaf>(hash, entry, n->leaves[pos]);
            added = true;
            return nn;
        }
        auto sub = std::make_shared<Node>();
        const std::uint32_t exist_idx = slice(lf->hash, shift + 5);
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
        bool chain_removed = false;
        auto nl = chain_erase(n->leaves[pos], key, chain_removed);
        if (!chain_removed) return copy_shared(n);  // key absent

        removed = true;
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
