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

#include <array>
#include <atomic>
#include <cassert>
#include <cstdint>
#include <iterator>
#include <memory>
#include <memory_resource>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>


namespace model::pmap {

/// Hasher for a std::string key. FNV-1a, 64-bit -- adequate for dispersing
/// string keys across the trie. A named, reusable functor (rather than a
/// private method baked into the trie) so it can be passed as the explicit
/// Hash argument wherever a PersistentMap is keyed by a real string
/// (Root::by_key, and the OUTER map of Root::by_cached_field) -- see
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

/// Node::slots capacity() vs size() summed across every Node in one trie --
/// see TrieCore::slot_stats. Namespace-scope (not nested in TrieCore) so
/// every instantiation (by_type_'s Id-keyed set, by_key_'s string-keyed
/// map, ...) reports through the same concrete type, poolable into one
/// Model-level diagnostic without a per-instantiation type each.
struct SlotStats {
    std::size_t node_count = 0;
    std::size_t leaf_count = 0;
    std::size_t total_capacity = 0;  ///< sum of slots.capacity() across every Node
    std::size_t total_size = 0;      ///< sum of slots.size() across every Node (== child count)
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
    // than the one link that is being edited anyway. Heads stay RcHandle-
    // backed (in Node::slots): heads ARE genuinely shared, across every
    // node clone and push-down that path-copying produces.
    //
    // For a Hash that DECLARES itself collision-free (hash_is_perfect_v --
    // e.g. model::IdHash), "impossible" above is exact, not just likely: a
    // chain longer than one link can never legitimately occur, so there is
    // nothing to link to in the first place. NoChain is that case's `next`:
    // zero-sized (via [[no_unique_address]] -- see Leaf's own comment for
    // what that attribute actually buys here), and its get() always reports
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
    struct Node;  // forward declaration: RcHandle::release() names it, but only in its
                  // out-of-line definition below (after Node/Leaf are complete) -- the
                  // declaration inside RcHandle's own class body needs neither name.

    // One shared refcounted header, embedded as the FIRST member of both Node and
    // Leaf (composition, not a base class -- see RcHandle's own comment for why).
    // [basic.compound]/[class.mem] guarantee that a pointer to a standard-layout
    // class, reinterpret_cast to a pointer to its first member, is the SAME address
    // (and vice versa) -- that's what lets RcHandle read/decrement the count and
    // read is_leaf_self without first knowing whether the pointee is a Node or a
    // Leaf. is_leaf_self/owner_mem are set once, at construction, strictly before
    // the object is ever shared (see alloc_node/alloc_leaf below) -- never mutated
    // again, so no synchronization is needed to read them from another thread
    // later: the same "immutable after construction, safe once published" property
    // every other field this project publishes to readers already relies on.
    struct RcBase {
        std::atomic<std::uint32_t> refcount{1};
        bool is_leaf_self = false;
        std::pmr::memory_resource* owner_mem = nullptr;  ///< nullptr => plain new/delete
    };

    // Intrusively refcounted, type-erased handle: ONE raw pointer (8 bytes) --
    // replacing shared_ptr<const void>'s 16 (an object pointer PLUS a pointer to a
    // separate control block). The refcount, the Node-or-Leaf tag, and the owning
    // pmr resource all live in the pointee's own RcBase instead of a side
    // allocation, so the allocation count per Node/Leaf is unchanged (RcBase is a
    // plain member, not a second `new`) -- the saving is purely in the HANDLE
    // stored at every Node::slots element and at TrieCore::root_, which is what
    // this project has many more of than it has actual Node/Leaf objects, any time
    // more than one published Root (or a live reader's snapshot, or an open
    // Transaction's base) still shares a subtree. Copy: one relaxed atomic
    // increment. Last release: one acq_rel decrement, and only on the thread that
    // actually observes the count hit zero, the real destroy+deallocate -- same
    // thread-safety contract shared_ptr already gave this design, which is what
    // set_in's mutate-in-place fast path (see its own long comment) depends on via
    // use_count().
    class RcHandle {
    public:
        RcHandle() noexcept = default;
        RcHandle(std::nullptr_t) noexcept {}

        /// Wraps a FRESHLY allocated Node/Leaf -- RcBase's own default member
        /// initializer already left refcount at 1, so this never increments.
        static RcHandle adopt(void* p) noexcept {
            RcHandle h;
            h.ptr_ = p;
            return h;
        }

        RcHandle(const RcHandle& o) noexcept : ptr_(o.ptr_) {
            if (ptr_) base()->refcount.fetch_add(1, std::memory_order_relaxed);
        }
        RcHandle(RcHandle&& o) noexcept : ptr_(o.ptr_) { o.ptr_ = nullptr; }
        RcHandle& operator=(const RcHandle& o) noexcept {
            if (this != &o) {
                RcHandle tmp(o);
                swap(tmp);
            }
            return *this;
        }
        RcHandle& operator=(RcHandle&& o) noexcept {
            if (this != &o) {
                release();
                ptr_ = o.ptr_;
                o.ptr_ = nullptr;
            }
            return *this;
        }
        ~RcHandle() { release(); }

        void swap(RcHandle& o) noexcept { std::swap(ptr_, o.ptr_); }
        void* get() const noexcept { return ptr_; }
        explicit operator bool() const noexcept { return ptr_ != nullptr; }

        /// Relaxed load -- same "approximate outside external synchronization"
        /// contract as shared_ptr::use_count(); see set_in's can_mutate check for
        /// the one place this project actually depends on that contract.
        std::uint32_t use_count() const noexcept {
            return ptr_ ? base()->refcount.load(std::memory_order_relaxed) : 0;
        }

    private:
        RcBase* base() const noexcept { return reinterpret_cast<RcBase*>(ptr_); }
        void release() noexcept;  // out-of-line: needs Node and Leaf complete, see below

        void* ptr_ = nullptr;
    };

    struct NoChain {
        const Leaf* get() const noexcept { return nullptr; }
    };
    static constexpr bool kPerfectHash = hash_is_perfect_v<Hash>;
    using ChainLink = std::conditional_t<kPerfectHash, NoChain, std::unique_ptr<const Leaf>>;

    struct Leaf {
        RcBase rc;  // must be the first member -- see RcBase's own comment
        Leaf(Entry e, ChainLink n) : entry(std::move(e)), next(std::move(n)) {}
        Entry entry;

        /// Collision chain; ChainLink is std::unique_ptr<const Leaf> under a
        /// non-perfect Hash (a real 8-byte pointer -- the attribute is a
        /// no-op there, nothing empty to shrink), or NoChain under a
        /// perfect one.
        ///
        /// [[no_unique_address]] (C++20) tells the compiler this member
        /// doesn't need its own distinct address the way every object
        /// normally must -- so when ChainLink is the empty NoChain, the
        /// compiler may overlap its storage with `entry` instead of the
        /// usual >=1-byte-plus-padding an empty member would otherwise cost
        /// (every object, even an empty one, still needs a UNIQUE address
        /// by default -- sizeof(NoChain) is 1, not 0 -- unless told
        /// otherwise). Same idea as the empty-base-optimization trick, just
        /// for an ordinary member instead of a base class. The payoff: for
        /// a perfect Hash (model::IdHash) a Leaf costs exactly what Entry
        /// alone costs -- zero overhead for a chain link that, by
        /// construction, can never legitimately point anywhere -- which
        /// matters here because TrieCore/Leaf backs every persistent index
        /// this project's Model maintains at 100k-1M objects. Same
        /// cost-consciousness as the two comments above (no cached `hash`
        /// field; unique_ptr, not shared_ptr, for the real chain case).
        /// Under C++17 the attribute doesn't exist at all - but it compiles
        /// just without the size optimization -- correctness is unaffected
        /// either way, since NoChain::get() always returns null regardless
        /// of where it's stored.
#if defined(__has_cpp_attribute) && __has_cpp_attribute(no_unique_address) >= 201803L
        [[no_unique_address]] ChainLink next;
#else
        ChainLink next;
#endif
    };
    static_assert(std::is_standard_layout_v<Leaf>,
                  "Leaf must stay standard-layout -- RcHandle::release() reinterpret_casts a "
                  "Leaf* to RcBase* via the first-member guarantee; see RcBase's own comment. "
                  "If some Entry type ever breaks this, that is a real problem to fix in Leaf's "
                  "layout (e.g. give Entry its own indirection), not a check to weaken.");

    // Each occupied bit in `bitmap` has exactly one child, either a Node or
    // a Leaf -- so two parallel vectors, one always empty per occupied slot,
    // would waste storage and a second heap buffer per node for nothing.
    // `slots` holds ONE type-erased RcHandle per occupied slot instead, and
    // `is_leaf`, a second bitmap parallel to (a subset of) `bitmap` and
    // indexed the SAME way (by idx, 0-31, not by the compacted vector
    // position), records which concrete type a slot holds. A slot's actual
    // type is always known before it's dereferenced (every caller already
    // branches on is_leaf first), so the cast back
    // (static_cast<const Node*>/static_cast<const Leaf*>) is never
    // ambiguous -- RcBase itself is ALSO self-describing
    // (RcBase::is_leaf_self), which is what lets RcHandle::release() free
    // the right type even where no sibling is_leaf bit is in scope (e.g.
    // releasing TrieCore::root_). See RcHandle's own comment (just above
    // Leaf's forward declaration) for the full design.
    // Why 32-bit (32-ary, 5-bit slices), not 64-bit (64-ary, 6-bit slices),
    // at this project's target scale (100k-1M objects, see the file header):
    // widening the bitmap looks like "half as many levels to path-copy per
    // commit," but it isn't, in this range. Depth is ceil(log_32 n) vs.
    // ceil(log_64 n): at n=100,000 that's 4 vs. 3 (one level saved), but at
    // n=1,000,000 -- the TOP of the target range -- it's 4 vs. 4 (32^4 =
    // 1,048,576, so 32-ary doesn't even need a 5th level yet; 64-ary only
    // pulls ahead past 64^4 = 16.7M, well outside this project's stated
    // scale). A marginal, inconsistent depth win at best across the actual
    // range this trie is sized for.
    //
    // Meanwhile it makes the cost that actually matters WORSE: the file
    // header's own stated goal is path-copying "only O(log32 n) NODES" per
    // commit, but each node copy isn't free -- `slots` is a
    // vector<RcHandle>, so copying a node costs one RcHandle copy (one
    // atomic increment) per populated child. 64-ary raises the worst case
    // from 32 RcHandles per node to 64, for the same or greater depth
    // across this range -- strictly worse for the operation this whole
    // structure exists to keep cheap, not better.
    //
    // It would also cost real complexity, not just a wider type: bit()'s
    // `1u << idx` becomes UB for idx>=32 (needs `1ull << idx`);
    // popcount_below's __builtin_popcount (32-bit) would need
    // __builtin_popcountll; and slice()'s 5-bit extraction plus the
    // hash-exhaustion depth check in set_in (`shift + 5 >= 64`) would both
    // need re-deriving around 6-bit slices and a different max-depth
    // constant. 32-ary is also the standard choice in this space for
    // exactly this reason (Clojure's PersistentHashMap, Scala's HashMap,
    // immer) -- not an arbitrary pick that happened to land here. Do not
    // "widen" this without new numbers showing this project's scale target
    // has actually grown past where 32-ary stops being enough.
    struct Node {
        RcBase rc;  // must be the first member -- see RcBase's own comment
        std::uint32_t bitmap = 0;
        std::uint32_t is_leaf = 0;  ///< subset of bitmap: which occupied idx-slots hold a Leaf
        std::vector<RcHandle> slots;  ///< positionally compacted, idx order
    };
    static_assert(std::is_standard_layout_v<Node>,
                  "Node must stay standard-layout -- RcHandle::release() reinterpret_casts a "
                  "Node* to RcBase* via the first-member guarantee; see RcBase's own comment");

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

    // RcHandle, already type-erased -- see RcHandle's own comment for the
    // full design. Passing root_ by const reference into set_in/erase_in
    // costs nothing beyond that reference: RcHandle never implicitly
    // converts between a "typed" and an "erased" form (there is only the
    // one, always-erased form), so there is no equivalent of a
    // shared_ptr<Node>-to-shared_ptr<const void> conversion that could
    // construct a temporary and skew use_count() -- the bug the old
    // shared_ptr<const void> choice existed specifically to avoid.
    RcHandle root_;

    // uint32_t, not size_t: every TrieCore instance counts some subset of
    // the model's live objects or distinct field values, so it's bounded by
    // the SAME 2^31 ceiling Id::index already imposes (model.h's own
    // comment on Id: "index gives 2^31 usable slots... Do not 'future-
    // proof' this"). A uint32_t ceiling (2^32) sits above that structural
    // limit, so Id::index exhausts first -- size_ can never be the binding
    // constraint. Narrowing it (paired with root_is_leaf_ right below)
    // costs nothing: the 3 bytes of alignment padding before mem_ (which
    // still needs 8-byte alignment) absorb both fields for free -- see the
    // static_assert below TrieCore's members.
    std::uint32_t size_ = 0;

    // Whether `root_` is a bare Leaf rather than a Node -- the fast path
    // for a trie holding exactly one entry (or, under a non-perfect Hash,
    // one hash-colliding chain), which otherwise pays for a whole Node
    // (bitmap + is_leaf + a vector<RcHandle>) just to hold a
    // single child. Every entry point that walks from root_ (set_entry,
    // erase_key, get_entry, each_entry/each_entry_short_circuit,
    // slot_stats, begin()) branches on this before falling back to the
    // existing as_node(root_)-based path; set_in/erase_in themselves are
    // never called with a leaf root -- see set_entry/erase_key.
    bool root_is_leaf_ = false;

    /// Where this instance's Node/Leaf HEAD allocations come
    /// from (the RcHandle ones stored in Node::slots -- never the
    /// unique_ptr tail links, which stay on plain new; see chain_copy/
    /// chain_set_links/chain_erase_links). nullptr (the default, and every
    /// existing caller's behavior -- persistent_map_tests.cpp included) means
    /// "use ordinary plain new/delete".
    ///
    /// Profiling one_million_random_basic_records_in_a_single_transaction_
    /// without_undo (examples/basic_record_bench.cpp) found cloning
    /// by_type_'s Node chain -- Node/Leaf allocation plus the RcHandle
    /// vector copy/dispose that comes with it -- to
    /// be the single largest cost of a commit, ahead of the model's own
    /// apply logic: ~27% of instructions, and a LARGER share of last-level
    /// cache misses once measured with cache simulation (callgrind
    /// --cache-sim=yes), since a fresh malloc'd Node/Leaf is by definition
    /// cold memory.
    ///
    /// Deliberately a raw, non-owning pointer, NOT a shared_ptr: unlike
    /// std::allocate_shared's own control block (which copies whatever
    /// allocator it's given, and would silently keep a bare
    /// memory_resource* alive-by-reference nowhere), the LIFETIME
    /// obligation here is pushed onto the caller who supplies the pointer
    /// -- see Model::node_pool_'s own comment for how Model discharges it
    /// (by construction order, not refcounting). set_entry()/erase_key()
    /// carry `mem_` forward from `*this` into every derived TrieCore, so
    /// once a lineage is seeded with a resource (Model does this the FIRST
    /// time a given type/field's index is touched -- see e.g. apply_create's
    /// by_type_[tag] seeding in model.cpp), every later insert/erase for
    /// that SAME lineage keeps using it automatically. Constant for the
    /// whole of any one set_entry()/erase_key() call (and everything it
    /// recurses into), which is what lets clone_node/make_leaf/set_in/
    /// erase_in/chain_set/chain_erase below read it straight off `this`
    /// instead of threading it through as an extra parameter on every
    /// recursive call.
    std::pmr::memory_resource* mem_ = nullptr;

    explicit TrieCore(RcHandle r, std::uint32_t n, std::pmr::memory_resource* mem, bool leaf_root)
        : root_(std::move(r)), size_(n), root_is_leaf_(leaf_root), mem_(mem) {
        // Compiler-checked, not just reasoned about: root_/size_/
        // root_is_leaf_/mem_ pack into exactly 24 bytes (root_ 8B + size_
        // 4B + root_is_leaf_ 1B + 3B alignment padding before mem_'s 8B --
        // see size_/root_is_leaf_'s own comments). sizeof(TrieCore) needs a
        // complete-class context, which a constructor body is and the
        // member-declaration region above is not -- hence checking it here
        // rather than immediately after the members.
        static_assert(sizeof(TrieCore) == 24,
                      "root_/size_/root_is_leaf_/mem_ no longer pack into 24B -- see their own "
                      "comments before adding padding back");
    }

    // mem_ == nullptr => plain `new`; otherwise routed through that resource
    // via `mem_->allocate` + placement-new. One allocation each, same as
    // before -- RcBase costs nothing extra to allocate, it's a plain member.
    Node* alloc_node() const {
        Node* n = mem_ ? ::new (mem_->allocate(sizeof(Node), alignof(Node))) Node() : new Node();
        n->rc.owner_mem = mem_;  // is_leaf_self stays false, RcBase's own default -- correct for Node
        return n;
    }
    Leaf* alloc_leaf(Entry e, ChainLink n) const {
        Leaf* lf = mem_ ? ::new (mem_->allocate(sizeof(Leaf), alignof(Leaf)))
                               Leaf(std::move(e), std::move(n))
                       : new Leaf(std::move(e), std::move(n));
        lf->rc.is_leaf_self = true;
        lf->rc.owner_mem = mem_;
        return lf;
    }

    // Return a new node equal to `n` but with slot `pos` replaced/inserted.
    // Returns the raw, exclusively-owned Node* rather than an RcHandle:
    // every caller mutates bitmap/is_leaf/slots on it AFTER this returns, so
    // wrapping it here would be immediately unwrapped again -- callers wrap
    // it in an RcHandle via RcHandle::adopt() exactly once, at whichever
    // return statement finalizes it.
    Node* clone_node(const Node* n) const {
        Node* c = alloc_node();
        if (n) {
            c->bitmap = n->bitmap;
            c->is_leaf = n->is_leaf;
            c->slots = n->slots;
        }
        return c;
    }

    static const Node* as_node(const RcHandle& s) noexcept {
        return static_cast<const Node*>(s.get());
    }
    static const Leaf* as_leaf(const RcHandle& s) noexcept {
        return static_cast<const Leaf*>(s.get());
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
    // `added`). The head is built via alloc_leaf (Node holds heads by
    // RcHandle, whose refcount lives inline in the Leaf itself -- one
    // allocation, same as before); tail links go through the unique_ptr
    // helpers above.
    //
    // Only ever called (see set_in) when lf_hash == hash. Under a perfect
    // Hash that equality IMPLIES KeyOf{}(lf->entry) == key -- distinct keys
    // can't share a hash -- so this is always a replace of the one entry
    // this slot can ever hold, never an append to a chain that (by
    // construction) never exists.
    RcHandle make_leaf(Entry e, ChainLink n) const {
        return RcHandle::adopt(alloc_leaf(std::move(e), std::move(n)));
    }

    RcHandle chain_set(const Leaf* lf, const K& key, const Entry& entry, bool& added) const {
        if constexpr (kPerfectHash) {
            assert(lf && KeyOf{}(lf->entry) == key &&
                  "Hash declared is_perfect but produced a real collision");
            added = false;
            return make_leaf(entry, NoChain{});
        } else {
            if (!lf) {
                added = true;
                return make_leaf(entry, nullptr);
            }
            if (KeyOf{}(lf->entry) == key) return make_leaf(entry, chain_copy(lf->next.get()));
            return make_leaf(lf->entry, chain_set_links(lf->next.get(), key, entry, added));
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
    RcHandle chain_erase(const Leaf* lf, const K& key) const {
        assert(lf && "caller verified the key is present in this chain");
        if constexpr (kPerfectHash) {
            assert(KeyOf{}(lf->entry) == key &&
                  "Hash declared is_perfect but produced a real collision");
            return nullptr;
        } else {
            if (KeyOf{}(lf->entry) == key) {
                const Leaf* t = lf->next.get();
                if (!t) return nullptr;  // the chain held only this key
                return make_leaf(t->entry, chain_copy(t->next.get()));
            }
            return make_leaf(lf->entry, chain_erase_links(lf->next.get(), key));
        }
    }

    // ---- set ----
    // Returns the new subtree root for the node currently held by `owner`
    // (aliasing a Node, or a null RcHandle for an empty subtree at this
    // position), and reports whether the key was newly inserted (vs
    // replaced) via `added`.
    //
    // MUTATE-IN-PLACE FAST PATH: `parent_private` says every ancestor from
    // the true root down to (but not including) `owner` has ALREADY been
    // confirmed exclusive to this call chain -- the top-level caller
    // (set_entry) passes true (there is no ancestor to worry about), and
    // every recursive call below passes down its OWN can_mutate decision,
    // never a freshly-derived one.
    //
    // `owner.use_count() == 1` is necessary but NOT sufficient on its own:
    // if `pb = pa` copies a whole map/set, the shared TOP node's refcount
    // correctly reads 2 and forces a clone there -- but each of ITS
    // children is still referenced by exactly ONE RcHandle (the single
    // shared top node's own slots vector), so a child's OWN use_count()
    // reads 1 even though it is only reachable through that still-shared
    // parent, and mutating it in place would corrupt what `pa` sees through
    // the very same parent -- persistent_map_tests.cpp's
    // perfect_hash_map_persists_old_version_across_derived_edits guards
    // exactly this scenario. So a node is only eligible to be mutated in
    // place when BOTH hold: its ancestors are already private
    // (parent_private), AND it itself isn't additionally referenced
    // (use_count() == 1).
    //
    // Model's per-index tries are only ever touched from inside
    // try_commit(), under commit_mu_ (see model.h's invariant 7), and are
    // only ever exposed to a reader by publish_now()'s atomic Root swap,
    // strictly after every insert in the transaction has run -- so a node
    // that passes BOTH checks genuinely cannot be observed by anything
    // else, at any point before this function returns and replaces it.
    // That's what turns clone_node's O(fanout) cost (a fresh Node/Leaf
    // allocation plus a full RcHandle vector copy -- measured as the single
    // largest cost of a commit, see clone_node's own comment)
    // into an O(1) in-place field write for every touch after the first of
    // a given node within the SAME apply -- e.g. every insert past the
    // first into the same bulk-create transaction's by_type_ set.
    //
    // A single-item transaction never sees use_count() drop to 1 at the
    // root: its one insert touches nodes still referenced by the
    // previously-published Root (or, for a brand-new tag's first touch,
    // additionally by the rollback capture -- see log_by_type_once's doc
    // comment in model.cpp), so it still clones every node on its path --
    // this in-place optimization can only ever remove clones for a
    // MULTI-insert transaction revisiting the same node, never add cost to
    // a single-insert one (see small_txn_bench.cpp, which exists to catch
    // exactly this class of regression).
    //
    // Ordering is what makes the use_count() check correct even with
    // parent_private threaded through: every branch below computes the
    // replacement child/leaf value FIRST, using owner's ORIGINAL
    // (not-yet-cloned) children, and decides mutate-vs-clone for THIS level
    // only afterward. clone_node bumps every child's refcount (it copies
    // the whole slots vector), so checking a child's use_count() AFTER its
    // parent was cloned would see the clone's extra reference and wrongly
    // report "shared" for a child that was still private one statement
    // earlier.
    RcHandle set_in(const RcHandle& owner, bool parent_private, std::uint64_t hash, int shift,
                    const K& key, const Entry& entry, bool& added) const {
        const Node* n = as_node(owner);
        const std::uint32_t idx = slice(hash, shift);
        const std::uint32_t b = bit(idx);
        const bool can_mutate = parent_private && n && owner.use_count() == 1;

        if (!n || !(n->bitmap & b)) {
            // Empty slot: insert a fresh single-entry leaf.
            const std::uint32_t pos = n ? popcount_below(n->bitmap, idx) : 0;
            // ChainLink{}, not nullptr: this code path is common to both
            // Leaf shapes (unlike chain_set/chain_erase, it never forks on
            // kPerfectHash), and nullptr has no conversion to NoChain.
            auto lf = make_leaf(entry, ChainLink{});
            added = true;
            if (can_mutate) {
                Node* mut = const_cast<Node*>(n);
                mut->bitmap |= b;
                mut->is_leaf |= b;
                // The target size is always known exactly (current + 1), so
                // reserve() it directly rather than let insert() fall back
                // to vector's growth-doubling, which has no notion of this
                // vector's 32-element ceiling.
                mut->slots.reserve(mut->slots.size() + 1);
                mut->slots.insert(mut->slots.begin() + pos, std::move(lf));
                return owner;
            }
            Node* nn = clone_node(n);
            nn->bitmap |= b;
            nn->is_leaf |= b;
            nn->slots.reserve(nn->slots.size() + 1);
            nn->slots.insert(nn->slots.begin() + pos, std::move(lf));
            return RcHandle::adopt(nn);
        }

        const std::uint32_t pos = popcount_below(n->bitmap, idx);

        if (!(n->is_leaf & b)) {
            // Slot holds a subtree: recurse using the ORIGINAL child (not a
            // clone's copy of it), passing THIS level's own can_mutate as
            // the child's parent_private -- see this function's doc
            // comment on why an isolated child use_count() check is not
            // enough on its own.
            auto new_child = set_in(n->slots[pos], can_mutate, hash, shift + 5, key, entry, added);
            if (can_mutate) {
                const_cast<Node*>(n)->slots[pos] = std::move(new_child);
                return owner;
            }
            Node* nn = clone_node(n);
            nn->slots[pos] = std::move(new_child);
            return RcHandle::adopt(nn);
        }

        // Slot holds a leaf. Its hash isn't stored -- rederive it from its
        // own entry (see Leaf's comment); this recompute happens at most
        // once per level of an O(log32 n) insert, never on a read path.
        const Leaf* lf = as_leaf(n->slots[pos]);
        const std::uint64_t lf_hash = entry_hash(lf->entry);
        if (lf_hash == hash) {
            // Same hash: replace-or-append within the chain (true collision or
            // same key).
            auto new_slot = chain_set(lf, key, entry, added);
            if (can_mutate) {
                const_cast<Node*>(n)->slots[pos] = std::move(new_slot);
                return owner;
            }
            Node* nn = clone_node(n);
            nn->slots[pos] = std::move(new_slot);
            return RcHandle::adopt(nn);
        }

        // Different hash sharing this slot: push the existing leaf down into a
        // new subtree, then insert the new key beside it.
        if (shift + 5 >= 64) {
            // Ran out of hash bits (astronomically unlikely with distinct hashes,
            // but handle it): merge into one collision chain.
            auto new_slot = make_leaf(entry, chain_copy(lf));
            added = true;
            if (can_mutate) {
                const_cast<Node*>(n)->slots[pos] = std::move(new_slot);
                return owner;
            }
            Node* nn = clone_node(n);
            nn->slots[pos] = std::move(new_slot);
            return RcHandle::adopt(nn);
        }
        Node* sub = clone_node(nullptr);
        const std::uint32_t exist_idx = slice(lf_hash, shift + 5);
        sub->bitmap = bit(exist_idx);
        sub->is_leaf = bit(exist_idx);
        sub->slots.push_back(n->slots[pos]);
        // `sub` is a raw, exclusively-owned Node* here -- adopting it below
        // costs nothing (no refcount to bump; RcBase's own count starts at
        // 1). parent_private is true unconditionally in the recursive call
        // (not `can_mutate`): `sub` is freshly allocated by THIS call no
        // matter whether the current level itself is shared, so it starts a
        // brand new private lineage regardless.
        RcHandle sub_owner = RcHandle::adopt(sub);
        auto sub2 = set_in(sub_owner, /*parent_private=*/true, hash, shift + 5, key, entry, added);
        if (can_mutate) {
            Node* mut = const_cast<Node*>(n);
            mut->slots[pos] = std::move(sub2);
            mut->is_leaf &= ~b;  // this slot now holds a Node, not a Leaf
            return owner;
        }
        Node* nn = clone_node(n);
        nn->slots[pos] = std::move(sub2);
        nn->is_leaf &= ~b;  // this slot now holds a Node, not a Leaf
        return RcHandle::adopt(nn);
    }

    // ---- erase ----
    // NOTE: when a subtree under a Node slot shrinks to a single child, that
    // child is kept as a one-entry Node one level down rather than being
    // collapsed back into a Leaf occupying the parent's slot directly (unlike
    // some HAMT implementations, which do path-collapse on erase). This is
    // simpler and still correct -- get_in/each_in walk through a one-child
    // Node exactly like any other -- but a key set that inserts and erases
    // heavily around the same hash prefixes can leave the trie permanently
    // one level deeper than the live entry count alone would need. Not a
    // correctness issue, just a depth (and therefore path-copy cost) that
    // never shrinks back down on its own.
    // Takes (and, on a no-op, returns) the OWNING RcHandle -- not just a raw
    // Node* -- for the same reason set_in does: it's what lets a "key not
    // found" result propagate back up as `return owner` (a cheap RcHandle
    // copy, one atomic refcount bump) instead of cloning every node on the
    // way down before discovering, at the bottom, that nothing needed to
    // change. Each level below detects "no-op" by pointer-comparing its
    // child's result against the child it passed in: since a genuine no-op
    // returns that same owner unchanged, `.get()` equality is exact, not a
    // heuristic. (An erase() on a key that IS present still clones every
    // node on its path -- same O(log32 n) cost as set() -- this only removes
    // the cost for the no-op case, which used to pay that same price for
    // nothing: see persistent_map_tests.cpp's erase_absent_key_* tests.)
    RcHandle erase_in(const RcHandle& owner, std::uint64_t hash, int shift, const K& key,
                      bool& removed) const {
        const Node* n = as_node(owner);
        if (!n) return owner;  // empty subtree: nothing to erase
        const std::uint32_t idx = slice(hash, shift);
        const std::uint32_t b = bit(idx);
        if (!(n->bitmap & b)) return owner;  // not present at this level: unchanged

        const std::uint32_t pos = popcount_below(n->bitmap, idx);

        if (!(n->is_leaf & b)) {
            auto sub = erase_in(n->slots[pos], hash, shift + 5, key, removed);
            if (sub.get() == n->slots[pos].get()) return owner;  // unchanged below: unchanged here too
            Node* nn = clone_node(n);
            const Node* subnode = as_node(sub);
            if (subnode && subnode->bitmap != 0) {
                nn->slots[pos] = std::move(sub);
                return RcHandle::adopt(nn);
            }
            // Subtree emptied: drop this slot.
            nn->bitmap &= ~b;
            nn->is_leaf &= ~b;
            nn->slots.erase(nn->slots.begin() + pos);
            return RcHandle::adopt(nn);
        }

        // Leaf slot.
        const Leaf* lf = as_leaf(n->slots[pos]);
        if (!leaf_get(lf, key)) return owner;  // key absent: unchanged

        removed = true;
        auto nl = chain_erase(lf, key);
        Node* nn = clone_node(n);
        if (nl) {
            nn->slots[pos] = std::move(nl);
            return RcHandle::adopt(nn);
        }
        // Chain emptied (it held only this key): remove the slot entirely.
        nn->bitmap &= ~b;
        nn->is_leaf &= ~b;
        nn->slots.erase(nn->slots.begin() + pos);
        return RcHandle::adopt(nn);
    }

    static const Entry* get_in(const Node* n, std::uint64_t hash, int shift,
                               const K& key) {
        while (n) {
            const std::uint32_t idx = slice(hash, shift);
            const std::uint32_t b = bit(idx);
            if (!(n->bitmap & b)) return nullptr;
            const std::uint32_t pos = popcount_below(n->bitmap, idx);
            if (!(n->is_leaf & b)) {
                n = as_node(n->slots[pos]);
                shift += 5;
                continue;
            }
            return leaf_get(as_leaf(n->slots[pos]), key);
        }
        return nullptr;
    }

    // Walks idx 0..31 (not the compacted slot position directly) because
    // is_leaf is indexed by idx, same as bitmap -- `pos` tracks the matching
    // position in `slots` alongside as occupied bits are found, the same
    // relationship popcount_below computes on demand elsewhere.
    //
    // A thin wrapper over each_in_short_circuit (f wrapped to always report
    // "keep going") rather than a second copy of the walk -- the wrapper
    // lambda's `return true` is a compile-time constant, so at -O2 (the
    // default preset, see CMakeLists.txt) this compiles to the exact same
    // code as the walk written out by hand: the `if (!true) return false`
    // this introduces per entry/level folds away entirely. Same trick
    // model.h's for_each_by_field uses over its own short-circuiting
    // sibling (scan_field_short_circuit) -- do not duplicate the traversal
    // itself to "avoid the indirection"; there is nothing at runtime to avoid.
    template <class F>
    static void each_in(const Node* n, F& f) {
        auto wrapped = [&](const auto& e) {
            f(e);
            return true;
        };
        each_in_short_circuit(n, wrapped);
    }

    // Same walk as each_in, but f returns bool (true = keep going, false =
    // stop) and the walk itself actually stops the moment f says so --
    // unlike a caller wrapping each_in in a "stop calling f once a flag
    // flips" guard, which still pays for visiting every remaining entry.
    // Needed for a real std::all_of/std::any_of over a trie-backed index
    // (see model.h's all_of_by_field, both branches): a predicate that fails
    // on the very first match should not force a scan of the other 999,999.
    template <class F>
    static bool each_in_short_circuit(const Node* n, F& f) {
        if (!n) return true;
        std::uint32_t pos = 0;
        for (std::uint32_t idx = 0; idx < 32; ++idx) {
            const std::uint32_t b = bit(idx);
            if (!(n->bitmap & b)) continue;
            if (n->is_leaf & b) {
                for (const Leaf* l = as_leaf(n->slots[pos]); l; l = l->next.get())
                    if (!f(l->entry)) return false;
            } else {
                if (!each_in_short_circuit(as_node(n->slots[pos]), f)) return false;
            }
            ++pos;
        }
        return true;
    }

public:
    TrieCore() = default;

    /// Binds this (empty) core to `mem` for every Node/Leaf it or any core
    /// derived from it (via set_entry/erase_key, which carry mem_ forward)
    /// ever allocates. See mem_'s own comment for the lifetime obligation
    /// this places on the caller -- `mem` must outlive every Node/Leaf ever
    /// allocated through it, which is why Model seeds this only with a
    /// resource whose lifetime is tied to the Model itself.
    explicit TrieCore(std::pmr::memory_resource* mem) : mem_(mem) {}

    std::size_t size() const noexcept { return size_; }
    bool empty() const noexcept { return size_ == 0; }

    /// Returns a new core with key -> entry. O(log32 n) node allocations;
    /// shares the rest of the structure with `*this`. Two fast paths avoid
    /// ever allocating a Node to hold a single child -- see root_is_leaf_'s
    /// own comment: a true-empty root promotes straight to a bare Leaf, and
    /// an existing leaf-root either updates in place (same hash) or gets
    /// wrapped in a freshly synthesized one-slot Node before handing off to
    /// the UNMODIFIED set_in() (different hash) -- the exact push-down
    /// set_in already performs for an interior leaf slot (see its "Slot
    /// holds a leaf... different hash" branch), just rooted at shift 0.
    TrieCore set_entry(const K& key, const Entry& entry) const {
        bool added = false;
        const std::uint64_t hash = hash_key(key);

        if (!root_) {
            auto lf = make_leaf(entry, ChainLink{});
            return TrieCore(std::move(lf), 1, mem_, /*leaf_root=*/true);
        }

        if (root_is_leaf_) {
            const Leaf* lf = as_leaf(root_);
            const std::uint64_t lf_hash = entry_hash(lf->entry);
            if (lf_hash == hash) {
                auto new_leaf = chain_set(lf, key, entry, added);
                return TrieCore(std::move(new_leaf), size_ + (added ? 1 : 0), mem_,
                                /*leaf_root=*/true);
            }
            Node* node = clone_node(nullptr);
            const std::uint32_t exist_idx = slice(lf_hash, 0);
            node->bitmap = bit(exist_idx);
            node->is_leaf = bit(exist_idx);
            node->slots.push_back(root_);  // reuse the existing Leaf, no clone
            RcHandle node_owner = RcHandle::adopt(node);
            auto r = set_in(node_owner, /*parent_private=*/true, hash, 0, key, entry, added);
            return TrieCore(std::move(r), size_ + (added ? 1 : 0), mem_, /*leaf_root=*/false);
        }

        // root_ itself, NOT root_.get()/as_node(root_): set_in needs the
        // actual RcHandle (by reference, no copy) to check use_count() --
        // see its own doc comment. parent_private=true: there is no
        // ancestor above the root to be shared with.
        auto r = set_in(root_, /*parent_private=*/true, hash, 0, key, entry, added);
        return TrieCore(r, size_ + (added ? 1 : 0), mem_, /*leaf_root=*/false);
    }

    /// Returns a new core without `key` (or, if absent, THE SAME core,
    /// sharing root_ unchanged -- not just structurally equal, actually
    /// aliasing it: erase_in returns `owner` untouched all the way up when
    /// nothing was found, so an absent key costs one RcHandle copy, zero
    /// node allocations, same as get()'s cost shape. See erase_in's own
    /// comment. Does NOT collapse an interior Node back into a leaf-root
    /// when erase shrinks it to one entry -- matches this trie's existing
    /// policy of never path-collapsing on erase (see erase_in's own
    /// comment); a leaf-root only ever arises via set_entry's 0->1
    /// transition.
    TrieCore erase_key(const K& key) const {
        bool removed = false;
        const std::uint64_t hash = hash_key(key);

        if (root_is_leaf_) {
            const Leaf* lf = as_leaf(root_);
            if (!leaf_get(lf, key)) return *this;  // absent: no-op, zero allocation
            removed = true;
            auto nl = chain_erase(lf, key);
            if (nl) return TrieCore(std::move(nl), size_ - 1, mem_, /*leaf_root=*/true);
            return TrieCore(nullptr, 0, mem_, /*leaf_root=*/false);
        }

        auto r = erase_in(root_, hash, 0, key, removed);
        return TrieCore(std::move(r), size_ - (removed ? 1 : 0), mem_, /*leaf_root=*/false);
    }

    const Entry* get_entry(const K& key) const {
        if (root_is_leaf_) return leaf_get(as_leaf(root_), key);
        return get_in(as_node(root_), hash_key(key), 0, key);
    }

    template <class F>
    void each_entry(F&& f) const {
        if (root_is_leaf_) {
            for (const Leaf* l = as_leaf(root_); l; l = l->next.get()) f(l->entry);
            return;
        }
        each_in(as_node(root_), f);
    }

    template <class F>
    bool each_entry_short_circuit(F&& f) const {
        if (root_is_leaf_) {
            for (const Leaf* l = as_leaf(root_); l; l = l->next.get())
                if (!f(l->entry)) return false;
            return true;
        }
        return each_in_short_circuit(as_node(root_), f);
    }

    /// capacity() vs size() of every Node::slots buffer in this trie -- see
    /// Model::slot_stats_diagnostics(), the diagnostic this backs. Kept
    /// separate from the trie's own hot paths (get/set/erase never call
    /// this): an O(#trie nodes) walk, the same reason lookup_diagnostics()
    /// is kept off Model::diagnostics()'s path -- see that method's own
    /// doc comment in model.h.
    SlotStats slot_stats() const {
        SlotStats s;
        if (root_is_leaf_) {
            ++s.leaf_count;  // no Node at all -- that's the entire point of root_is_leaf_
            return s;
        }
        walk_slot_stats(as_node(root_), s);
        return s;
    }

private:
    static void walk_slot_stats(const Node* n, SlotStats& s) {
        if (!n) return;
        ++s.node_count;
        s.total_capacity += n->slots.capacity();
        s.total_size += n->slots.size();
        std::uint32_t pos = 0;
        for (std::uint32_t idx = 0; idx < 32; ++idx) {
            const std::uint32_t b = bit(idx);
            if (!(n->bitmap & b)) continue;
            if (n->is_leaf & b) {
                ++s.leaf_count;
            } else {
                walk_slot_stats(as_node(n->slots[pos]), s);
            }
            ++pos;
        }
    }

public:

    // Frame stack depth needed to walk this trie externally (one Node per
    // level). 13 levels of 5-bit hash slices exhaust all 64 hash bits (shift
    // = 0, 5, ..., 60) -- see persistent_map_tests.cpp's comment on the
    // "ran out of hash bits" branch being provably unreachable for two
    // distinct keys. A fixed array, not a std::vector, so constructing an
    // iterator never allocates -- the whole reason for_each_by_field & co.
    // (model.h) avoid materializing a vector in the first place would be
    // defeated if the iterator backing a range-for did it anyway.
    static constexpr int kMaxDepth = 13;

    struct Frame {
        const Node* node = nullptr;
        std::uint32_t idx = 0;  ///< next bit index (0..32) not yet examined at this node
    };

public:
    /// External forward iterator over the trie's entries, same order
    /// each_in_short_circuit visits them in (unspecified -- trie layout
    /// order, not insertion or sorted order). Reifies that recursive walk as
    /// explicit (stack, leaf-chain-pointer) state instead of a call stack, so
    /// it can be resumed one entry at a time from operator++ -- what
    /// each_in_short_circuit itself cannot do, since it only ever gets to
    /// decide "stop" by not calling back into the walk again, never "pause
    /// and hand control back to the caller between two calls to f".
    ///
    /// Read-only: the trie is immutable once published (same rule as every
    /// other persistent structure in this project), so there is no
    /// non-const flavor to provide.
    class iterator {
    public:
        using iterator_category = std::forward_iterator_tag;
        using value_type = Entry;
        using difference_type = std::ptrdiff_t;
        using pointer = const Entry*;
        using reference = const Entry&;

        iterator() = default;

        reference operator*() const { return chain_->entry; }
        pointer operator->() const { return &chain_->entry; }

        iterator& operator++() {
            advance();
            return *this;
        }
        iterator operator++(int) {
            iterator tmp = *this;
            advance();
            return tmp;
        }

        friend bool operator==(const iterator& a, const iterator& b) noexcept {
            return a.chain_ == b.chain_;
        }
        friend bool operator!=(const iterator& a, const iterator& b) noexcept { return !(a == b); }

    private:
        friend class TrieCore;
        explicit iterator(const Node* root) {
            if (root) {
                stack_[depth_++] = Frame{root, 0};
                advance();
            }
        }

        // root_is_leaf_ case: no Frame to push (there is no Node), so
        // depth_ stays 0 -- advance()'s `while (depth_ > 0)` is then a
        // no-op once the chain is exhausted, landing correctly on the same
        // (chain_==nullptr, depth_==0) state end() already uses.
        explicit iterator(const Leaf* chain) noexcept : chain_(chain) {}

        // Moves to the next entry (or to the end state -- chain_ == nullptr,
        // depth_ == 0 -- if none remain). Exactly each_in_short_circuit's
        // walk, just paused/resumed via `stack_`/`chain_` instead of
        // recursion -- do not let this drift into a second, different
        // traversal order than that one; see this class's own doc comment.
        void advance() {
            if (chain_) {
                chain_ = chain_->next.get();
                if (chain_) return;
            }
            while (depth_ > 0) {
                Frame& top = stack_[depth_ - 1];
                if (top.idx >= 32) {
                    --depth_;
                    continue;
                }
                const std::uint32_t idx = top.idx++;
                const std::uint32_t b = bit(idx);
                if (!(top.node->bitmap & b)) continue;
                const std::uint32_t pos = popcount_below(top.node->bitmap, idx);
                if (top.node->is_leaf & b) {
                    chain_ = as_leaf(top.node->slots[pos]);
                    return;
                }
                assert(depth_ < kMaxDepth &&
                       "trie deeper than kMaxDepth -- see its own comment");
                stack_[depth_++] = Frame{as_node(top.node->slots[pos]), 0};
            }
        }

        std::array<Frame, kMaxDepth> stack_{};
        int depth_ = 0;
        const Leaf* chain_ = nullptr;
    };

    iterator begin() const {
        if (root_is_leaf_) return iterator(as_leaf(root_));
        return iterator(as_node(root_));
    }
    iterator end() const { return iterator(); }
};

// Out-of-line: needs Node and Leaf complete (sizeof/alignof, and to call the
// right destructor), which they only are once TrieCore's whole body -- Node
// and Leaf included -- has been parsed. RcHandle::release() has access to
// TrieCore's other private nested types the same way any RcHandle member
// would from inside the class body: nested-class access to the enclosing
// class's private members doesn't depend on where the member is defined.
template <class K, class Entry, class Hash, class KeyOf>
inline void TrieCore<K, Entry, Hash, KeyOf>::RcHandle::release() noexcept {
    if (!ptr_) return;
    RcBase* b = reinterpret_cast<RcBase*>(ptr_);
    if (b->refcount.fetch_sub(1, std::memory_order_acq_rel) == 1) {
        if (b->is_leaf_self) {
            Leaf* lf = static_cast<Leaf*>(ptr_);
            std::pmr::memory_resource* m = lf->rc.owner_mem;
            lf->~Leaf();
            if (m) m->deallocate(lf, sizeof(Leaf), alignof(Leaf));
            else ::operator delete(lf);
        } else {
            Node* nd = static_cast<Node*>(ptr_);
            std::pmr::memory_resource* m = nd->rc.owner_mem;
            nd->~Node();
            if (m) m->deallocate(nd, sizeof(Node), alignof(Node));
            else ::operator delete(nd);
        }
    }
    ptr_ = nullptr;
}

}  // namespace detail

template <class K, class V, class Hash>
class PersistentMap {
    using Core = detail::TrieCore<K, std::pair<K, V>, Hash, detail::PairKeyOf<K, V>>;
    Core core_;
    explicit PersistentMap(Core c) : core_(std::move(c)) {}

public:
    PersistentMap() = default;

    /// An empty map whose Node/Leaf allocations (this one and
    /// every one later derived from it via set()/erase()) are pooled
    /// through `mem` instead of plain new/delete. See detail::TrieCore::
    /// mem_'s own comment for the lifetime obligation this places on the
    /// caller.
    explicit PersistentMap(std::pmr::memory_resource* mem) : core_(mem) {}

    std::size_t size() const noexcept { return core_.size(); }
    bool empty() const noexcept { return core_.empty(); }

    /// Returns a new map with key=val. O(log32 n) node allocations; shares the
    /// rest of the structure with `*this`.
    ///
    /// NOT ALWAYS A REAL CLONE -- like View<T>, this has a hazard worth
    /// reading before reusing this type outside its one current caller
    /// (Model's own indexes). set_in's mutate-in-place fast path (see its own
    /// long comment) mutates a node in place, rather than cloning it, whenever
    /// that node is provably unshared (use_count()==1 at every ancestor down
    /// to it). That's provably safe for Model, which only ever touches these
    /// tries from inside try_commit() under commit_mu_ and never exposes one
    /// to a reader mid-transaction -- but it means `*this` itself can be the
    /// node that gets mutated: `auto old = m; m = m.set(k, v);` can
    /// retroactively change what `old` reads if nothing else is keeping
    /// `old`'s root alive with an extra owning reference. Safe as long as
    /// you always discard (or explicitly, separately capture) the prior
    /// value the same way Model does; do not assume `old` above is
    /// insulated from `m`'s later mutations.
    PersistentMap set(const K& key, const V& val) const {
        return PersistentMap(core_.set_entry(key, std::pair<K, V>(key, val)));
    }

    /// Returns a new map without `key` (or an equal map if absent). Same
    /// mutate-in-place hazard as set() above.
    PersistentMap erase(const K& key) const { return PersistentMap(core_.erase_key(key)); }

    const V* get(const K& key) const {
        const std::pair<K, V>* e = core_.get_entry(key);
        return e ? &e->second : nullptr;
    }

    /// Visits every (key, value) pair once. Order is the trie's own layout
    /// (hash-slice bucket order, walked idx 0..31 per node -- see
    /// detail::TrieCore::each_in) -- NOT insertion order and NOT sorted by
    /// key. Every model.h caller built on this (Root::by_key's iteration,
    /// etc.) inherits that same "unspecified order" contract already stated
    /// for the model's own find_by_field and friends.
    template <class F>
    void for_each(F&& f) const {
        core_.each_entry([&](const std::pair<K, V>& e) { f(e.first, e.second); });
    }

    /// Short-circuiting form of for_each: `f(key, value)` returns bool (true
    /// = keep going, false = stop), and this returns whether the walk ran to
    /// completion (false iff `f` stopped it early). See TrieCore::
    /// each_entry_short_circuit for what this is built on.
    template <class F>
    bool for_each_short_circuit(F&& f) const {
        return core_.each_entry_short_circuit(
            [&](const std::pair<K, V>& e) { return f(e.first, e.second); });
    }

    /// Range-based-for support: `for (auto& [k, v] : m)`. Same trie-layout-
    /// order caveat as for_each above. See detail::TrieCore::iterator for
    /// what backs this -- an external walk, not a materialized vector, so
    /// (unlike find_by_field/find_referrers & co. in model.h) this costs
    /// nothing extra to allocate; each ++ is real trie-walk work
    /// though, not free -- prefer for_each_short_circuit in a hot loop that
    /// wants to stop early without composing range adaptors on top.
    using iterator = typename Core::iterator;
    using const_iterator = iterator;
    iterator begin() const { return core_.begin(); }
    iterator end() const { return core_.end(); }

    /// Node::slots capacity() vs size(), summed across this map -- see
    /// Model::slot_stats_diagnostics(), the diagnostic this backs.
    SlotStats slot_stats() const { return core_.slot_stats(); }
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

    /// An empty set whose Node/Leaf allocations (this one and
    /// every one later derived from it via insert()/erase()) are pooled
    /// through `mem` instead of plain new/delete. See detail::TrieCore::
    /// mem_'s own comment for the lifetime obligation this places on the
    /// caller.
    explicit PersistentSet(std::pmr::memory_resource* mem) : core_(mem) {}

    std::size_t size() const noexcept { return core_.size(); }
    bool empty() const noexcept { return core_.empty(); }

    /// Returns a new set with `key` present. O(log32 n) node allocations;
    /// shares the rest of the structure with `*this`. Same mutate-in-place
    /// aliasing hazard as PersistentMap::set -- see its doc comment.
    PersistentSet insert(const K& key) const { return PersistentSet(core_.set_entry(key, key)); }

    /// Returns a new set without `key` (or an equal set if absent). Same
    /// mutate-in-place hazard as PersistentMap::set.
    PersistentSet erase(const K& key) const { return PersistentSet(core_.erase_key(key)); }

    bool contains(const K& key) const { return core_.get_entry(key) != nullptr; }

    /// Visits every key once. Same "trie layout order, not insertion or
    /// sorted order" caveat as PersistentMap::for_each above -- see its
    /// comment.
    template <class F>
    void for_each(F&& f) const {
        core_.each_entry([&](const K& k) { f(k); });
    }

    /// Short-circuiting form of for_each: `f(key)` returns bool (true = keep
    /// going, false = stop), and this returns whether the walk ran to
    /// completion (false iff `f` stopped it early). See TrieCore::
    /// each_entry_short_circuit for what this is built on.
    template <class F>
    bool for_each_short_circuit(F&& f) const {
        return core_.each_entry_short_circuit([&](const K& k) { return f(k); });
    }

    /// Range-based-for support: `for (auto& k : s)`. See PersistentMap::
    /// begin()/end() (and detail::TrieCore::iterator) for the same notes on
    /// ordering and cost -- identical here, just over K entries instead of
    /// (K, V) pairs.
    using iterator = typename Core::iterator;
    using const_iterator = iterator;
    iterator begin() const { return core_.begin(); }
    iterator end() const { return core_.end(); }

    /// Node::slots capacity() vs size(), summed across this map -- see
    /// Model::slot_stats_diagnostics(), the diagnostic this backs.
    SlotStats slot_stats() const { return core_.slot_stats(); }
};

/// Add `v` to the persistent-SET bucket at `key` of a PersistentMap<K,
/// PersistentSet<V, VHash>, Hash> multimap -- the shared two-level shape
/// behind model.h's Root::by_cached_field and Root::by_cached_reference
/// (an outer map keyed by field tag or value, an inner set of every
/// currently-matching Id). Creates a `mem`-seeded bucket if this is the
/// first holder of that key -- deliberately never a bare `{}`, which would
/// start an UNPOOLED lineage for that one bucket (see PersistentMap's/
/// PersistentSet's own pooling constructor comment above).
template <class K, class Hash, class V, class VHash>
PersistentMap<K, PersistentSet<V, VHash>, Hash> bucket_insert(
    const PersistentMap<K, PersistentSet<V, VHash>, Hash>& m, const K& key, const V& v,
    std::pmr::memory_resource* mem) {
    const PersistentSet<V, VHash>* bucket = m.get(key);
    return m.set(key, (bucket ? *bucket : PersistentSet<V, VHash>(mem)).insert(v));
}

/// Remove `v` from the bucket at `key`, dropping the bucket ENTIRELY once it
/// empties -- so a value/target with no remaining holders leaves no
/// tombstone entry behind, matching bucket_insert's counterpart obligation.
/// A no-op (returns `m` unchanged) if `key` was never indexed at all, or
/// its bucket never held `v`.
template <class K, class Hash, class V, class VHash>
PersistentMap<K, PersistentSet<V, VHash>, Hash> bucket_erase(
    const PersistentMap<K, PersistentSet<V, VHash>, Hash>& m, const K& key, const V& v) {
    const PersistentSet<V, VHash>* bucket = m.get(key);
    if (!bucket) return m;
    PersistentSet<V, VHash> nb = bucket->erase(v);
    return nb.empty() ? m.erase(key) : m.set(key, nb);
}

}  // namespace model::pmap
