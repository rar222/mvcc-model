#pragma once
//
// A snapshot-isolated object model with multi-writer optimistic concurrency.
//
//   - any thread can write: begin() a Transaction from a Snapshot, mutate a
//     private view, try_commit() it
//   - a commit succeeds only if (a) no other transaction has committed since
//     that touched the same ids, and (b) every Ref<T> in the final state
//     still resolves against the LATEST state, not just the transaction's base
//   - cascade delete is resolved once, at commit time, against the single
//     authoritative reverse index -- never eagerly, never per-transaction
//   - readers take an O(1) immutable snapshot; refs resolve within it
//   - subscribers get a bounded, coalescing queue of change events
//
// See DESIGN.md for why it is built this way, and for how this differs from
// its single-writer sibling project (snapshot-model). The load-bearing
// decision carried over unchanged: there is no per-object refcount anywhere.
// Chunks hold raw pointers so that copying one is a memcpy, and lifetime is
// handled by a version watermark instead.
//
// The load-bearing decision that's NEW here: cascade-delete resolution is
// deferred to commit time. That one choice is what keeps the reverse index
// (referrers_) a single, writer-owned structure -- never per-transaction,
// never needing to be merged -- even though many threads can now build
// transactions concurrently. See CLAUDE.md invariant 8.

#include <algorithm>
#include <atomic>
#include <cassert>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <type_traits>
#include <typeinfo>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "model/persistent_map.h"

namespace model {

// ---------------------------------------------------------------------------
// Identity
// ---------------------------------------------------------------------------

/// The raw key. Prefer Ref<T> / Opt<T> everywhere in your own code -- Id is the
/// untyped form, and shows up only in change events (which are heterogeneous)
/// and in the model's internals.
///
/// Dense slot index plus a generation, bumped on every reuse of a slot. Slots
/// recycle constantly, so without the generation a stale handle would silently
/// alias whatever object landed in that slot next.
///
/// 32-bit on purpose, and asymmetric with the model's 64-bit version
/// counters. The rule: 64 bits for values bounded by TIME (Model::version_
/// increments forever -- at 100 commits/sec a uint32 would wrap in ~16
/// months of uptime, guaranteed), 32 bits for values bounded by POPULATION.
/// index gives 2^31 usable slots (top bit is kLocalIdBit), ~2000x the
/// target scale; gen wraps only after 2^32 reuses of ONE slot -- worst-case
/// LIFO hammering of a single slot for ~16 months -- and alloc_slot()
/// handles that by retiring the slot, at the cost of one slot, counted in
/// exhausted_slots(). Widening to 64+64 would double every stored
/// Ref<>/Opt<> (the whole point of thin refs), grow RefEdge 24->32 bytes,
/// double Chunk's gen array (+33% memcpy per COWed chunk per commit), and
/// demote IdHash from a perfect 64-bit pack to a lossy hash -- real costs
/// on the hottest structures, buying headroom the design never uses. Do not
/// "future-proof" this.
struct Id {
    std::uint32_t index = 0;  ///< slot number in the spine (chunk = index >> kChunkBits);
                              ///< top bit set = LOCAL placeholder, see kLocalIdBit
    std::uint32_t gen = 0;    ///< which lifetime of that slot; 0 means null

    explicit operator bool() const noexcept { return gen != 0; }  ///< non-null?

    /// Memberwise (index AND gen). Deliberately not just index: two Ids with
    /// the same slot but different generations name DIFFERENT objects (one
    /// dead, recycled since) and must never compare equal -- that's the
    /// whole reason gen exists (see the struct comment).
    friend bool operator==(Id, Id) noexcept = default;
};

/// Hasher for the model's own unordered containers keyed by Id (a
/// Transaction's remove_intents_, Subscription's coalescing merge, ...).
struct IdHash {
    std::size_t operator()(Id id) const noexcept {
        // Compute in uint64_t, NOT size_t: on a 32-bit size_t platform the
        // old `size_t(gen) << 32` was a shift past the type's width -- UB.
        // The full 64-bit pack is a perfect (collision-free) hash of Id.
        const std::uint64_t h = (static_cast<std::uint64_t>(id.gen) << 32) | id.index;
        if constexpr (sizeof(std::size_t) >= sizeof(std::uint64_t)) {
            return static_cast<std::size_t>(h);  // 64-bit size_t: identical to before
        } else {
            // 32-bit size_t: fold, so gen still participates instead of
            // being silently truncated away.
            return static_cast<std::size_t>(h ^ (h >> 32));
        }
    }
};

/// The top bit of Id::index marks a LOCAL placeholder id, assigned by
/// Transaction::create() to an object that doesn't have a real slot yet.
/// Real slots (Model::next_slot_) start at 0 and grow by allocation, so at
/// this project's target scale (100k-1M objects) they never approach this
/// bit; Model::apply_create asserts defensively if next_slot_ ever does.
///
/// A local id is a purely transaction-scoped concept: it lets one Transaction
/// build a graph of new, interlinked objects (create an Account, then create
/// an Order referencing it, in the same transaction) before either has a real
/// id. It is remapped to a real Id -- see RefRemapper -- the moment the object
/// is actually installed during try_commit()'s apply phase, and must never
/// escape a Transaction (there is no way to look one up in a Snapshot; it
/// isn't a real slot).
inline constexpr std::uint32_t kLocalIdBit = 0x8000'0000u;

/// True for a Transaction-scoped placeholder id (see kLocalIdBit), false for
/// a real slot. Everything that can meet both kinds -- Transaction's peek/
/// update/remove, RefRemapper, CommitResult::resolve -- branches on this.
inline bool is_local(Id id) noexcept {
    return (id.index & kLocalIdBit) != 0;
}

namespace detail {
/// Opaque, internal-only string encoding of an Id's raw bytes -- never a
/// user-visible string. An Id is already unique (index + generation), so
/// its bytes are a perfectly good PersistentMap key with no extra hashing
/// collisions to worry about. Used for every index keyed by IDENTITY rather
/// than by a declared field's value: Root::by_type, and the inner buckets
/// of Root::by_cached_field / Root::by_cached_reference.
inline std::string id_key(Id id) {
    return std::string(reinterpret_cast<const char*>(&id), sizeof(id));
}
}  // namespace detail

// ---------------------------------------------------------------------------
// Type tags -- a cheap, RTTI-free checked downcast
// ---------------------------------------------------------------------------

/// A type's identity, as an opaque address. Compared (never dereferenced) to
/// check "is this object a T?" -- one virtual call plus a pointer compare,
/// on the read path that runs millions of times a second. dynamic_cast is
/// not in that budget, which is why this exists (see CLAUDE.md).
using TypeTag = const void*;

/// The tag for T: the address of a per-T function-local static, so every use
/// of type_tag<T>() in the program agrees on one value with zero
/// registration. (Address-based identity assumes one program image -- the
/// usual caveat about comparing across shared-library boundaries applies.)
template <class T>
inline TypeTag type_tag() noexcept {
    static const char anchor = 0;
    return &anchor;
}

/// Same trick, one level down: a stable, RTTI-free tag per distinct
/// pointer-to-member *value* (e.g. &Gadget::label vs &Gadget::serial), used
/// so a field can be its own index key -- no int slot number to keep in sync
/// between where a field is declared indexed and where it's looked up.
template <auto Field>
inline const void* field_tag() noexcept {
    static const char anchor = 0;
    return &anchor;
}

/// Recovers the enclosing class from a pointer-to-member type, so
/// Snapshot::find_by_key<&Gadget::label>(...) needs no explicit <Gadget>.
/// Specialized for both a plain data member (&Gadget::label) and a nullary
/// const method (&Order::computed_key) -- the latter is what lets a
/// *computed* key be declared and looked up the same way as a stored field.
template <class M>
struct member_class;
template <class C, class V>
struct member_class<V C::*> {
    using type = C;
};
template <class C, class V>
struct member_class<V (C::*)() const> {
    using type = C;
};
template <class M>
using member_class_t = typename member_class<M>::type;

/// Recovers the field's own value type from a pointer-to-member type -- e.g.
/// Ref<Account> from &Order::account -- so find_referrers<&Order::account>(...)
/// can require exactly the right Ref<>/Opt<> type for `target`, at the
/// parameter type level rather than a runtime static_assert. Same data-member
/// / nullary-const-method split as member_class.
template <class M>
struct member_value;
template <class C, class V>
struct member_value<V C::*> {
    using type = V;
};
template <class C, class V>
struct member_value<V (C::*)() const> {
    using type = V;
};
template <class M>
using member_value_t = typename member_value<M>::type;

// ---------------------------------------------------------------------------
// Typed references
// ---------------------------------------------------------------------------

/// A non-nullable reference to a T.
///
/// Guaranteed to resolve inside any snapshot containing the object holding it.
/// The writer enforces that (Model::validate), and deleting the target cascades:
/// every Ref<> referrer dies with it.
///
/// There is no way to null a Ref<>. The mutating visitor simply has no overload
/// that touches one, so "non-nullable" is a property of the type system rather
/// than a convention someone has to remember.
template <class T>
class Ref {
public:
    using target_type = T;  ///< deduction hook: lets for_each_referrer<&Order::account>
                            ///< require exactly Ref<Account> for its target parameter

    Ref() = default;  ///< null until assigned; validate() rejects it at create time

    /// Wrap an untyped Id (e.g. a Change::id) -- explicit and UNCHECKED: the
    /// type claim is only verified when the ref is resolved/found against a
    /// Snapshot, whose tag check catches a wrong T then.
    explicit Ref(Id id) noexcept : id_(id) {}

    /// The untyped Id, for correlating with Change events or storing in
    /// heterogeneous containers. Type information is deliberately dropped.
    Id raw() const noexcept { return id_; }
    explicit operator bool() const noexcept { return static_cast<bool>(id_); }  ///< non-null?

    /// Same-target comparison, by Id value (index AND generation) -- not by
    /// which object each side happens to currently resolve to (there may be
    /// none, if either side is stale). Two Refs from different snapshots can
    /// legitimately compare equal or unequal without either being resolved.
    friend bool operator==(Ref, Ref) noexcept = default;

private:
    Id id_{};
};

/// A nullable reference to a T. Deleting the target does NOT delete the holder --
/// the field is nulled instead.
template <class T>
class Opt {
public:
    using target_type = T;  ///< same deduction hook as Ref<T>::target_type

    Opt() = default;                          ///< null -- a legal, permanent state for an Opt
    explicit Opt(Id id) noexcept : id_(id) {}  ///< unchecked wrap, same caveat as Ref(Id)

    /// A non-null Ref is always a valid Opt. Not the reverse.
    Opt(Ref<T> r) noexcept : id_(r.raw()) {}

    /// Clear to null by hand -- the same state a cascade null produces.
    /// (Ref<T> has no equivalent; that asymmetry IS the integrity guarantee.)
    void reset() noexcept { id_ = Id{}; }

    Id raw() const noexcept { return id_; }  ///< the untyped Id; see Ref::raw()
    explicit operator bool() const noexcept { return static_cast<bool>(id_); }  ///< non-null?

    /// Same-target comparison, by Id value -- see Ref::operator==. Two nulls
    /// (both default-constructed, or one reset()) always compare equal.
    friend bool operator==(Opt, Opt) noexcept = default;

private:
    Id id_{};
};

// ---------------------------------------------------------------------------
// Object model
// ---------------------------------------------------------------------------

/// Callback shape for enumerating an object's outgoing references (see
/// ObjectBase::each_ref): the field's identity tag, its diagnostic-only
/// name, the id it currently points at (possibly null), and whether the
/// field is an Opt<> (nullable) rather than a Ref<>. Drives the reverse
/// index, cascade resolution, and validation -- everything that needs to
/// walk edges without knowing concrete types.
using RefFn = std::function<void(const void* field, const char* name, Id target, bool nullable)>;

/// Reports every outgoing reference. Nullability is read off the *field type*, so
/// it cannot be mislabelled. The field is identified by whatever field_tag<Field>
/// the caller passes in -- same trick as define_keys() -- so there is no
/// hand-assigned int slot to keep in sync between define_references(),
/// null_ref(), and the reverse index. (Plain `const void*` here, not a
/// `Field` template parameter on operator() itself: define_references() is
/// generic over V -- it's called with RefReader, RefNuller, AND RefRemapper --
/// and a dependent `v.operator()<Field>(...)` call would need a `.template`
/// disambiguator. Computing the tag at the call site with the free function
/// field_tag<Field>() and passing it as an ordinary argument avoids that.)
///
/// `name` is diagnostic-only (IntegrityError messages) -- a string literal the
/// caller passes alongside field_tag<Field>(). A mismatch here can only
/// mislabel a diagnostic; nothing correctness-critical reads it.
struct RefReader {
    const RefFn& fn;
    template <class T>
    void operator()(const void* field, const char* name, const Ref<T>& r) const {
        fn(field, name, r.raw(), false);
    }
    template <class T>
    void operator()(const void* field, const char* name, const Opt<T>& r) const {
        fn(field, name, r.raw(), true);
    }
};

/// Nulls one nullable field. Note there is no Ref<T> overload that clears
/// anything -- that is the compile-time half of the integrity guarantee.
struct RefNuller {
    const void* target;
    template <class T>
    void operator()(const void*, const char*, Ref<T>&) const {}
    template <class T>
    void operator()(const void* field, const char*, Opt<T>& r) const {
        if (field == target) r.reset();
    }
};

/// Rewrites every LOCAL placeholder id (see kLocalIdBit) in an object's ref
/// fields to the real Id it was assigned during try_commit()'s apply phase.
/// This is what lets a transaction build "create an Account, then create an
/// Order referencing it" in one go: the Order's Ref<Account> holds a local id
/// until apply_create() installs the Account and records the mapping, at
/// which point applying the Order remaps it to the real one. A field that
/// isn't local (already a real id, or null) is left untouched.
///
/// Sets *unmapped (there are no exceptions in this project -- see CLAUDE.md)
/// if a local id has no entry in the table -- which means it pointed at a
/// local object that was never created, or one created-then-removed within
/// the same transaction (see Transaction::remove(), the create-then-uncreate
/// case). try_commit() then rejects the transaction as CommitStatus::Invalid:
/// a genuine transaction-building bug, not a concurrency conflict (there is
/// no real Id to check against the transaction's base, so it cannot be
/// classified as a Conflict -- see Model::validate's use of
/// IntegrityError::bad_target for the contrast). The field is left null; the
/// half-remapped object is never installed, so the value doesn't matter.
struct RefRemapper {
    const std::unordered_map<std::uint32_t, Id>& table;  // local Id::index -> real Id
    bool* unmapped;  // set on a local id with no mapping; the apply aborts

    Id resolve(Id id) const {
        auto it = table.find(id.index);
        if (it == table.end()) {
            *unmapped = true;
            return Id{};
        }
        return it->second;
    }

    template <class T>
    void operator()(const void*, const char*, Ref<T>& r) const {
        if (is_local(r.raw())) r = Ref<T>(resolve(r.raw()));
    }
    template <class T>
    void operator()(const void*, const char*, Opt<T>& r) const {
        if (r.raw() && is_local(r.raw())) r = Opt<T>(resolve(r.raw()));
    }
};

/// Callback shape for declaring WHICH Ref<>/Opt<> fields -- already listed in
/// define_references() -- should also get a reverse-lookup index. See
/// define_cached_references() and Snapshot::find_cached_referrers.
using RefIndexFn = std::function<void(const void* field)>;

/// Visitor for define_cached_references(): `v.index<&Order::account>()`
/// marks that field for the index. No value to supply -- unlike
/// FieldKeyReader::key(), this is pure metadata (WHICH field, not what
/// value it holds right now), so define_cached_references() must not read
/// instance state: Object<Derived> calls it once per TYPE (a function-local
/// static caches the result), not once per object, and if it happened to
/// return something different for a different instance, only the FIRST
/// caller's answer would ever be used.
struct RefIndexReader {
    const RefIndexFn& fn;
    template <auto Field>
    void index() const {
        fn(field_tag<Field>());
    }
};

/// Callback shape for enumerating a type's declared lookup fields (see
/// ObjectBase::each_field_key / each_cached_field / each_scan_field): the
/// field's identity tag plus its value in canonical string form
/// (to_field_key). One shape serves all three lookup families.
using FieldKeyFn = std::function<void(const void* field, std::string key)>;

/// Canonical string form of a define_keys()-indexed field's value: a
/// std::string as-is, or an arithmetic type via std::to_string. Shared by
/// FieldKeyReader::key() (writing the index) and Snapshot::find_by_key()
/// (reading it) so the two can never drift apart on what a value maps to --
/// which is also what makes find_by_key type safe: it takes the field's own
/// value type, not a bare std::string, so a caller can't pass a key of the
/// wrong shape for that field in the first place.
template <class V>
std::string to_field_key(const V& v) {
    if constexpr (std::is_same_v<V, std::string>) {
        return v;
    } else if constexpr (std::is_arithmetic_v<V>) {
        return std::to_string(v);
    } else {
        static_assert(!sizeof(V), "define_keys() field must be std::string or an arithmetic type");
    }
}

/// Reports every field a type opts into fast lookup via define_keys(). Unlike
/// RefReader (which only ever sees Ref<T>/Opt<T>), this accepts any field
/// whose value to_field_key() can turn into a canonical string key. A type
/// with no define_keys() reports nothing -- "zero or more" fields, not
/// mandatory.
///
/// The field is identified by its own pointer-to-member (via field_tag),
/// not a hand-assigned int slot: `v.key<&Gadget::label>(s.label)`. That's
/// what lets Snapshot::find_by_key<&Gadget::label>(...) look a field up
/// directly, with no int to keep in sync between declaration and lookup.
struct FieldKeyReader {
    const FieldKeyFn& fn;
    template <auto Field, class V>
    void key(const V& v) const {
        fn(field_tag<Field>(), to_field_key(v));
    }
};

namespace detail {
// Detection traits for the four opt-in declarations (define_references,
// define_keys, define_cached_fields, define_scan_fields). Object<Derived>
// branches on these with `if constexpr`, so a user type declares only what
// it needs -- omitting one costs nothing and overrides nothing -- and a
// declaration with the wrong signature simply doesn't match (it is silently
// unused, which is why every declared field deserves a test; see CLAUDE.md).
template <class D, class = void>
struct has_define_references : std::false_type {};
template <class D>
struct has_define_references<D, std::void_t<decltype(D::define_references(
                                    std::declval<const D&>(), std::declval<const RefReader&>()))>>
    : std::true_type {};

template <class D, class = void>
struct has_define_keys : std::false_type {};
template <class D>
struct has_define_keys<D, std::void_t<decltype(D::define_keys(
                              std::declval<const D&>(), std::declval<const FieldKeyReader&>()))>>
    : std::true_type {};

template <class D, class = void>
struct has_define_cached_fields : std::false_type {};
template <class D>
struct has_define_cached_fields<D, std::void_t<decltype(D::define_cached_fields(
                                       std::declval<const D&>(), std::declval<const FieldKeyReader&>()))>>
    : std::true_type {};

template <class D, class = void>
struct has_define_scan_fields : std::false_type {};
template <class D>
struct has_define_scan_fields<D, std::void_t<decltype(D::define_scan_fields(
                                     std::declval<const D&>(), std::declval<const FieldKeyReader&>()))>>
    : std::true_type {};

template <class D, class = void>
struct has_define_cached_references : std::false_type {};
template <class D>
struct has_define_cached_references<D, std::void_t<decltype(D::define_cached_references(
                                          std::declval<const D&>(), std::declval<const RefIndexReader&>()))>>
    : std::true_type {};

/// Demangles a typeid name for use as a diagnostic label (Object<Derived>::type()).
/// Itanium ABI (GCC/Clang) only; passed through unchanged elsewhere (already
/// readable on MSVC). Defined in model.cpp so <cxxabi.h> doesn't leak into every
/// translation unit that includes this header.
std::string demangle_type_name(const std::type_info& ti);
}  // namespace detail

/// The type-erased base of every stored object -- what the model actually
/// holds in its chunks and passes through its internals, which must handle
/// heterogeneous objects (change events, cascade BFS, index maintenance)
/// without knowing concrete types. Never derive from it directly: derive
/// from Object<Derived>, which implements every virtual from the type's own
/// declarations. Instances are owned by the Model once committed (or by a
/// Transaction while pending); user code only ever sees `const` access
/// through a Snapshot, or a mutable pointer scoped to its own Transaction.
class ObjectBase {
public:
    /// Assigned by the machinery -- Transaction::create() (a local
    /// placeholder) and try_commit()'s apply phase (the real slot). Never
    /// set it yourself; an object's identity is not user data.
    Id id;

    virtual ~ObjectBase() = default;  ///< objects are deleted through base pointers

    /// A faithful copy of the derived object (Object<Derived> implements it
    /// via the copy constructor -- keep derived types copyable). This is the
    /// copy-on-write primitive: Transaction::update() clones the committed
    /// object for local editing, and the cascade BFS clones a referrer
    /// before nulling one field, so published state is never mutated.
    virtual ObjectBase* clone() const = 0;

    /// The type's identity, for the checked downcast (see TypeTag). Always
    /// type_tag<Derived>() -- Object<Derived> implements it.
    virtual TypeTag tag() const noexcept = 0;

    /// Diagnostic label only (IntegrityError messages) -- never compared or
    /// used for dispatch; tag()/type_tag<T>() is what the model actually
    /// checks types against. Object<Derived> derives this from typeid()
    /// automatically; there is nothing to override.
    virtual const char* type() const = 0;

    /// Visit every outgoing Ref<>/Opt<> field, as declared in
    /// define_references(). The default (no references) is what a type
    /// without define_references() gets. Feeds validation, the reverse
    /// index, and cascade resolution -- a field missing here is invisible
    /// to all three (see Object<>'s warning).
    virtual void each_ref(const RefFn&) const {}

    /// Null the one Opt<> field identified by `field` -- the cascade BFS's
    /// write primitive when a target dies but the referrer survives. A
    /// Ref<> field cannot be nulled through this (RefNuller has no clearing
    /// overload), which is the compile-time half of the integrity guarantee.
    virtual void null_ref(const void* /*field*/) {}

    /// Rewrites this object's own ref fields via a RefRemapper. Only ever
    /// called by Model::apply_create/apply_update, once, during try_commit()'s
    /// apply phase -- see RefRemapper's own comment for why this exists.
    virtual void remap_refs(const RefRemapper&) {}

    /// Fields declared in define_keys(), for fast lookup (Snapshot::find_by_key).
    /// Zero or more -- there is no mandatory key at all; a type that declares
    /// none (e.g. one only ever traversed by Ref<>, never looked up directly)
    /// reports nothing. Unlike a stable identity key, these are explicitly
    /// NOT assumed stable: the model diffs old vs. new value at apply time and
    /// keeps the index in sync.
    virtual void each_field_key(const FieldKeyFn&) const {}

    /// Fields declared in define_cached_fields(), for indexed MULTI-match
    /// lookup (Snapshot::find_by_cached_field). Same visitor shape and same
    /// diff-at-apply-time maintenance as each_field_key; see Object<> for the
    /// contract and the cost model.
    virtual void each_cached_field(const FieldKeyFn&) const {}

    /// Fields declared in define_scan_fields(), for UNINDEXED multi-match
    /// lookup (Snapshot::find_by_scan_field). Nothing in the model maintains
    /// anything for these -- the declaration is purely the visibility gate
    /// that keeps the three lookup families uniform (undeclared == invisible
    /// to the lookup). See Object<>.
    virtual void each_scan_field(const FieldKeyFn&) const {}

    /// The reverse-index SUBSET of each_ref(): only the Ref<>/Opt<> fields
    /// declared in define_cached_references(), for indexed "who points at
    /// this?" lookup (Snapshot::find_cached_referrers) -- the read-side
    /// counterpart of the writer-private referrers_ index (that one can
    /// never be handed to a reader; see CLAUDE.md invariant 8). Same
    /// diff-at-apply-time maintenance discipline as each_cached_field, just
    /// keyed by the field's TARGET instead of an arbitrary value. See
    /// Object<> for the contract and the cost model.
    virtual void each_cached_reference(const RefFn&) const {}
};

/// CRTP base. Derive from it and declare your reference fields ONCE, in
/// define_references() -- optional, like define_keys(): a type with no
/// outgoing Ref<>/Opt<> fields simply omits it.
///
///     class Order final : public model::Object<Order> {
///     public:
///         std::string code;
///         model::Ref<Account> account;   // deleting the account kills this order
///         model::Opt<Order>   parent;    // deleting the parent nulls this field
///
///         template <class Self, class V>
///         static void define_references(Self& s, V&& v) {
///             v(field_tag<&Order::account>(), "account", s.account);
///             v(field_tag<&Order::parent>(), "parent", s.parent);
///         }
///     };
///
/// That one list drives clone(), the reverse index, cascade delete, nulling,
/// AND local-id remapping (RefRemapper). A ref field missing from it is
/// invisible to the model -- no compiler error, no assert, just a dangling
/// reference in production. Keep it complete, and test every field you add.
/// The field identifies itself via field_tag<Field>, so there's no numbering
/// scheme to keep in sync.
///
/// There is no mandatory identity key. If you want one -- most types do, for
/// Snapshot::find_by_key -- declare it the same way any other lookup field
/// is declared, via define_keys(). `Field` can name a plain data member
/// (value passed explicitly) or a nullary const method, so a *computed* key
/// (e.g. a type prefix plus a name) works exactly like a stored one:
///
///     std::string computed_key() const { return "ord:" + code; }
///
///     template <class Self>
///     static void define_keys(Self& s, const model::FieldKeyReader& v) {
///         v.key<&Order::computed_key>(s.computed_key());  //
///         find_by_key<&Order::computed_key>("ord:O1")
///     }
///
/// Each define_keys() field is assumed unique within its type; a duplicate
/// value silently overwrites the earlier entry. For "give me every match,
/// not just one," there are two MULTI-match families, declared with the same
/// visitor shape and named the same way -- each define_X drives find_by_X
/// and view_by_X, and in every family a field you did NOT declare is
/// invisible to its lookup (empty result, same as find_by_key on a field
/// define_keys() never mentioned):
///
///   define_keys()          -> find_by_key         / view_by_key
///       unique, indexed: O(log n); later write wins on duplicates.
///   define_scan_fields()   -> find_by_scan_field  / view_by_scan_field
///       every match, UNINDEXED: O(#T objects) per query, but zero
///       write-side cost -- nothing is maintained per commit. For fields
///       queried rarely.
///   define_cached_fields() -> find_by_cached_field / view_by_cached_field
///       every match, INDEXED: O(log n + #matches) per query, paid for by
///       one index entry per object per field, maintained on every
///       create/delete/value-change inside the serialized commit path (each
///       cached field costs about what by_type does). For fields queried
///       often at scale.
///
///     template <class Self>
///     static void define_scan_fields(Self& s, const model::FieldKeyReader& v) {
///         v.key<&Order::qty>(s.qty);  // find_by_scan_field<&Order::qty>(5) -> ALL matches
///     }
///
/// (define_cached_fields is declared identically; a field may appear in more
/// than one family.) The fully undeclared escape hatch remains the
/// Snapshot::find_all() predicate scan.
///
/// A FOURTH family, structurally different (it indexes by TARGET, not by
/// value, and only applies to Ref<>/Opt<> fields already listed in
/// define_references()): define_cached_references() -> find_cached_referrers
/// / view_cached_referrers, the indexed O(log n + #matches) alternative to
/// the always-available for_each_referrer / find_referrers O(#T) scan.
/// Declare only the field's IDENTITY (define_references() already supplies
/// the target and nullability):
///
///     template <class Self>
///     static void define_cached_references(Self& s, const model::RefIndexReader& v) {
///         v.index<&Order::account>();  // find_cached_referrers<&Order::account>(acct)
///     }
///
/// Same cost model as define_cached_fields (one index entry per object per
/// declared reference field, maintained every commit) -- so cache the
/// references that get queried often (e.g. "every Order for this Account"),
/// and leave rarely-queried ones (a nullable `parent`, say) on the scan.
template <class Derived>
class Object : public ObjectBase {
public:
    ObjectBase* clone() const override { return new Derived(static_cast<const Derived&>(*this)); }

    TypeTag tag() const noexcept override { return type_tag<Derived>(); }

    const char* type() const override {
        // Computed once per Derived (function-local static, thread-safe init),
        // not once per call or per object -- demangling isn't a hot-path cost.
        static const std::string name = detail::demangle_type_name(typeid(Derived));
        return name.c_str();
    }

    void each_ref(const RefFn& fn) const override {
        if constexpr (detail::has_define_references<Derived>::value)
            Derived::define_references(static_cast<const Derived&>(*this), RefReader{fn});
    }

    void null_ref(const void* field) override {
        if constexpr (detail::has_define_references<Derived>::value)
            Derived::define_references(static_cast<Derived&>(*this), RefNuller{field});
    }

    void remap_refs(const RefRemapper& remapper) override {
        if constexpr (detail::has_define_references<Derived>::value)
            Derived::define_references(static_cast<Derived&>(*this), remapper);
    }

    void each_field_key(const FieldKeyFn& fn) const override {
        if constexpr (detail::has_define_keys<Derived>::value)
            Derived::define_keys(static_cast<const Derived&>(*this), FieldKeyReader{fn});
    }

    void each_cached_field(const FieldKeyFn& fn) const override {
        if constexpr (detail::has_define_cached_fields<Derived>::value)
            Derived::define_cached_fields(static_cast<const Derived&>(*this), FieldKeyReader{fn});
    }

    void each_scan_field(const FieldKeyFn& fn) const override {
        if constexpr (detail::has_define_scan_fields<Derived>::value)
            Derived::define_scan_fields(static_cast<const Derived&>(*this), FieldKeyReader{fn});
    }

    void each_cached_reference(const RefFn& fn) const override {
        if constexpr (detail::has_define_cached_references<Derived>::value) {
            // Computed once per Derived, not once per object: define_cached_
            // references() is required to be pure metadata (see
            // RefIndexReader), so any instance's answer is every instance's
            // answer -- a function-local static (thread-safe init) is exactly
            // the "compute once per type" tool already used for type()'s
            // demangled name above.
            static const std::unordered_set<const void*> wanted = [this] {
                std::unordered_set<const void*> w;
                RefIndexFn collect = [&](const void* f) { w.insert(f); };
                Derived::define_cached_references(static_cast<const Derived&>(*this), RefIndexReader{collect});
                return w;
            }();
            if constexpr (detail::has_define_references<Derived>::value) {
                RefFn filtered = [&](const void* field, const char* name, Id target, bool nullable) {
                    if (wanted.count(field)) fn(field, name, target, nullable);
                };
                Derived::define_references(static_cast<const Derived&>(*this), RefReader{filtered});
            }
        }
    }
};

// ---------------------------------------------------------------------------
// Storage
// ---------------------------------------------------------------------------

/// Slots per chunk = 2^kChunkBits. An Id::index splits into (chunk = index
/// >> kChunkBits, slot = index & kChunkMask). 256 balances the two costs
/// that pull in opposite directions: a bigger chunk means a bigger memcpy
/// every time a commit COWs it (a chunk is ~3KB at 256), a smaller chunk
/// means a longer spine to copy into every published Root.
inline constexpr std::uint32_t kChunkBits = 8;
inline constexpr std::uint32_t kChunkSize = 1u << kChunkBits;  // 256 slots
inline constexpr std::uint32_t kChunkMask = kChunkSize - 1;    // low bits: slot within chunk

/// The last usable generation. A slot that reaches this is retired rather than
/// reused, because the next bump would wrap to 0 -- the null id -- and a stale
/// handle would alias the new object. See Model::alloc_slot.
inline constexpr std::uint32_t kGenMax = 0xFFFFFFFFu;

/// Copy-on-write leaf. Raw pointers on purpose: copying a chunk must be a memcpy
/// with zero atomics. shared_ptr here would mean kChunkSize atomic RMWs per
/// dirtied chunk, which is what kills this design at scale. Object lifetime
/// is NOT managed here -- the version watermark (Model's reaper) frees an
/// object only once no snapshot that could see it remains.
struct Chunk {
    const ObjectBase* obj[kChunkSize] = {};  ///< null == empty slot; parallel to gen[]
    std::uint32_t gen[kChunkSize] = {};      ///< current generation of each slot; a lookup
                                             ///< whose Id::gen mismatches is stale, not found
};

/// One published, immutable version of the whole model -- everything a
/// Snapshot can see, in one struct. try_commit() builds a fresh Root per
/// commit and swaps it in atomically (Model::root_); nothing in a published
/// Root is ever mutated afterward (CLAUDE.md invariant 3). Cheap to derive:
/// the spine is a vector of shared_ptrs (chunks not touched by the commit
/// are shared with the previous Root), and the index maps share structure
/// persistently, so building one is proportional to the CHANGES, never to
/// model size.
struct Root {
    std::uint64_t version = 0;  ///< strictly increasing; what Snapshot::version() reports

    /// The object store: spine[index >> kChunkBits]->obj[index & kChunkMask].
    /// Dense, so a lookup is two indexed loads -- no hashing, no probing.
    std::vector<std::shared_ptr<const Chunk>> spine;

    /// One persistent map per type: every Id of that type, keyed by an
    /// internal encoding of the Id itself (never a user-visible string --
    /// there is no mandatory key). This is what makes for_each<T>() (and
    /// everything built on it: find_all, find_referrers, ...) O(#T objects)
    /// instead of a full spine scan. Keyed by TypeTag rather than a type-name
    /// string: the same address-comparison the rest of the model already uses
    /// for type checks (see type_tag<T>()), so no extra per-type registration
    /// is needed. One entry per distinct type (small, static-ish), so copying
    /// it per commit is O(#types), not O(#objects).
    std::unordered_map<TypeTag, pmap::PersistentMap<Id>> by_type;

    /// One persistent map per define_keys()-declared field, keyed by the
    /// field's own field_tag<>() -- which already encodes both the type and
    /// the field, so there's no separate per-type grouping needed here the
    /// way by_type needs TypeTag. Empty for types that declare none. This is
    /// the unique lookup-by-value index (later write wins); there is no
    /// separate mandatory "primary key" index.
    std::unordered_map<const void*, pmap::PersistentMap<Id>> by_field;

    /// One persistent MULTIMAP per define_cached_fields()-declared field:
    /// canonical value string -> a persistent set of every Id whose field
    /// currently holds that value (inner map keyed by the Id's own bytes,
    /// the same encoding by_type uses). Unlike by_field, every match is
    /// kept. The bucket is a persistent map, NEVER a flat vector: a flat
    /// bucket would make each mutation O(#duplicates of that value), which
    /// for a low-cardinality field (a status, a category) is O(n) per op --
    /// the exact size-proportional cost this design exists to avoid.
    std::unordered_map<const void*, pmap::PersistentMap<pmap::PersistentMap<Id>>> by_cached_field;

    /// One persistent MULTIMAP per define_cached_references()-declared
    /// Ref<>/Opt<> field: TARGET's Id bytes -> a persistent set of every
    /// REFERRER's Id whose field currently points there. Same bucket
    /// discipline as by_cached_field (persistent map buckets, never flat
    /// vectors -- a hub object referenced by thousands would otherwise make
    /// every one of those referrers' commits O(#referrers)). This is the
    /// published, read-side counterpart of the writer-private referrers_
    /// (Model::referrers_): that index drives cascade delete and can never
    /// be handed to a reader (CLAUDE.md invariant 8), so a field worth fast
    /// reverse lookup needs this SEPARATE, opt-in structure.
    std::unordered_map<const void*, pmap::PersistentMap<pmap::PersistentMap<Id>>> by_cached_reference;
};

class Model;
class Transaction;
struct CommitResult;

template <class T>
class View;

// ---------------------------------------------------------------------------
// Snapshot -- the read side
// ---------------------------------------------------------------------------

/// Cheap to copy, immutable, safe to share across threads. Holding one keeps its
/// version's objects alive, so drop it promptly: the reaper cannot advance past
/// the oldest live snapshot. A Transaction's base() also pins one -- see below.
///
/// A Snapshot must not outlive its Model. It looks like an owning handle -- it is
/// full of shared_ptr -- but the objects themselves are owned by the Model.
class Snapshot {
public:
    /// Null snapshot: sees nothing, resolves nothing, pins nothing. What a
    /// failed CommitResult carries; also the harmless moved-from state.
    Snapshot() = default;

    /// The committed version this snapshot reads (0 for a null snapshot).
    /// Comparable across snapshots of the same Model to order observations.
    std::uint64_t version() const noexcept { return root_ ? root_->version : 0; }
    explicit operator bool() const noexcept { return root_ != nullptr; }  ///< non-null?

    /// Total object count, summed across every type. O(#types), not
    /// O(#objects) -- the per-type maps themselves are O(1) to size.
    std::size_t size() const {
        if (!root_) return 0;
        std::size_t n = 0;
        for (const auto& [tag, m] : root_->by_type) n += m.size();
        return n;
    }

    /// Resolve a Ref stored inside an object of this snapshot. The writer's
    /// invariant guarantees it resolves. Never null.
    template <class T>
    const T& resolve(Ref<T> r) const noexcept {
        const ObjectBase* p = find_raw(r.raw());
        // If this fires, a commit published a dangling Ref -- always a writer
        // bug, since validate() is supposed to make it impossible.
        assert(p && "Ref dangled -- referential integrity is broken");
        assert(p->tag() == type_tag<T>() && "Ref<T> resolved to an object of another type");
        return *static_cast<const T*>(p);
    }

    /// Resolve an Opt. Deliberately nullable.
    template <class T>
    const T* resolve(Opt<T> r) const noexcept {
        return cast<T>(find_raw(r.raw()));
    }

    /// Look up a handle that came from *outside* this snapshot -- one you stored
    /// somewhere and held across commits. It may have been deleted, or its slot
    /// recycled. Checked; returns null if so.
    template <class T>
    const T* find(Ref<T> r) const noexcept {
        return cast<T>(find_raw(r.raw()));
    }
    template <class T>
    const T* find(Opt<T> r) const noexcept {
        return cast<T>(find_raw(r.raw()));
    }

    /// Fast lookup by a field declared via define_keys() (zero or more per
    /// type -- there is no mandatory key) -- named the same way a Ref<> field
    /// is, e.g. `s.find_by_key<&Account::name>("Widgets Inc")`, or
    /// `s.find_by_key<&Gadget::serial>(222)` for an arithmetic field.
    /// `Field` can also name a nullary const method instead of a data member,
    /// for a computed key: `s.find_by_key<&Order::computed_key>("ord:O1")`.
    /// The class AND the value's type are both deduced from the field itself --
    /// `value` must be the field's own value type (its declared parameter
    /// type is member_value_t<decltype(Field)>, not a bare std::string), so
    /// passing "222" for an int64_t field or 222 for a string field is a
    /// compile error, not a lookup that silently never matches. A duplicate
    /// value across two objects means the later write wins -- use the
    /// multi-match families (find_by_scan_field / find_by_cached_field) for
    /// "every match." Null if the key is absent, or the field wasn't
    /// declared as indexed.
    template <auto Field>
    const member_class_t<decltype(Field)>* find_by_key(
        const member_value_t<decltype(Field)>& value) const {
        return cast<member_class_t<decltype(Field)>>(
            find_by_key_raw(field_tag<Field>(), to_field_key(value)));
    }

    /// View-returning form of find_by_key.
    template <auto Field>
    std::optional<View<member_class_t<decltype(Field)>>> view_by_key(
        const member_value_t<decltype(Field)>& value) const;

    /// UNINDEXED multi-match lookup: every object whose `Field` -- declared
    /// via define_scan_fields() -- currently equals `value`. Named like
    /// find_by_key: `s.find_by_scan_field<&Order::qty>(5)`, and `Field` may
    /// also be a nullary const method.
    ///
    /// This is a SLOW SCAN, O(#ClassT objects) per query, comparing the
    /// field's actual typed value -- no index is consulted OR maintained, so
    /// a scan field costs nothing on the write side. Declaring it in
    /// define_scan_fields() is purely what makes it queryable: an undeclared
    /// field returns empty, exactly as find_by_key does for a field
    /// define_keys() never mentioned -- the three lookup families share that
    /// rule. For a field queried often enough to deserve an index, declare
    /// it in define_cached_fields() and use find_by_cached_field instead.
    template <auto Field>
    std::vector<const member_class_t<decltype(Field)>*> find_by_scan_field(
        const member_value_t<decltype(Field)>& value) const {
        using ClassT = member_class_t<decltype(Field)>;
        std::vector<const ClassT*> out;
        bool checked = false, declared = false;
        for_each<ClassT>([&](const ClassT& o) {
            if (!checked) {
                // define_scan_fields() is static per type: ask the first
                // object once, on behalf of the whole scan.
                checked = true;
                o.each_scan_field([&](const void* field, std::string) {
                    if (field == field_tag<Field>()) declared = true;
                });
            }
            if (!declared) return;
            // Field is either a data member or a nullary const method -- the
            // same two shapes member_class/member_value accept everywhere else.
            if constexpr (std::is_member_object_pointer_v<decltype(Field)>) {
                if (o.*Field == value) out.push_back(&o);
            } else {
                if ((o.*Field)() == value) out.push_back(&o);
            }
        });
        return out;
    }

    /// View-returning form of find_by_scan_field.
    template <auto Field>
    std::vector<View<member_class_t<decltype(Field)>>> view_by_scan_field(
        const member_value_t<decltype(Field)>& value) const;

    /// INDEXED multi-match lookup: every object whose `Field` -- declared via
    /// define_cached_fields() -- currently equals `value`. O(log n + #matches),
    /// backed by Root::by_cached_field, so it needs no scan; compare
    /// find_by_scan_field, the unindexed O(#ClassT) form for fields not
    /// worth an index. Named like find_by_key: `s.find_by_cached_field<&Order::qty>(5)`,
    /// with class and value type both deduced from the field itself. Empty if
    /// nothing matches -- or if the field was never declared cached (an
    /// undeclared field is invisible to this index, same as find_by_key).
    /// Result order is unspecified (index order, not insertion order).
    template <auto Field>
    std::vector<const member_class_t<decltype(Field)>*> find_by_cached_field(
        const member_value_t<decltype(Field)>& value) const {
        using ClassT = member_class_t<decltype(Field)>;
        std::vector<const ClassT*> out;
        if (!root_) return out;
        auto it = root_->by_cached_field.find(field_tag<Field>());
        if (it == root_->by_cached_field.end()) return out;
        const pmap::PersistentMap<Id>* bucket = it->second.get(to_field_key(value));
        if (!bucket) return out;
        bucket->for_each([&](const std::string&, Id id) {
            if (const ClassT* p = cast<ClassT>(find_raw(id))) out.push_back(p);
        });
        return out;
    }

    /// View-returning form of find_by_cached_field.
    template <auto Field>
    std::vector<View<member_class_t<decltype(Field)>>> view_by_cached_field(
        const member_value_t<decltype(Field)>& value) const;

    /// Visit every object of type T. O(#T objects): backed by a per-type
    /// index of every Id of that type (Root::by_type), not a full spine scan.
    template <class T, class F>
    void for_each(F&& f) const {
        if (!root_) return;
        auto it = root_->by_type.find(type_tag<T>());
        if (it == root_->by_type.end()) return;
        it->second.for_each([&](const std::string&, Id id) {
            if (const T* p = cast<T>(find_raw(id))) f(*p);
        });
    }

    /// Slow linear scan over every object of type T, keeping those for which
    /// `pred` returns true. There is no index behind this *filter* -- unlike
    /// find_by_key, it is O(number of T objects) by design. Reach for
    /// find_by_key when you have the exact indexed value instead of a
    /// predicate.
    template <class T, class Pred>
    std::vector<const T*> find_all(Pred&& pred) const {
        std::vector<const T*> out;
        for_each<T>([&](const T& o) {
            if (pred(o)) out.push_back(&o);
        });
        return out;
    }

    /// Same slow scan as find_all, but returns View<T>s bound to this snapshot.
    template <class T, class Pred>
    std::vector<View<T>> find_all_view(Pred&& pred) const;

    /// Slow scan: every object whose `Field` currently points at `target`,
    /// i.e. "who references this?" for an arbitrary Ref<U>/Opt<U> field. This
    /// is the read-side mirror of the writer's cascade-delete reverse index
    /// (Model::referrers_) -- but that index is writer-private and can never
    /// be handed to a reader, so this costs O(number of ClassT objects), not
    /// the writer's O(1). Named the same way find_by_key is:
    /// `s.for_each_referrer<&Order::account>(acct, f)`. The class is deduced
    /// from the field, and `target`'s required type (exactly the field's own
    /// Ref<U>/Opt<U>::target_type) is enforced by the parameter type itself,
    /// not a runtime static_assert. Works on ANY Ref<>/Opt<> field, declared
    /// cached or not -- for one queried often enough to be worth an index,
    /// declare it in define_cached_references() and use
    /// find_cached_referrers instead.
    template <auto Field, class F>
    void for_each_referrer(Ref<typename member_value_t<decltype(Field)>::target_type> target,
                           F&& f) const {
        using ClassT = member_class_t<decltype(Field)>;
        for_each<ClassT>([&](const ClassT& o) {
            if ((o.*Field).raw() == target.raw()) f(o);
        });
    }

    /// Same scan as for_each_referrer, collected into a vector.
    template <auto Field>
    std::vector<const member_class_t<decltype(Field)>*> find_referrers(
        Ref<typename member_value_t<decltype(Field)>::target_type> target) const {
        using ClassT = member_class_t<decltype(Field)>;
        std::vector<const ClassT*> out;
        for_each_referrer<Field>(target, [&](const ClassT& o) { out.push_back(&o); });
        return out;
    }

    /// View-returning form of for_each_referrer.
    template <auto Field, class F>
    void for_each_referrer_view(Ref<typename member_value_t<decltype(Field)>::target_type> target,
                                F&& f) const;

    /// View-returning form of find_referrers.
    template <auto Field>
    std::vector<View<member_class_t<decltype(Field)>>> find_referrers_view(
        Ref<typename member_value_t<decltype(Field)>::target_type> target) const;

    /// INDEXED counterpart of find_referrers: O(log n + #matches) instead of
    /// O(#ClassT objects), backed by Root::by_cached_reference. Only works
    /// for a field declared in define_cached_references() -- empty
    /// otherwise, same "undeclared is invisible" rule as find_by_cached_field
    /// (there is no silent fallback to the scan; call find_referrers by name
    /// for that). Named the same way: `s.find_cached_referrers<&Order::account>(acct)`.
    template <auto Field>
    std::vector<const member_class_t<decltype(Field)>*> find_cached_referrers(
        Ref<typename member_value_t<decltype(Field)>::target_type> target) const {
        using ClassT = member_class_t<decltype(Field)>;
        std::vector<const ClassT*> out;
        if (!root_) return out;
        auto it = root_->by_cached_reference.find(field_tag<Field>());
        if (it == root_->by_cached_reference.end()) return out;
        const pmap::PersistentMap<Id>* bucket = it->second.get(detail::id_key(target.raw()));
        if (!bucket) return out;
        bucket->for_each([&](const std::string&, Id id) {
            if (const ClassT* p = cast<ClassT>(find_raw(id))) out.push_back(p);
        });
        return out;
    }

    /// View-returning form of find_cached_referrers.
    template <auto Field>
    std::vector<View<member_class_t<decltype(Field)>>> view_cached_referrers(
        Ref<typename member_value_t<decltype(Field)>::target_type> target) const;

    // ---- views ------------------------------------------------------------
    //
    // A View<T> is an object bound to the snapshot it came from. Traversal
    // through a view always uses *that* snapshot, so you cannot accidentally
    // resolve a v40 object's reference against a v44 root -- a mistake the raw
    // resolve() API happily compiles. See View below.

    /// Bind an object you already read from THIS snapshot. Unchecked --
    /// pairing an object from one snapshot with another compiles and is
    /// exactly the version-mixing bug views exist to prevent, so only pass
    /// objects this snapshot handed you.
    template <class T>
    View<T> view(const T& obj) const noexcept;

    /// Null if the handle is stale (deleted, or its slot recycled).
    template <class T>
    std::optional<View<T>> view(Ref<T> r) const;

    /// Visit every object of type T as a view.
    template <class T, class F>
    void for_each_view(F&& f) const;

    // ---- write side ---------------------------------------------------------

    /// Start a Transaction based on this Snapshot -- equivalent to
    /// `model.begin(snapshot)`, but doesn't require holding the Model
    /// separately. Handy when a Snapshot is what you already have (e.g. one
    /// you're also using for analysis, so you want the transaction's view to
    /// match it exactly). The Snapshot itself is untouched -- copied into the
    /// new Transaction's base(), same as Model::begin(Snapshot) does.
    Transaction begin() const;

    /// Untyped escape hatch -- for change events, which are heterogeneous
    /// (a Change::id could be any type). Generation-checked exactly like the
    /// typed find()/resolve() built on it: null for an absent slot OR a
    /// stale id whose generation no longer matches (see Chunk::gen). Prefer
    /// find<T>()/resolve<T>() when you know the type -- this is what they
    /// call underneath, before the tag check.
    const ObjectBase* find_raw(Id id) const noexcept;

    /// Untyped counterpart of find_by_key -- the `key` string is only unique
    /// within `field`'s own keyspace (see field_tag), never global, so the
    /// field must be supplied to disambiguate. Used where the field isn't
    /// known at compile time; find_by_key<Field>(value) is the typed,
    /// preferred entry point that calls this.
    const ObjectBase* find_by_key_raw(const void* field, const std::string& key) const;

private:
    friend class Model;
    friend class Transaction;

    /// RAII registration of this snapshot's version in Model::live_ (defined
    /// in model.cpp). Its destructor is what tells the reaper a version may
    /// have become reclaimable. Held by shared_ptr so COPIES of a Snapshot
    /// share ONE registration -- the version is released exactly once, when
    /// the last copy drops.
    struct Lease;

    /// One virtual call plus a pointer compare. No RTTI, no dynamic_cast.
    template <class T>
    static const T* cast(const ObjectBase* o) noexcept {
        if (!o || o->tag() != type_tag<T>()) return nullptr;
        return static_cast<const T*>(o);
    }

    std::shared_ptr<const Root> root_;  ///< the immutable version everything above reads;
                                        ///< shared with the Model and other snapshots
    std::shared_ptr<Lease> lease_;      ///< keeps root_'s version registered while any copy lives
};

// ---------------------------------------------------------------------------
// Change events
// ---------------------------------------------------------------------------

/// What happened to one Id in one committed transaction. Updated covers any
/// reinstall of the object -- a field write via Transaction::update() AND a
/// cascade-nulled Opt<> field -- so subscribers cannot tell those apart
/// (they see the final value either way). A given Id (generation included)
/// appears at most once per changeset.
enum class ChangeKind : std::uint8_t { Created, Updated, Deleted };

/// Changesets span every type in the model, so they carry the untyped Id --
/// and, since a Deleted change's object is already gone by the time a
/// subscriber looks (there is no Snapshot it can still be found in), `tag`
/// too: it's the one piece of type information that survives the object
/// itself. Compare it against type_tag<T>() directly, same as any other tag
/// check in this API. For a live (Created/Updated) change you can still go
/// the long way -- snapshot.find_raw(c.id) plus a tag check, or
/// snapshot.find<T>(Ref<T>(c.id)), or Transaction::peek_as<T>(c.id) -- but
/// `tag` means you never need a Snapshot just to find out WHAT changed.
struct Change {
    Id id;           ///< which object; wrap in Ref<T>/use peek_as<T> to read it (typed)
    ChangeKind kind;  ///< what happened to it -- see ChangeKind's caveats
    TypeTag tag;      ///< the object's type, valid even for Deleted; compare to type_tag<T>()
};

/// Registered once via Model::set_pre_commit, called on every try_commit()
/// thereafter that reaches the apply phase -- for an external system that
/// needs to see (and can veto) a transaction before it publishes, without
/// every call site having to remember to check by hand. Runs synchronously,
/// under the model's commit lock, once the transaction's changes have reached
/// their FINAL, fully-resolved form -- including cascade deletes, since those
/// are resolved earlier in the same apply phase (see CLAUDE.md invariant 9).
///
/// Returning false vetoes the commit: try_commit() unwinds everything applied
/// so far and returns CommitStatus::Vetoed; nothing was published, and the
/// Transaction is spent (its local overlay was moved from during apply) --
/// begin() a fresh one to retry. The hook must NOT throw: this project builds
/// with -fno-exceptions (see CLAUDE.md), so a throw is std::terminate, not an
/// error path. Report "no" by returning false.
///
/// Read-only by convention: the hook sees the resolved changeset and can
/// peek_as<T> any of it via the Model reference, but must not call
/// try_commit() itself or otherwise begin a new transaction -- this runs
/// inside an existing try_commit(), which already holds the commit lock.
/// The same lock also rules out calling current_version(),
/// retired_pending(), exhausted_slots(), or set_pre_commit() from the hook
/// (self-deadlock on the non-recursive commit_mu_); snapshot() is fine, and
/// yields the still-current PRE-commit version, since the transaction being
/// inspected has not published yet.
///
/// Unchanged by, and unaware of, PreTransactionsFn below: if a
/// pre-transactions phase ran first, this hook still runs at exactly the
/// same point (after the MAIN transaction's own apply) seeing exactly the
/// same thing (the main transaction's own resolved changeset) -- it has no
/// way to tell whether pre-transactions ran, and doesn't need to.
using PreCommitFn = std::function<bool(Model&, const std::vector<Change>&)>;

/// Registered once via Model::set_pre_transactions, called on every
/// try_commit() attempt (if installed) IMMEDIATELY after commit_mu_ is
/// acquired -- before the main Transaction is touched in ANY way: not
/// conflict-checked, not applied. This is the ONLY point at which
/// Model::run_pre_commit_transaction() may be called; use it to run zero or
/// more OTHER Transactions first, atomically with respect to every other
/// writer (commit_mu_ never releases in between) and atomically with
/// respect to the main transaction (which hasn't started yet).
///
/// Each call to run_pre_commit_transaction() is a REAL, independent commit
/// -- conflict-checked against the changelog (which now includes every
/// EARLIER pre-transaction this same phase already published) and, if it
/// passes, published for real: its own version, its own changelog entry,
/// its own subscriber Update. There is no undo once one of them succeeds
/// (CLAUDE.md invariant 3: published state is immutable) -- if a LATER
/// pre-transaction in the same phase then fails, try_commit() returns
/// CommitStatus::PrecommitConflict and skips the main transaction, but
/// every pre-transaction that already published stays published. Design
/// for that: make each pre-transaction a complete, independently-correct
/// unit of work, not a step that depends on a later one succeeding.
///
/// Once any call fails (returns anything other than CommitStatus::
/// Committed), calling it again is a hook bug (assert): the attempt is
/// already doomed to CommitStatus::PrecommitConflict regardless of what
/// runs after, so a well-behaved hook checks each return value and stops.
///
/// Because the main transaction is applied AFTER every pre-transaction has
/// already published, it gets its conflict-checking for free: its own
/// check_id_overlap (against the now-extended changelog) and its own
/// apply-time Ref<> validation (against the now-current state) will report
/// CommitStatus::Conflict if it touches -- or references -- anything a
/// pre-transaction just changed, exactly as if that pre-transaction were an
/// ordinary commit from another writer. No new conflict-detection code is
/// needed for this; it is the existing multi-writer OCC mechanism, applied
/// to writes this SAME try_commit() call happened to make first.
///
/// This hook is entirely separate from PreCommitFn/set_pre_commit -- see
/// PreCommitFn's own comment for why the two are not merged into one call.
using PreTransactionsFn = std::function<void(Model&)>;

/// One delivery to a subscriber: a consistent state plus what changed since
/// the previous delivery -- read the changed objects out of THIS update's
/// own snapshot, never a fresh Model::snapshot() (which may already be
/// newer, and would tear the "state matches changes" pairing).
struct Update {
    Snapshot snapshot;            ///< state as of these changes; pins its version until dropped
    std::vector<Change> changes;  ///< every id that changed since the last delivery
    bool coalesced = false;       ///< true if intermediate versions were folded away
};

/// One bounded queue per subscriber, drained on the subscriber's own thread.
///
/// On overflow the writer coalesces rather than blocking or growing: newest
/// snapshot wins, changesets are unioned per Id, intermediate snapshots dropped.
/// Every queued event pins a snapshot, and every pinned snapshot pins the objects
/// retired since -- so an unbounded queue is a memory leak with a slow fuse.
/// Subscribers see "current state plus everything that changed since you last
/// looked", not every intermediate version.
class Subscription {
public:
    /// `depth` = max queued Updates before overflow coalesces. Deeper keeps
    /// more distinct intermediate versions for a slow consumer -- and pins
    /// that many snapshots. Obtain via Model::subscribe(), not directly.
    explicit Subscription(std::size_t depth) : cap_(depth) {}

    bool wait(Update& out);       ///< blocks; false once the model shuts down
    bool try_drain(Update& out);  ///< non-blocking

private:
    friend class Model;

    void push(Update u);         ///< called by try_commit() at publish; coalesces when full
    void collapse(Update tail);  ///< the overflow path: merge queue + tail into ONE Update
    void close();                ///< Model::shutdown(): wake blocked wait()ers to return false

    std::mutex m_;                ///< guards everything below; never held while user code runs
    std::condition_variable cv_;  ///< signals wait(): queue non-empty, or closed
    std::deque<Update> q_;        ///< pending deliveries, oldest first; length <= cap_
    std::size_t cap_;             ///< the constructor's depth
    bool closed_ = false;         ///< set once by close(); wait() drains what's left, then false
};

// ---------------------------------------------------------------------------
// Model
// ---------------------------------------------------------------------------

/// The store itself: owns every committed object, the writer-side indexes,
/// and the background reaper thread. One instance per object graph; all the
/// other types in this header (Snapshot, Transaction, Subscription, View)
/// are windows onto one Model and must not outlive it.
class Model {
public:
    Model();   ///< starts at version 1 (empty), spawns the reaper thread
    ~Model();  ///< joins the reaper, frees everything. Every Snapshot,
               ///< Transaction, Subscription, and View must already be gone
               ///< -- a Lease releasing against a destroyed Model is UB.

    // Not copyable (owns a thread, mutexes, and every object's identity) and
    // not movable either: Snapshots/Transactions hold interior pointers back
    // to this Model, so its address is part of the contract.
    Model(const Model&) = delete;
    Model& operator=(const Model&) = delete;

    // ---- read side (any thread) ---------------------------------------------

    /// The current committed state, O(1): a lock-free root load plus version
    /// registration under the small ver_mu_ -- never commit_mu_, so readers
    /// never wait on writers (see CLAUDE.md invariant 10). Take one per unit
    /// of work and drop it; holding one pins retired objects (see Snapshot).
    Snapshot snapshot();

    /// Register for change events; see Subscription for the queue/coalescing
    /// contract and the queue_depth tradeoff. Subscribe BEFORE shutdown();
    /// the returned object stays valid until dropped.
    std::shared_ptr<Subscription> subscribe(std::size_t queue_depth = 8);
    void shutdown();  ///< wakes blocked subscribers so their threads can exit

    // ---- write side (any thread -- try_commit() serializes internally) -----

    /// Start a transaction based on the current committed state. Building it
    /// (create/update/remove/peek) touches only the Transaction's own local
    /// state -- never Model's shared state, never any lock -- so any number
    /// of threads can each build one fully in parallel with zero contention.
    Transaction begin();

    /// Start a transaction based on an explicit, already-held Snapshot (e.g.
    /// one you're also using for analysis, so you want the transaction's view
    /// to match exactly).
    Transaction begin(Snapshot base);

    /// Attempt to publish a Transaction. This is the ONE serialization point:
    /// briefly locks the whole model while it (1) checks the transaction's
    /// write-set against everything committed since its base version, (2)
    /// applies creates/updates/resolves delete-intents into cascade fan-out
    /// against the single authoritative reverse index, re-validating every
    /// Ref<> against the LATEST state (not just the transaction's base), and
    /// (3), if nothing conflicted, publishes atomically. See CommitResult and
    /// CLAUDE.md's "try_commit() phase order" for the exact algorithm.
    ///
    /// A conflicting, vetoed, or invalid attempt leaves `txn` fully consumed
    /// either way (its local overlay has already been moved from during
    /// apply) -- discard it and begin() a fresh Transaction to retry.
    CommitResult try_commit(Transaction& txn);

    /// Install (or clear, with {}) the pre-commit hook. See PreCommitFn.
    /// Takes the commit lock, so it is safe to call while other threads
    /// commit -- the hook swaps in between commits, never mid-commit. Never
    /// call it from inside the hook itself: commit_mu_ is not recursive.
    void set_pre_commit(PreCommitFn fn) {
        std::lock_guard lk(commit_mu_);
        pre_commit_ = std::move(fn);
    }

    /// Install (or clear, with {}) the pre-transactions hook. See
    /// PreTransactionsFn. Same locking contract as set_pre_commit: takes
    /// commit_mu_ itself, so call it between commits, never from inside
    /// either hook.
    void set_pre_transactions(PreTransactionsFn fn) {
        std::lock_guard lk(commit_mu_);
        pre_transactions_ = std::move(fn);
    }

    /// Callable ONLY from inside a running PreTransactionsFn callback
    /// (asserted via in_pre_transactions_phase_ -- commit_mu_ is not
    /// recursive, so this must NOT itself try to lock it; it runs already
    /// holding the lock the enclosing try_commit() took). Applies and
    /// publishes `txn` as a complete, independent commit: conflict-checked
    /// against everything committed so far (including earlier
    /// pre-transactions this same phase already published), then, if that
    /// passes, applied and published for real -- its own version bump, its
    /// own changelog entry, its own subscriber Update. PreCommitFn is NOT
    /// invoked for it; only the main transaction's veto hook runs (see
    /// PreCommitFn).
    ///
    /// Once a call returns anything other than CommitStatus::Committed, the
    /// enclosing try_commit() is already doomed to
    /// CommitStatus::PrecommitConflict -- calling this again afterward is a
    /// hook bug (asserted). Check the return value and stop.
    CommitResult run_pre_commit_transaction(Transaction& txn);

    /// Reported via CommitResult::error (status == Invalid) when a
    /// Transaction's own creates or updates would violate referential
    /// integrity. Not an exception -- this project builds with
    /// -fno-exceptions; see CLAUDE.md. Most such failures are classified as a
    /// Conflict by try_commit() (see bad_target) rather than as Invalid --
    /// Invalid only ever means a target that never existed even at the
    /// transaction's own base (or a null non-nullable Ref, or a stray local
    /// id), which is a genuine transaction-building bug, not contention.
    struct IntegrityError {
        std::string message;
        Id bad_target;  ///< Id{} if not applicable (e.g. a null non-nullable Ref)
    };

    /// Objects retired but not yet freed, because a reader might still see them.
    /// Serviced by the background reaper, so this counts its backlog.
    /// retired_ is commit_mu_-protected (unlike the single-writer sibling,
    /// where it was safe to read unsynchronized from the one writer thread
    /// that owned it -- here any thread may call this while another holds
    /// commit_mu_, so the read needs the same lock apply() does).
    std::size_t retired_pending() const {
        std::lock_guard lk(commit_mu_);
        return retired_pending_.load(std::memory_order_relaxed) + retired_.size();
    }

    /// Block until the background reaper has freed everything currently
    /// reclaimable (i.e. everything whose version is below the live watermark).
    /// Reclamation is normally asynchronous; this is a barrier for shutdown
    /// sequences and deterministic tests. Returns the number still pinned by live
    /// snapshots (which cannot be freed until those snapshots drop).
    std::size_t wait_for_reclamation();

    /// The latest committed version -- may be stale the instant it returns
    /// (another thread can commit immediately after). For observability and
    /// tests; decisions about state belong on a Snapshot, whose version()
    /// stays consistent with what it actually shows. Takes commit_mu_, so
    /// never call it from a pre-commit hook (self-deadlock; see PreCommitFn).
    std::uint64_t current_version() const {
        std::lock_guard lk(commit_mu_);
        return version_;
    }

    /// Slots permanently withdrawn because their generation was exhausted.
    /// Expected to stay near zero; a climbing value means extreme reuse of a few
    /// slots.
    std::size_t exhausted_slots() const {
        std::lock_guard lk(commit_mu_);
        return exhausted_slots_;
    }

    /// Testing seam: force a slot's generation, so generation-exhaustion can be
    /// exercised without actually cycling a slot four billion times. Not for
    /// production use.
    void debug_set_generation(std::uint32_t slot, std::uint32_t gen);

    /// Testing seam: current changelog retention depth, for asserting it gets
    /// pruned once no open Transaction/Snapshot needs it. Not for production use.
    std::size_t debug_changelog_size() const;

    /// Testing seam: (version, refcount) for every currently-live snapshot
    /// registration (readers' Snapshots AND every open Transaction::base()).
    /// The minimum key is the reclamation watermark. Not for production use.
    std::vector<std::pair<std::uint64_t, int>> debug_live_versions() const;

private:
    friend struct Snapshot::Lease;
    friend class Transaction;

    /// One incoming edge in the reverse index (referrers_): "`from`'s
    /// `field` points at me." `nullable` is the cascade decision, captured
    /// at edge creation: a nullable referrer gets its field nulled when the
    /// target dies; a non-nullable one dies too.
    struct RefEdge {
        Id from;
        const void* field;
        bool nullable;
    };

    /// One committed transaction's resolved changeset, retained for
    /// try_commit()'s id-overlap conflict check. See prune_changelog: safe to
    /// discard once no open Transaction's base_version() (or reader Snapshot)
    /// could still need it, which is exactly the existing live_ watermark --
    /// no separate retention bookkeeping needed.
    struct ChangelogEntry {
        std::uint64_t version;
        std::vector<Change> changes;
    };

    // All of the following run only under commit_mu_ (invariant 7), except
    // release_version / reaper_loop / enqueue_retired, which have their own
    // locking noted below.

    /// The writer's view of the LATEST state (generation-checked, like
    /// Snapshot::find_raw but against spine_, which may be mid-commit).
    /// This is what validate() checks against -- "re-validated against
    /// latest, not just base" falls out of using peek() here.
    const ObjectBase* peek(Id id) const;

    /// First touch of a chunk per attempt clones it (tracked in dirty_);
    /// later writes hit the clone in place. Published chunks stay immutable.
    Chunk* cow(std::uint32_t chunk_index);

    /// Pop the LIFO free list (retiring generation-exhausted slots as they
    /// surface -- see Id's width note), else mint a fresh slot. Undo-logged.
    std::uint32_t alloc_slot();

    std::optional<IntegrityError> validate(const ObjectBase* o) const;  // nullopt == valid

    // Reverse-index (referrers_) maintenance: add/drop an object's whole
    // outgoing edge set (create/delete), or diff before->after (update).
    // Every edit is undo-logged -- a rollback that missed one would leave a
    // phantom or missing edge for a LATER cascade to resolve against.
    void add_out_refs(const ObjectBase* o);
    void drop_out_refs(const ObjectBase* o);
    void reconcile_referrer_edges(const ObjectBase* before, const ObjectBase* after);

    // Same trio for the unique key index (by_field_)...
    void add_field_keys(const ObjectBase* o);
    void drop_field_keys(const ObjectBase* o);
    void reconcile_field_keys(const ObjectBase* before, const ObjectBase* after);

    // ...and for the multimap index (by_cached_field_).
    void add_cached_fields(const ObjectBase* o);
    void drop_cached_fields(const ObjectBase* o);
    void reconcile_cached_fields(const ObjectBase* before, const ObjectBase* after);

    // Same trio again, for the reverse-lookup multimap (by_cached_reference_)
    // -- keyed by each declared field's TARGET, walked via
    // each_cached_reference() instead of each_cached_field().
    void add_cached_references(const ObjectBase* o);
    void drop_cached_references(const ObjectBase* o);
    void reconcile_cached_references(const ObjectBase* before, const ObjectBase* after);

    /// Mark an object invisible from the NEXT version on; the reaper frees
    /// it once no live snapshot is older than that (invariant 4). Never
    /// delete a published object directly.
    void retire(const ObjectBase* o);

    /// Lease's destructor: deregister a version from live_ (ver_mu_) and
    /// nudge the reaper (reap_mu_) -- taken SEQUENTIALLY, not nested, to
    /// stay off the lock-order cycle (see invariant 10).
    void release_version(std::uint64_t v);

    void reaper_loop();  ///< body of the background reaper thread (reap_mu_, briefly ver_mu_)

    /// Hand a commit's retirees to the reaper (reap_mu_). Batched so a large
    /// cascade is one lock acquisition, and freeing happens off-thread.
    void enqueue_retired(std::vector<std::pair<std::uint64_t, const ObjectBase*>> batch);

    // Undo log. Every mutation of writer-private state during a try_commit()
    // attempt pushes its inverse here; rollback_apply() runs the inverses in
    // reverse if that attempt fails; a successful try_commit() clears the log.
    // Kept as closures so each write op records its own undo inline, which is
    // far harder to get out of sync than a parallel variant type. The log is
    // proportional to the changes made, never to model size.
    void log(std::function<void()> undo) { undo_.push_back(std::move(undo)); }

    /// Point a slot at an object (or null) with a new generation, through
    /// cow(); logs the exact inverse (previous object + generation).
    void set_slot(std::uint32_t slot, const ObjectBase* obj, std::uint32_t gen);

    // ---- try_commit() internals (ALL require commit_mu_ already held) ------
    // apply_* return nullopt on success, or the integrity violation that
    // aborted this attempt -- the caller must then rollback_apply().
    std::optional<IntegrityError> apply_create(std::unique_ptr<ObjectBase> o,
                                               std::unordered_map<std::uint32_t, Id>& remap);
    std::optional<IntegrityError> apply_update(std::unique_ptr<ObjectBase> clone,
                                               std::unordered_map<std::uint32_t, Id>& remap);
    std::vector<Id> remove_raw(Id id);  // cascade BFS, called from try_commit()'s apply phase
    // remove_raw's helper for a NULLABLE referrer: clone + install, so the
    // caller can null_ref() the field that pointed at the victim, then
    // reconcile immediately. See the .cpp for why reconciliation must happen
    // in the caller, after null_ref() -- not in here.
    ObjectBase* clone_for_cascade_null(Id id);

    /// The conflict check: every slot this transaction updated or intends to
    /// remove, tested against every changelog entry newer than its base.
    /// Read-only; runs before anything is applied, so a Conflict here costs
    /// no rollback. Creates can't conflict (fresh slots) and aren't checked.
    std::vector<Id> check_id_overlap(const Transaction& txn) const;

    /// Drop changelog entries at or below the live watermark -- no open
    /// Transaction (its base pins a version in live_) can need them again.
    void prune_changelog();

    void rollback_apply();  // unwinds one failed try_commit() apply attempt

    // ---- shared by try_commit() and run_pre_commit_transaction() -----------
    // Both publish a Transaction as a real, independent commit while already
    // holding commit_mu_; apply_transaction_contents/classify_apply_failure/
    // check_and_apply/publish_now are that shared machinery, factored out so
    // a pre-transaction gets EXACTLY the same conflict/apply/publish behavior
    // as the main transaction, with nothing bespoke to keep in sync.

    /// The creates-then-updates-then-removes apply loop (see try_commit()'s
    /// phase-order comment for why that order matters). Returns nullopt on
    /// success; the caller must then check changes_ and eventually publish
    /// or, on failure, pass the returned error to classify_apply_failure().
    std::optional<IntegrityError> apply_transaction_contents(
        Transaction& txn, std::unordered_map<std::uint32_t, Id>& remap);

    /// Unwinds the failed apply attempt (rollback_apply()) and turns the
    /// IntegrityError into the right CommitResult: Conflict(RefIntegrity) if
    /// the dangling target existed at txn's own base (someone else deleted it
    /// concurrently), Invalid otherwise (a genuine transaction-building bug).
    CommitResult classify_apply_failure(IntegrityError err, const Transaction& txn);

    /// check_id_overlap() + apply_transaction_contents(), collapsed into one
    /// call: nullopt means "proceed, remap is populated, changes_ may or may
    /// not be empty"; otherwise the CommitResult IS the final result (a
    /// Conflict from either the overlap check or classify_apply_failure()).
    std::optional<CommitResult> check_and_apply(Transaction& txn,
                                                 std::unordered_map<std::uint32_t, Id>& remap);

    /// The publish tail: version bump, new Root, atomic store under ver_mu_,
    /// subscriber notify, retirees handed to the reaper, changelog append,
    /// scratch cleared. Always succeeds -- by the time it's called, nothing
    /// left to reject. Consumes changes_ (member scratch) into the result.
    CommitResult publish_now(std::unordered_map<std::uint32_t, Id> remap);

    /// check_and_apply() + publish_now(), with NO veto-hook seam -- used only
    /// by run_pre_commit_transaction(), which must never invoke pre_commit_
    /// for a pre-transaction (that hook is reserved for the main
    /// transaction). try_commit() does NOT call this: the main transaction
    /// still needs the veto seam between apply and publish, so it inlines
    /// check_and_apply() + publish_now() itself. Requires commit_mu_ already
    /// held.
    CommitResult commit_locked(Transaction& txn);

    // ---- read/publish path -------------------------------------------------
    // root_ is atomic so snapshot() acquires the current version with a lock-free
    // load -- no shared mutex on the hot read path. Version bookkeeping (for the
    // reclamation watermark) still needs a short critical section, but it is
    // split onto its own small mutex (ver_mu_) so it never contends with a
    // commit's apply work or a reader's actual traversal.
    std::atomic<std::shared_ptr<const Root>> root_;

    mutable std::mutex ver_mu_;
    std::map<std::uint64_t, int>
        live_;  ///< live snapshot (incl. txn base) versions; min = watermark

    std::mutex subs_mu_;                              ///< guards subs_ only; publish copies the
                                                      ///< list out so pushes run without it held
    std::vector<std::shared_ptr<Subscription>> subs_;  ///< every live subscriber; shared_ptr so a
                                                       ///< subscriber outliving shutdown() is safe

    // ---- background reaper --------------------------------------------------
    // Retired objects are handed to a dedicated thread rather than freed inline
    // in try_commit(), so a large cascade never stalls a commit, and destructors
    // run off both the committing thread and any reader thread.
    std::thread reaper_;               ///< started by the ctor, joined by the dtor
    std::mutex reap_mu_;               ///< guards everything below except retired_pending_
    std::condition_variable reap_cv_;  ///< wakes the reaper: work arrived, or stopping
    std::condition_variable reap_done_cv_;  ///< wakes wait_for_reclamation(): a pass finished
    std::vector<std::pair<std::uint64_t, const ObjectBase*>> reap_queue_;
    ///< ^ (version at which each object became invisible, object); freed once
    ///< the live watermark reaches that version
    std::atomic<std::size_t> retired_pending_{0};  ///< reaper backlog, for observability
    std::uint64_t reap_done_round_ = 0;            ///< bumped after each reap pass
    bool dirty_reap_ = false;                      ///< a reap pass is due
    bool reaper_stop_ = false;                     ///< dtor -> reaper: drain and exit

    // ---- commit-lock-protected state ----------------------------------------
    // Touched ONLY by whichever thread currently holds commit_mu_, only from
    // inside try_commit(). This is exactly the old single-writer-thread state
    // (referrers_, by_type_, ...), unchanged in shape -- it's now protected by
    // an actual mutex instead of "single-threaded by convention," which is
    // strictly safer, not a rewrite. See CLAUDE.md invariant 7 and the lock
    // order rule (commit_mu_ -> ver_mu_ -> reap_mu_, never reversed).
    mutable std::mutex commit_mu_;
    std::uint64_t version_ = 0;  ///< 64-bit because it's bounded by TIME, not population:
                                 ///< it increments forever, and at 100 commits/sec a uint32
                                 ///< would wrap in ~16 months of uptime. See Id's width note
                                 ///< for the full 32-vs-64 rule.
    std::vector<std::shared_ptr<const Chunk>> spine_;  ///< the writer's working spine; COWed
                                                       ///< chunks land here, published via Root
    std::unordered_set<std::uint32_t> dirty_;  ///< chunks already cloned THIS attempt (see cow());
                                               ///< cleared per attempt so publish stays immutable
    // The writer's working copies of Root's three indexes -- same persistent
    // structures, so publishing them into a new Root is a cheap map copy.
    std::unordered_map<TypeTag, pmap::PersistentMap<Id>> by_type_;
    std::unordered_map<const void*, pmap::PersistentMap<Id>> by_field_;
    std::unordered_map<const void*, pmap::PersistentMap<pmap::PersistentMap<Id>>> by_cached_field_;
    std::unordered_map<const void*, pmap::PersistentMap<pmap::PersistentMap<Id>>> by_cached_reference_;

    /// The reverse index driving cascade delete: target SLOT (bare index --
    /// only the live generation of a slot can ever be referenced, so the
    /// full Id would be redundant) -> every edge pointing at it. Writer-only
    /// and mutable in place; the one structure that could never be handed to
    /// readers, and the reason cascade resolution must happen at commit time
    /// (invariant 8). Linear scan per target; see CLAUDE.md scope notes.
    std::unordered_map<std::uint32_t, std::vector<RefEdge>> referrers_;

    std::vector<std::uint32_t> free_slots_;  ///< recycled slots, LIFO -- reuse concentrates on
                                             ///< hot slots, keeping the spine dense
    std::uint32_t next_slot_ = 0;            ///< high-water mark: next never-used slot
    std::size_t exhausted_slots_ = 0;        ///< see exhausted_slots() accessor
    std::vector<Change> changes_;  ///< scratch: this attempt's resolved changeset
    std::vector<std::pair<std::uint64_t, const ObjectBase*>> retired_;
    ///< ^ this attempt's retirees, same shape as reap_queue_: handed to the
    ///< reaper on publish, drained back out by the undo log on rollback
    PreCommitFn pre_commit_;  ///< empty = no hook; swapped only under commit_mu_ (set_pre_commit)
    std::deque<ChangelogEntry>
        changelog_;  ///< for try_commit()'s conflict check; see prune_changelog

    PreTransactionsFn pre_transactions_;  ///< empty = no hook; swapped only under commit_mu_
                                          ///< (set_pre_transactions)
    bool in_pre_transactions_phase_ = false;  ///< guards run_pre_commit_transaction(): true only
                                              ///< while pre_transactions_(*this) is on the stack
    bool precommit_failed_ = false;   ///< a pre-transaction this attempt already failed --
                                      ///< try_commit() will report PrecommitConflict and skip
                                      ///< the main transaction; a further run_pre_commit_
                                      ///< transaction() call this attempt is a hook bug (assert)
    std::unique_ptr<CommitResult> precommit_failure_;  ///< the failing CommitResult, reported back
                                                       ///< as PrecommitConflict. unique_ptr, not
                                                       ///< optional: CommitResult is only forward-
                                                       ///< declared this far up (see line ~805) --
                                                       ///< optional<T> needs T complete as a member,
                                                       ///< unique_ptr<T> doesn't.

    // Undo log and the objects created this attempt (which rollback_apply()
    // must delete, since they were never published and nothing else owns them).
    // Scratch: cleared at the start of every try_commit() attempt.
    std::vector<std::function<void()>> undo_;
    std::vector<const ObjectBase*> txn_created_;
};

// ---------------------------------------------------------------------------
// Multi-writer commit results
// ---------------------------------------------------------------------------

/// How a try_commit() attempt ended. Only Committed published anything; the
/// other four unwound completely (Conflict/Vetoed/Invalid) or never touched
/// the main transaction at all (PrecommitConflict), and differ in what to do
/// next:
///
///   Committed         the transaction is now the latest version.
///   Conflict          lost a race with a concurrent commit -- not a bug.
///                     RETRY: begin() a fresh Transaction (its new base sees
///                     the winner) and rebuild. See ConflictInfo for what
///                     collided. Also reported when the main transaction
///                     collides with a pre-transaction THIS SAME attempt
///                     already published (see PreTransactionsFn) -- from the
///                     main transaction's perspective that pre-transaction
///                     is just another concurrent commit, indistinguishable
///                     from one made by a different thread.
///   Vetoed            the pre-commit hook said no. Retrying unchanged will
///                     just be vetoed again; whatever the hook checks must
///                     change first.
///   Invalid           the Transaction itself was malformed (a null
///                     non-nullable Ref, a target dead even at the
///                     transaction's own base, or a Ref holding a stray
///                     local id) -- a transaction-building bug at the call
///                     site, not contention; retrying unchanged cannot
///                     succeed. See CommitResult::error for the specifics.
///                     This status exists because the project has no
///                     exceptions to throw (see CLAUDE.md).
///   PrecommitConflict a Model::run_pre_commit_transaction() call inside the
///                     installed PreTransactionsFn did not return Committed.
///                     The main transaction was never even conflict-checked,
///                     let alone applied -- `txn` is unconsumed and may be
///                     retried as-is (unlike every other non-Committed
///                     status, which spends the local overlay during apply).
///                     CommitResult::conflict/error carry through from
///                     whichever pre-transaction failed, so the caller can
///                     tell why. IMPORTANT: any EARLIER pre-transaction in
///                     the same phase that already succeeded stays published
///                     regardless (invariant 3: published state is
///                     immutable) -- PrecommitConflict means only "the main
///                     transaction didn't run," not "nothing happened."
enum class CommitStatus { Committed, Conflict, Vetoed, Invalid, PrecommitConflict };

/// Which of the two conflict rules fired (see CLAUDE.md's OCC contract:
/// "a commit succeeds if no one touched the same ids and every Ref<T> is
/// still valid"):
///   IdSetOverlap  a commit newer than this transaction's base touched an id
///                 this transaction updated or removed.
///   RefIntegrity  an object this transaction created/updated references a
///                 target that a CONCURRENT commit deleted (it was alive at
///                 the transaction's base -- that's what distinguishes this
///                 from Invalid).
enum class ConflictReason { IdSetOverlap, RefIntegrity };

/// The specifics behind CommitStatus::Conflict -- enough to log, or to
/// decide a retry is pointless (e.g. your target is simply gone).
struct ConflictInfo {
    ConflictReason reason;
    std::vector<Id> ids;  ///< the specific id(s) that conflicted
};

/// Everything try_commit() has to say about one attempt. Check `status`
/// first; every other field documents which statuses make it meaningful.
struct CommitResult {
    CommitStatus status;  ///< what happened -- gates the meaning of every field below

    /// Valid only if status == Committed; deliberately NULL on every failure
    /// status. A Snapshot pins its version against the reaper, and failures
    /// are the HOT path under contention (retry loops) -- if every
    /// conflicted attempt handed back a pinned snapshot, a caller who holds
    /// CommitResults (perfectly natural for logging/diagnostics) would
    /// silently stall reclamation, the exact fat-ref failure mode View<T>'s
    /// docs warn about. It wouldn't help a retry either: the only correct
    /// base for the next attempt is whatever is latest at retry time, and
    /// begin() takes that itself. To inspect the state that beat you (e.g.
    /// to look up conflict->ids), call Model::snapshot() -- one call, and
    /// the pinning cost becomes opt-in instead of paid by every rejection.
    /// One wrinkle: the empty-transaction fast path returns Committed with
    /// the transaction's own base(), which may be stale relative to other
    /// writers -- nothing changed, so any version is "after" that commit.
    Snapshot snapshot;

    std::vector<Change> changes;           ///< FULL resolved changeset, incl. cascade deletes;
                                           ///< empty unless status == Committed
    std::optional<ConflictInfo> conflict;  ///< set if status == Conflict; also set if status ==
                                           ///< PrecommitConflict and the failing pre-transaction's
                                           ///< own outcome was itself a Conflict (forwarded through
                                           ///< as-is -- see CommitStatus::PrecommitConflict)

    /// local Id::index (kLocalIdBit set) -> real Id, populated only when
    /// status == Committed. A Ref<T>/Opt<T> returned by Transaction::create()
    /// holds a LOCAL id -- meaningful only inside that transaction, and only
    /// until try_commit() is called. Once try_commit() returns, that local id
    /// is spent: it must never be stored, compared, or handed to a Snapshot,
    /// and it must never be reused across a DIFFERENT Transaction (a local id
    /// is only unique within the transaction that minted it). Use resolve()
    /// below to translate it into the real id this object now has -- do not
    /// read this map directly.
    std::unordered_map<std::uint32_t, Id> local_remap;

    /// Set if status == Invalid: what was wrong with the transaction. The
    /// whole attempt was unwound; nothing was published. Also set if status
    /// == PrecommitConflict and the failing pre-transaction's own outcome
    /// was itself Invalid (forwarded through as-is -- see
    /// CommitStatus::PrecommitConflict).
    std::optional<Model::IntegrityError> error;

    /// Translates a Ref<T> obtained from Transaction::create() BEFORE this
    /// try_commit() call into its real, post-commit handle. A ref that was
    /// never local (read from base(), or returned by Transaction::update())
    /// passes through unchanged. Only meaningful when status == Committed --
    /// a local id from a Conflict/Vetoed/Invalid attempt was never installed
    /// anywhere; begin() a fresh Transaction and create() again instead.
    template <class T>
    Ref<T> resolve(Ref<T> local) const {
        if (!is_local(local.raw())) return local;
        auto it = local_remap.find(local.raw().index);
        return Ref<T>(it == local_remap.end() ? Id{} : it->second);
    }
    /// Opt<T> form. Null passes through as null.
    template <class T>
    Opt<T> resolve(Opt<T> local) const {
        if (!local || !is_local(local.raw())) return local;
        auto it = local_remap.find(local.raw().index);
        return Opt<T>(it == local_remap.end() ? Id{} : it->second);
    }
};

// ---------------------------------------------------------------------------
// Transaction -- the write side's private, per-caller working set
// ---------------------------------------------------------------------------

/// A transaction's local view: everything you create/update/remove is visible
/// only here until Model::try_commit() succeeds. Building one touches no
/// shared state at all and takes no lock -- see Model::begin(). Not copyable
/// (owns unique_ptrs to not-yet-installed objects); movable.
///
/// Dropping a Transaction without committing it is a complete, silent
/// rollback: since nothing shared was ever touched, there is nothing to undo.
class Transaction {
public:
    // Movable so it can be returned from begin() and handed between owners;
    // the moved-from shell is inert (null base, empty overlay). NOT copyable:
    // it owns unique_ptrs to not-yet-installed objects, and two copies
    // racing try_commit() with the same local ids would be incoherent.
    Transaction(Transaction&&) = default;
    Transaction& operator=(Transaction&&) = default;
    Transaction(const Transaction&) = delete;
    Transaction& operator=(const Transaction&) = delete;

    /// The pinned Snapshot this transaction reads through (peek/update clone
    /// from here). Also usable directly, e.g. to look at pre-transaction
    /// state -- it's an ordinary Snapshot.
    const Snapshot& base() const noexcept { return base_; }

    /// Shorthand for base().version(): what try_commit()'s conflict check
    /// compares the changelog against.
    std::uint64_t base_version() const noexcept { return base_.version(); }

    /// Takes ownership. Returns a LOCAL id, usable immediately as a Ref<T>/
    /// Opt<T> target for another object created (or updated) in this SAME
    /// transaction. Becomes a real id only once Model::try_commit() succeeds;
    /// referencing it from a DIFFERENT transaction, or after this one is
    /// dropped, is a use-after-scope bug the type system cannot catch --
    /// don't let a local Ref<T> escape its Transaction. Referential integrity
    /// against already-committed state is checked at apply time
    /// (try_commit()), not here -- building a transaction never touches
    /// shared state, so there is nothing yet to check it against.
    template <class T>
    Ref<T> create(std::unique_ptr<T> o) {
        static_assert(std::is_base_of_v<ObjectBase, T>, "T must derive from Object<T>");
        const Id local_id{kLocalIdBit | next_local_id_++, 1};
        o->id = local_id;
        local_created_.push_back(std::move(o));
        pending_changes_.push_back({local_id, ChangeKind::Created, type_tag<T>()});
        return Ref<T>(local_id);
    }

    /// Copy-on-write handle, scoped to this transaction: clones from base()
    /// (or returns the same clone again, if this id was already touched this
    /// transaction) and returns a mutable pointer, fully local until commit.
    /// Unlike the single-writer design's update(), there is no deferred
    /// reconciliation step to worry about -- write whatever you like, in any
    /// order, any number of times; try_commit() sees only the final value.
    /// Null if the target doesn't exist as of this transaction's base(), or
    /// (if it's a local id) was removed earlier in this same transaction.
    template <class T>
    T* update(Ref<T> r) {
        return static_cast<T*>(update_impl(r.raw()));
    }
    template <class T>
    T* update(Opt<T> r) {
        return static_cast<T*>(update_impl(r.raw()));
    }

    /// Records an INTENT to delete -- unlike the single-writer design's
    /// remove(), this does NOT resolve cascade fan-out now (that can only be
    /// computed correctly against the single, up-to-date, authoritative
    /// reverse index, which is exactly what try_commit()'s serialized apply
    /// phase has and a Transaction's private overlay does not). The full
    /// resolved kill list -- including everything cascade-deleted -- is only
    /// available afterward, in CommitResult::changes.
    ///
    /// A consequence worth knowing: within THIS transaction, objects that
    /// would cascade-die from this intent still peek()/exists() as alive
    /// until commit -- cascade effects are invisible locally, by design (the
    /// alternative is a duplicate local reverse index, real complexity for a
    /// narrow benefit). If `r` is a local id (created earlier in this same
    /// transaction, never committed), the create is cancelled outright
    /// instead: there's no real id yet for anything else to reference, so
    /// there's nothing to resolve at commit time either.
    template <class T>
    void remove(Ref<T> r) {
        remove_impl(r.raw());
    }
    template <class T>
    void remove(Opt<T> r) {
        remove_impl(r.raw());
    }

    /// "Is `r` alive as far as THIS transaction can tell?" -- peek() != null,
    /// so it honors local creates, edits, and remove() intents, but NOT other
    /// transactions' uncommitted work, and not cascade fan-out from this
    /// transaction's own intents (invisible until commit; see remove()).
    template <class T>
    bool exists(Ref<T> r) const {
        return peek(r) != nullptr;
    }
    template <class T>
    bool exists(Opt<T> r) const {
        return peek(r) != nullptr;
    }

    /// Read this transaction's current local view of `r`: this transaction's
    /// own pending edit if there is one, else base()'s committed value, else
    /// null. Masked (returns null) if `r`'s id has a pending remove() intent,
    /// even though the object technically still exists in base().
    template <class T>
    const T* peek(Ref<T> r) const {
        return peek_impl<T>(r.raw());
    }
    template <class T>
    const T* peek(Opt<T> r) const {
        return peek_impl<T>(r.raw());
    }

    /// Safe typed read from an untyped Id -- e.g. a Change::id out of
    /// pending_changes(). Checks the object's actual type before casting
    /// (same as Snapshot::find<T>); null if dead/masked, or a type mismatch.
    template <class T>
    const T* peek_as(Id id) const {
        return peek_impl<T>(id);
    }

    /// The pre-transaction value for an id update()'d this transaction (the
    /// value as of base(), before any local edit). Only covers ids this
    /// transaction actually called update() on with a REAL (non-local) id --
    /// for a freshly create()'d object there is nothing to show as "before"
    /// (null is correct), and this does not attempt to cover remove()
    /// intents (read those from base() directly, or from a Snapshot taken
    /// before this transaction started).
    template <class T>
    const T* peek_before(Id id) const {
        auto it = update_baseline_.find(id.index);
        if (it == update_baseline_.end() || it->second->id != id ||
            it->second->tag() != type_tag<T>())
            return nullptr;
        return static_cast<const T*>(it->second);
    }

    /// This transaction's pending creates (local ids) and updates (real
    /// ids), in the order they happened. Does NOT include remove() intents --
    /// see remove_intents() -- since "Deleted" isn't accurate until the
    /// cascade is resolved at commit time.
    const std::vector<Change>& pending_changes() const { return pending_changes_; }

    /// The real ids this transaction intends to remove (and cascade-resolve)
    /// at commit time. Unordered -- the actual delete order (and everything
    /// it drags down) is decided by try_commit()'s cascade BFS, not by the
    /// order remove() was called.
    const std::unordered_set<Id, IdHash>& remove_intents() const { return remove_intents_; }

private:
    friend class Model;

    /// Only Model::begin() (and Snapshot::begin(), through it) constructs
    /// one -- a Transaction is meaningless without a Model to commit to and
    /// a pinned base to read through.
    Transaction(Model* m, Snapshot base) : model_(m), base_(std::move(base)) {}

    /// Untyped body of update()/update(): local id -> the already-owned
    /// local_created_ entry (create()-then-update() in the same txn, no new
    /// clone needed); real id already touched this txn -> the existing
    /// clone; otherwise clone base()'s value into local_updated_, remember
    /// the pre-edit baseline (peek_before()) and record the pending change.
    /// Null for a masked (remove()-intended) or nonexistent id. The clone
    /// happens AT MOST ONCE per id per transaction -- repeated update()
    /// calls on the same id return the SAME clone, so writes accumulate.
    ObjectBase* update_impl(Id id) {
        if (is_local(id)) {
            const std::uint32_t idx = id.index & ~kLocalIdBit;
            return idx < local_created_.size() ? local_created_[idx].get() : nullptr;
        }
        if (remove_intents_.count(id)) return nullptr;
        // local_updated_ is keyed by bare slot index (a transaction only ever
        // clones ONE generation of a given slot -- whichever was alive at
        // base()), so a lookup here must also check the clone's OWN id
        // matches the FULL id requested, generation included. Without this,
        // a caller asking about a stale generation of an already-updated
        // slot (e.g. a handle recycled since base()) would get back the
        // WRONG object instead of a correct "not found".
        if (auto it = local_updated_.find(id.index); it != local_updated_.end())
            return (it->second->id == id) ? it->second.get() : nullptr;

        const ObjectBase* base_obj = base_.find_raw(id);
        if (!base_obj) return nullptr;
        auto clone = std::unique_ptr<ObjectBase>(base_obj->clone());
        update_baseline_.try_emplace(id.index, base_obj);
        ObjectBase* raw = clone.get();
        local_updated_.emplace(id.index, std::move(clone));
        pending_changes_.push_back({id, ChangeKind::Updated, raw->tag()});
        return raw;
    }

    /// Untyped body of remove()/remove(): for a LOCAL id, cancels the
    /// create() outright -- drops the owned object and scrubs it from
    /// pending_changes() -- since nothing has ever been published for it to
    /// reference. For a real id, just records the intent (remove_intents_);
    /// no cascade work happens here, see the class-level remove() doc for why.
    void remove_impl(Id id) {
        if (is_local(id)) {
            const std::uint32_t idx = id.index & ~kLocalIdBit;
            if (idx < local_created_.size()) local_created_[idx].reset();  // cancel locally
            auto rm = [&](const Change& c) { return c.id == id; };
            pending_changes_.erase(
                std::remove_if(pending_changes_.begin(), pending_changes_.end(), rm),
                pending_changes_.end());
            return;
        }
        remove_intents_.insert(id);
    }

    /// Untyped body of peek()/peek()/peek_as(): resolves `id` against this
    /// transaction's local overlay first (a local create, or an already-
    /// cloned update), falls through to base() only if neither applies, and
    /// is masked to null by a pending remove() intent regardless of what
    /// base() would say. The final tag check is what makes peek_as<T> safe
    /// to call on an untyped Change::id without a separate cast.
    template <class T>
    const T* peek_impl(Id id) const {
        const ObjectBase* o = nullptr;
        if (is_local(id)) {
            const std::uint32_t idx = id.index & ~kLocalIdBit;
            o = idx < local_created_.size() ? local_created_[idx].get() : nullptr;
        } else if (remove_intents_.count(id)) {
            return nullptr;
        } else if (auto it = local_updated_.find(id.index); it != local_updated_.end()) {
            // Same generation check as update_impl -- see its comment. A
            // stale generation of an already-updated slot is dead, not the
            // (different-generation) object this txn happens to have cloned.
            o = (it->second->id == id) ? it->second.get() : nullptr;
        } else {
            o = base_.find_raw(id);
        }
        return (o && o->tag() == type_tag<T>()) ? static_cast<const T*>(o) : nullptr;
    }

    Model* model_ = nullptr;  ///< where try_commit() applies this; asserted against
                              ///< cross-model misuse in try_commit()
    Snapshot base_;  ///< pins base_version_ in Model::live_, same as any reader's snapshot

    std::vector<std::unique_ptr<ObjectBase>>
        local_created_;  ///< index == local id's low bits;
                         ///< null entry == cancelled (see remove_impl)
    std::unordered_map<std::uint32_t, std::unique_ptr<ObjectBase>>
        local_updated_;  ///< keyed by real slot index
    std::unordered_map<std::uint32_t, const ObjectBase*>
        update_baseline_;                            ///< points into base_'s Root,
                                                     ///< kept alive by base_ itself
    std::unordered_set<Id, IdHash> remove_intents_;  ///< real ids only -- see remove_impl
    std::uint32_t next_local_id_ = 0;      ///< mints local ids; per-TRANSACTION, so the same
                                           ///< value recurs across transactions (see create())
    std::vector<Change> pending_changes_;  ///< what pending_changes() returns, in call order
};

// ---------------------------------------------------------------------------
// Hot path -- keep inlineable
// ---------------------------------------------------------------------------

inline const ObjectBase* Snapshot::find_raw(Id id) const noexcept {
    if (!id || !root_) return nullptr;
    const std::uint32_t c = id.index >> kChunkBits;
    const std::uint32_t i = id.index & kChunkMask;
    if (c >= root_->spine.size()) return nullptr;
    const Chunk& ch = *root_->spine[c];
    if (ch.gen[i] != id.gen) return nullptr;  // slot was recycled
    return ch.obj[i];
}

// ---------------------------------------------------------------------------
// View -- an object paired with the snapshot it came from
// ---------------------------------------------------------------------------

/// An object and the snapshot it was read from, travelling together.
///
///     auto ord = *s.view_by_key<&Order::computed_key>("ord:O1");
///     ord->qty;                                  // fields, as usual
///     const Account& a = *ord[&Order::account];  // Ref<>  -> View<Account>, never null
///     if (auto dad = ord[&Order::parent])        // Opt<>  -> optional<View<Order>>
///         dad->code;
///
/// Traversal always uses the view's OWN snapshot, so mixing versions is not
/// expressible. Compare the raw API, where `s2.resolve(orderFromS1->account)`
/// compiles and is undefined.
///
/// SCOPED, NOT STORED. A View holds the snapshot by pointer, not by value:
///   - by value would mean two atomic refcount bumps per traversal step, on the
///     hottest path in the system (see the benchmark in examples/bench.cpp);
///   - and a stored view would pin its version alive, stalling the reaper --
///     which is the fat-ref failure mode this design exists to avoid.
/// So a View must not outlive the Snapshot it came from. Keep one on the stack,
/// pass it down, drop it. Do not put one in a member or a container.
template <class T>
class View {
public:
    /// Pairs an object with the snapshot it came from -- unchecked (`obj`
    /// must actually live in `s`; nothing here verifies that, see
    /// Snapshot::view(const T&)) and by REFERENCE, not copy: this is the one
    /// place a View's two pointers get set, and it takes an lvalue Snapshot
    /// specifically so a caller can't accidentally bind to a temporary (see
    /// the deleted overload below) and leave s_ dangling immediately.
    View(const Snapshot& s, const T& obj) noexcept : s_(&s), o_(&obj) {}
    View(Snapshot&&, const T&) = delete;  // never bind to a temporary snapshot

    // Plain field access -- a view reads like a pointer to the object...
    const T& operator*() const noexcept { return *o_; }
    const T* operator->() const noexcept { return o_; }
    // ...while operator[] below is the traversal step that stays on this
    // view's own snapshot.

    /// Follow a non-nullable field. Always yields a view.
    template <class U>
    View<U> operator[](Ref<U> T::* field) const noexcept {
        return View<U>(*s_, s_->resolve(o_->*field));
    }

    /// Follow a nullable field. Empty if the target was cascaded away.
    template <class U>
    std::optional<View<U>> operator[](Opt<U> T::* field) const noexcept {
        if (const U* p = s_->resolve(o_->*field)) return View<U>(*s_, *p);
        return std::nullopt;
    }

    /// The snapshot this view reads through -- to drop back to the raw API
    /// (find_by_key, for_each, ...) mid-traversal without re-plumbing which
    /// version you were on.
    const Snapshot& snapshot() const noexcept { return *s_; }

    /// "Who references this?" -- every object whose `Field` points at this
    /// one. Slow scan; see Snapshot::for_each_referrer. Named the same way:
    /// `acct.for_each_referrer<&Order::account>(f)`.
    template <auto Field, class F>
    void for_each_referrer(F&& f) const {
        s_->for_each_referrer_view<Field>(Ref<T>(o_->id), std::forward<F>(f));
    }

    /// Same scan as for_each_referrer, collected into a vector of views.
    template <auto Field>
    std::vector<View<member_class_t<decltype(Field)>>> find_referrers() const {
        return s_->find_referrers_view<Field>(Ref<T>(o_->id));
    }

private:
    const Snapshot* s_;  ///< by POINTER, never by value -- see the class comment
    const T* o_;         ///< owned by the Model, alive as long as *s_ is
};

template <class T>
View<T> Snapshot::view(const T& obj) const noexcept {
    return View<T>(*this, obj);
}

template <auto Field>
std::optional<View<member_class_t<decltype(Field)>>> Snapshot::view_by_key(
    const member_value_t<decltype(Field)>& value) const {
    using ClassT = member_class_t<decltype(Field)>;
    if (const ClassT* p = find_by_key<Field>(value)) return View<ClassT>(*this, *p);
    return std::nullopt;
}

template <auto Field>
std::vector<View<member_class_t<decltype(Field)>>> Snapshot::view_by_scan_field(
    const member_value_t<decltype(Field)>& value) const {
    using ClassT = member_class_t<decltype(Field)>;
    std::vector<View<ClassT>> out;
    for (const ClassT* p : find_by_scan_field<Field>(value)) out.push_back(View<ClassT>(*this, *p));
    return out;
}

template <auto Field>
std::vector<View<member_class_t<decltype(Field)>>> Snapshot::view_by_cached_field(
    const member_value_t<decltype(Field)>& value) const {
    using ClassT = member_class_t<decltype(Field)>;
    std::vector<View<ClassT>> out;
    for (const ClassT* p : find_by_cached_field<Field>(value)) out.push_back(View<ClassT>(*this, *p));
    return out;
}

template <class T>
std::optional<View<T>> Snapshot::view(Ref<T> r) const {
    if (const T* p = find(r)) return View<T>(*this, *p);
    return std::nullopt;
}

template <class T, class F>
void Snapshot::for_each_view(F&& f) const {
    for_each<T>([&](const T& o) { f(View<T>(*this, o)); });
}

template <class T, class Pred>
std::vector<View<T>> Snapshot::find_all_view(Pred&& pred) const {
    std::vector<View<T>> out;
    for_each_view<T>([&](View<T> v) {
        if (pred(*v)) out.push_back(v);
    });
    return out;
}

template <auto Field, class F>
void Snapshot::for_each_referrer_view(
    Ref<typename member_value_t<decltype(Field)>::target_type> target, F&& f) const {
    using ClassT = member_class_t<decltype(Field)>;
    for_each_view<ClassT>([&](View<ClassT> v) {
        if (((*v).*Field).raw() == target.raw()) f(v);
    });
}

template <auto Field>
std::vector<View<member_class_t<decltype(Field)>>> Snapshot::find_referrers_view(
    Ref<typename member_value_t<decltype(Field)>::target_type> target) const {
    using ClassT = member_class_t<decltype(Field)>;
    std::vector<View<ClassT>> out;
    for_each_referrer_view<Field>(target, [&](View<ClassT> v) { out.push_back(v); });
    return out;
}

template <auto Field>
std::vector<View<member_class_t<decltype(Field)>>> Snapshot::view_cached_referrers(
    Ref<typename member_value_t<decltype(Field)>::target_type> target) const {
    using ClassT = member_class_t<decltype(Field)>;
    std::vector<View<ClassT>> out;
    for (const ClassT* p : find_cached_referrers<Field>(target)) out.push_back(View<ClassT>(*this, *p));
    return out;
}

}  // namespace model
