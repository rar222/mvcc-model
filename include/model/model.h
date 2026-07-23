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
//
// Comment style used throughout this file:
//   ///        doc comment for whatever declaration follows it
//   ///<       trailing doc comment for the declaration it follows, same line
//   ///< ^     continuation of a ///< comment, wrapped onto the next line(s)
//              because it didn't fit trailing the declaration -- the ^ points
//              back up at the member it documents, so it isn't mistaken for
//              an ordinary /// comment documenting whatever comes AFTER it
//   //         plain comment: section dividers, or prose that isn't
//              Doxygen-extractable documentation of one specific declaration

#include <algorithm>
#include <any>
#include <atomic>
#include <cassert>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <functional>
#include <limits>
#include <list>
#include <map>
#include <memory>
#include <memory_resource>
#include <mutex>
#include <optional>
#include <shared_mutex>
#include <string>
#include <thread>
#include <type_traits>
#include <typeinfo>
#include <unordered_map>
#include <unordered_set>
#include <variant>
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

    /// True exactly when operator() above takes its unfolded branch: the
    /// "full 64-bit pack is a perfect hash of Id" claim in operator()'s own
    /// comment holds ONLY there -- the 32-bit-size_t fold is NOT
    /// collision-free. pmap::TrieCore (persistent_map.h) reads this opt-in
    /// static member -- via pmap::detail::hash_is_perfect_v, the same
    /// optional-static-member detection idiom this file's own
    /// has_define_references<> uses -- to decide whether a Leaf keyed by
    /// this Hash can skip collision-chain storage entirely. Declaring this
    /// true when it isn't would silently corrupt Root::by_type and every
    /// other IdHash-keyed persistent map on a 32-bit platform the moment two
    /// Ids' folded hashes coincided; that's why it's conditional on the
    /// exact same test operator() itself branches on, not a bare `true`.
    static constexpr bool is_perfect = sizeof(std::size_t) >= sizeof(std::uint64_t);
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
/// update/remove, RefRemapper, CommitResult::to_real -- branches on this.
inline bool is_local(Id id) noexcept {
    return (id.index & kLocalIdBit) != 0;
}

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
/// This is what lets a transaction build a graph of new, interlinked
/// objects in one go -- and because the whole remap table is minted BEFORE
/// the first create applies (see apply_transaction_contents' pre-mint pass,
/// and commit_bulk_without_undo(), which does the same), creates may reference each
/// other in ANY order: backward, forward, mutually-cyclic, or a field
/// pointing at the very object being created. Creation order carries no
/// meaning for reference resolution. A field that isn't local (already a
/// real id, or null) is left untouched.
///
/// Sets *unmapped (there are no exceptions in this project -- see CLAUDE.md)
/// if a local id has no entry in the table -- which means it pointed at a
/// local object that was never created (e.g. a stray local id from some
/// OTHER transaction), or one created-then-removed within the same
/// transaction (see Transaction::remove(), the create-then-uncreate case).
/// try_commit() then rejects the transaction as CommitStatus::Invalid:
/// a genuine transaction-building bug, not a concurrency conflict (there is
/// no real Id to check against the transaction's base, so it cannot be
/// classified as a Conflict -- see Model::validate's use of
/// IntegrityError::bad_target for the contrast). The field is left null; the
/// half-remapped object is never installed, so the value doesn't matter.
struct RefRemapper {
    const std::unordered_map<std::uint32_t, Id>& table;  // local Id::index -> real Id
    bool* unmapped;  // set on a local id with no mapping; the apply aborts

    /// Local id -> real id, by table lookup -- NOT a Snapshot::resolve()-style
    /// dereference (nothing here ever returns an object, only another Id), so
    /// this is named the same way CommitResult::to_real() explains its own
    /// naming: "resolve" is reserved elsewhere in this header for the
    /// dereferencing operation. See to_real()'s doc comment.
    Id translate(Id id) const {
        auto it = table.find(id.index);
        if (it == table.end()) {
            *unmapped = true;
            return Id{};
        }
        return it->second;
    }

    template <class T>
    void operator()(const void*, const char*, Ref<T>& r) const {
        if (is_local(r.raw())) r = Ref<T>(translate(r.raw()));
    }
    template <class T>
    void operator()(const void*, const char*, Opt<T>& r) const {
        if (r.raw() && is_local(r.raw())) r = Opt<T>(translate(r.raw()));
    }
};

/// The mirror image of RefRemapper: RefRemapper only ever rewrites LOCAL
/// ids (the pre-mint pass); this rewrites OLD REAL ids that are themselves
/// being recreated by THIS SAME undo application (see Model::apply_undo),
/// leaving every other id -- still alive and unaffected -- completely
/// untouched. Kept as its own type rather than generalizing RefRemapper,
/// matching this codebase's own stated preference (see commit_bulk_without_undo()'s
/// _no_log split) for separate, grep-able names over a flag threaded
/// through shared logic. No "unmapped" failure mode: an id not in `table`
/// simply passes through unchanged, which is correct here -- it names
/// something outside this undo entry, not a build-time bug to reject.
struct UndoRemapper {
    const std::unordered_map<Id, Id, IdHash>& table;  // old real id -> new local id

    /// Old real id -> new local id, by table lookup -- same "not a
    /// Snapshot::resolve()-style dereference" reasoning as RefRemapper::
    /// translate(), which this mirrors.
    Id translate(Id id) const {
        auto it = table.find(id);
        return it != table.end() ? it->second : id;
    }

    template <class T>
    void operator()(const void*, const char*, Ref<T>& r) const {
        r = Ref<T>(translate(r.raw()));
    }
    template <class T>
    void operator()(const void*, const char*, Opt<T>& r) const {
        if (r.raw()) r = Opt<T>(translate(r.raw()));
    }
};

namespace detail {
/// Global (process-wide, NOT per-Model) registry: field_tag<Field>() -> a
/// human name, for fields whose declaration opted in by passing one to
/// FieldKeyReader::key() or RefIndexReader::index() below. A property of
/// how the FIELD is declared, not of any particular Model or object
/// instance -- unlike Model's own (per-Model) field lookup stats, this is
/// intentionally shared across every Model in the process, the same scope
/// field_tag<Field>() itself already has.
///
/// Naming is opt-in and entirely optional: a field never given a name
/// still works everywhere (find_by_cached_field, the reverse index, ...) --
/// it just falls back to being identified by its declaring type and its
/// opaque address wherever something tries to display it for a human (see
/// Model::LookupDiagnostics::to_string()).
inline std::mutex g_field_name_mu;
inline std::unordered_map<const void*, std::string>& field_names() {
    static std::unordered_map<const void*, std::string> names;
    return names;
}

/// Registers `name` for `field`, the FIRST time this exact call site ever
/// runs (see register_field_name_once below) -- so the mutex here is taken
/// at most once per distinct Field, ever, for the life of the process, not
/// on every define_keys()/define_cached_fields()/define_cached_references()
/// traversal (those run on every create/update, which would otherwise make
/// this a real write-side cost). First name wins if ever called twice for
/// the same field with different strings (shouldn't happen -- a field's
/// name is fixed by its own declaration).
inline void register_field_name(const void* field, const char* name) {
    std::lock_guard lk(g_field_name_mu);
    field_names().try_emplace(field, name);
}

/// The name registered for `field`, or empty if none ever was.
inline std::string field_name_of(const void* field) {
    std::lock_guard lk(g_field_name_mu);
    auto it = field_names().find(field);
    return it != field_names().end() ? it->second : std::string();
}

/// One-time-per-Field registration, called from FieldKeyReader::key() and
/// RefIndexReader::index(). The `static` guard is the SAME "instantiate
/// once per template argument, then it's free" trick field_tag<Field>()
/// itself relies on (a function-local static's initialization is
/// thread-safe and happens exactly once) -- extended here to also run a
/// one-time side-effecting registration instead of merely returning an
/// address, so every call after the first for a given Field costs one
/// already-initialized-flag check, never the mutex above.
template <auto Field>
inline void register_field_name_once(const char* name) noexcept {
    if (!name) return;
    static const bool registered = (register_field_name(field_tag<Field>(), name), true);
    (void)registered;
}
}  // namespace detail

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
///
/// `name`, if supplied, is registered (once, see detail::
/// register_field_name_once) for display in Model::LookupDiagnostics::
/// to_string() -- purely cosmetic, like RefReader's own `name` parameter;
/// nothing correctness-critical reads it.
struct RefIndexReader {
    const RefIndexFn& fn;
    template <auto Field>
    void index(const char* name = nullptr) const {
        fn(field_tag<Field>());
        detail::register_field_name_once<Field>(name);
    }
};

/// Callback shape for enumerating a type's declared lookup fields (see
/// ObjectBase::each_field_key / each_cached_field / each_scan_field): the
/// field's identity tag plus its value in canonical string form
/// (to_field_key). One shape serves all three lookup families -- and, being
/// one shape rather than one per field type, is what lets a single
/// heterogeneous vector of (field, key) pairs represent a type's whole
/// define_keys() set (see collect_update_baseline_field_keys()) with no
/// variant.
using FieldKeyFn = std::function<void(const void* field, std::string key)>;

/// Canonical string form of a define_keys()-indexed field's value: a
/// std::string as-is, or an arithmetic type via std::to_string. Shared by
/// FieldKeyReader::key() (writing the index) and Snapshot::find_by_key()
/// (reading it) so the two can never drift apart on what a value maps to --
/// which is also what makes find_by_key type safe: it takes the field's own
/// value type, not a bare std::string, so a caller can't pass a key of the
/// wrong shape for that field in the first place. Same reason
/// collect_update_baseline_field_keys() stores baseline values in this form
/// rather than the field's native type: it lets old-vs-new be a plain ==
/// regardless of the field's declared type, and lets that baseline live in
/// the same string-keyed shape as by_field_ itself.
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
///
/// `name`, if supplied, is registered (once, see detail::
/// register_field_name_once) for display in Model::LookupDiagnostics::
/// to_string() -- purely cosmetic; nothing correctness-critical reads it.
/// One shape serves define_keys()/define_cached_fields()/
/// define_scan_fields() alike, so naming a field in any ONE of them (a
/// field reused across more than one, like Account::name, only needs it
/// once) names it for all.
struct FieldKeyReader {
    const FieldKeyFn& fn;
    template <auto Field, class V>
    void key(const V& v, const char* name = nullptr) const {
        fn(field_tag<Field>(), to_field_key(v));
        detail::register_field_name_once<Field>(name);
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
    /// via the copy constructor -- keep derived types copyable AND
    /// copy-assignable, see assign_from below). This is the copy-on-write
    /// primitive: Transaction::update() clones the committed object for
    /// local editing, and the cascade BFS clones a referrer before nulling
    /// one field, so published state is never mutated.
    virtual ObjectBase* clone() const = 0;

    /// Overwrite this object's own fields with `other`'s (Object<Derived>
    /// implements it via the copy-assignment operator -- the same
    /// copyability clone() already requires, just assigned instead of
    /// constructed). Used by Model::apply_undo to restore a RestoreUpdate
    /// action's captured pre-image onto a currently-live object generically,
    /// without knowing its concrete type. `other` must be the same concrete
    /// type as `this` -- same precondition tag()-checked casts everywhere
    /// else in this API already carry, just not re-asserted here.
    virtual void assign_from(const ObjectBase& other) = 0;

    /// Approximate in-memory footprint of this object's own value, for
    /// budgeting undo history by bytes rather than by entry count (see
    /// Model::set_max_undo_memory_bytes / UndoEntry::approx_bytes). Object<
    /// Derived> implements this as `sizeof(Derived)` -- exact for the
    /// object's fixed layout (numeric fields, Ref<>/Opt<>, embedded arrays),
    /// but it does NOT follow heap-owned members (a std::string past SSO, a
    /// std::vector's buffer): those are undercounted unless a concrete type
    /// overrides this to add e.g. `.capacity()` of its own owned buffers.
    /// This is a size estimate for a caller-configured memory cap, not an
    /// exact accounting -- treat it the same way.
    virtual std::size_t byte_size() const = 0;

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

    /// Same idea as remap_refs, via UndoRemapper instead of RefRemapper --
    /// a separate virtual because remap_refs() is hard-bound to that one
    /// concrete visitor type (unlike each_ref(), which takes a type-erased
    /// std::function). Only ever called by Model::apply_undo. See
    /// UndoRemapper's own comment for why this exists.
    virtual void remap_undo_refs(const UndoRemapper&) {}

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
/// A computed key CANNOT resolve a Ref<>/Opt<> field to reach another
/// object's data (e.g. an Order key incorporating its Account's name).
/// Two independent reasons, not one:
///
///   - Mechanically, there is nowhere to resolve THROUGH. computed_key() is
///     a bare nullary const method -- member_value<V (C::*)() const> -- and
///     Object<Derived>::each_field_key() calls it as `s.computed_key()`,
///     `s` being only the object itself. No Snapshot or Model is threaded
///     through define_keys()/FieldKeyReader::key()/each_field_key() at all.
///     A Ref<T>/Opt<T> is deliberately just an 8-byte {index, gen} with no
///     snapshot pointer (see Ref<T>'s own doc comment) -- resolving one
///     needs a Snapshot, which computed_key() has no way to obtain.
///
///   - Structurally, even threading one in would be unsound. define_keys()
///     only ever runs when the KEY-OWNING object itself is created or
///     updated (add_field_keys/reconcile_field_keys, called from
///     apply_create/apply_update) -- whatever string computed_key() returns
///     is baked into Root::by_field ONCE, at that instant, and never
///     recomputed later. If the key depended on a REFERENCED object's
///     field, updating that OTHER object without touching the referrer
///     would leave the referrer's indexed key silently stale: find_by_key
///     would miss it under its new value and still wrongly match it under
///     the old one. Nothing in this design tracks "key K depends on field F
///     of some other object, re-derive K when F changes" -- referrers_/
///     by_cached_reference solve a narrower, one-hop problem (who points at
///     X, for cascade delete/null AT X's OWN commit), not "recompute a
///     derived value elsewhere whenever any upstream field changes." That
///     is a materialized-view-with-invalidation problem, a different and
///     much bigger feature than "each type defines its own keys" -- and
///     the same class of before/after timing bug invariant 9 (CLAUDE.md)
///     already goes out of its way to avoid, reintroduced in a new,
///     self-inflicted shape.
///
/// What to do instead:
///   - Denormalize the field you need onto the referencing object at
///     create()/update() time (copy Account::name onto Order), so
///     computed_key() only ever reads `*this` -- the same pattern any
///     scan/cached field already uses.
///   - Or do a two-step lookup at QUERY time instead of baking the join
///     into the index: `snapshot.resolve(order.account).name`, then a
///     separate find_by_key/find on whatever that's actually looking for --
///     using Snapshot, which does have resolve access, just at read time
///     rather than baked into the write-time index.
///
/// Each define_keys() field is assumed unique within its type; a create or
/// update that would duplicate another live object's value is rejected as
/// CommitStatus::Invalid instead of silently overwriting it (see
/// Model::validate_field_key_uniqueness) -- EXCEPT commit_bulk_without_undo(),
/// which has no undo log to unwind a rejected batch against and keeps the
/// old silent-overwrite behavior; see its own doc comment. For "give me
/// every match, not just one," there are two MULTI-match families, declared
/// with the same visitor shape and named the same way -- each define_X
/// drives find_by_X and view_by_X, and in every family a field you did NOT
/// declare is invisible to its lookup (empty result, same as find_by_key on
/// a field define_keys() never mentioned):
///
///   define_keys()          -> find_by_key         / view_by_key
///       unique, indexed: O(log n); a duplicate value is rejected, not
///       overwritten.
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

    // sizeof(Derived), not sizeof(*this) -- *this is statically typed as
    // Object<Derived> at this point in the class body, and sizeof an
    // expression uses its STATIC type, which would give the wrong (base)
    // size. sizeof(Derived) is what clone() above actually allocates.
    std::size_t byte_size() const override { return sizeof(Derived); }

    void assign_from(const ObjectBase& other) override {
        static_cast<Derived&>(*this) = static_cast<const Derived&>(other);
    }

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

    void remap_undo_refs(const UndoRemapper& remapper) override {
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

    /// One persistent SET per type: every Id of that type. This is what makes
    /// for_each<T>() (and everything built on it: find_all, find_referrers,
    /// ...) O(#T objects) instead of a full spine scan. Keyed by TypeTag
    /// rather than a type-name string: the same address-comparison the rest
    /// of the model already uses for type checks (see type_tag<T>()), so no
    /// extra per-type registration is needed. One entry per distinct type
    /// (small, static-ish), so copying it per commit is O(#types), not
    /// O(#objects). A set, not a map: the Id is both the key and the only
    /// thing worth knowing about it here, so there is no value to store.
    std::unordered_map<TypeTag, pmap::PersistentSet<Id, IdHash>> by_type;

    /// One persistent map per define_keys()-declared field, keyed by the
    /// field's own field_tag<>() -- which already encodes both the type and
    /// the field, so there's no separate per-type grouping needed here the
    /// way by_type needs TypeTag. Empty for types that declare none. This is
    /// the unique lookup-by-value index (later write wins); there is no
    /// separate mandatory "primary key" index.
    std::unordered_map<const void*, pmap::PersistentMap<std::string, Id, pmap::StringHash>> by_field;

    /// One persistent MULTIMAP per define_cached_fields()-declared field:
    /// canonical value string -> a persistent set of every Id whose field
    /// currently holds that value. Unlike by_field, every match is kept. The
    /// bucket is a persistent SET, NEVER a flat vector: a flat bucket would
    /// make each mutation O(#duplicates of that value), which for a
    /// low-cardinality field (a status, a category) is O(n) per op -- the
    /// exact size-proportional cost this design exists to avoid.
    std::unordered_map<const void*,
                       pmap::PersistentMap<std::string, pmap::PersistentSet<Id, IdHash>,
                                          pmap::StringHash>>
        by_cached_field;

    /// One persistent MULTIMAP per define_cached_references()-declared
    /// Ref<>/Opt<> field: TARGET's Id -> a persistent set of every REFERRER's
    /// Id whose field currently points there. Same bucket discipline as
    /// by_cached_field (persistent set buckets, never flat vectors -- a hub
    /// object referenced by thousands would otherwise make every one of
    /// those referrers' commits O(#referrers)). This is the published,
    /// read-side counterpart of the writer-private referrers_
    /// (Model::referrers_): that index drives cascade delete and can never
    /// be handed to a reader (CLAUDE.md invariant 8), so a field worth fast
    /// reverse lookup needs this SEPARATE, opt-in structure.
    std::unordered_map<const void*,
                       pmap::PersistentMap<Id, pmap::PersistentSet<Id, IdHash>, IdHash>>
        by_cached_reference;
};

class Model;
class Transaction;
class BulkTransaction;
struct CommitResult;
enum class CommitStatus;

template <class T>
class View;

// ---------------------------------------------------------------------------
// Field lookup stats -- cached vs. uncached call counts, for the life of a Model
// ---------------------------------------------------------------------------

/// Call counts for one field's cached vs. uncached lookup family --
/// find_by_scan_field vs. find_by_cached_field, or for_each_referrer/
/// find_referrers vs. find_cached_referrers. Purely observational: lets a
/// caller decide, from ACTUAL usage over the model's whole lifetime, whether
/// define_cached_fields()/define_cached_references() is paying for itself
/// on a given field, or whether that field should be a scan field (or
/// nothing) instead. See Model::lookup_stats() and Model::diagnostics().
struct LookupCounts {
    std::uint64_t cached_calls = 0;    ///< find_by_cached_field / find_cached_referrers
    std::uint64_t uncached_calls = 0;  ///< find_by_scan_field / for_each_referrer / find_referrers
};

/// Identifies one looked-up field: its declaring type (captured as
/// `typeid(ClassT)` at the call site -- see Model::record_field_lookup) plus
/// the field itself (`field_tag<Field>()`). At namespace scope, not nested
/// in Model, because it's also the key type of Model::LookupDiagnostics::
/// stats -- a PUBLIC-facing shape -- and there's nothing Model-specific
/// about it: it's just "which field," the same identity field_tag<Field>()
/// itself already carries.
struct FieldLookupKey {
    const std::type_info* type;
    const void* field;
    friend bool operator==(FieldLookupKey, FieldLookupKey) noexcept = default;
};

/// Hasher for FieldLookupKey. Unlike IdHash, this makes no collision-free
/// claim and needs none: it only keys field_lookup_counts_, a diagnostics-
/// only map (see Model::record_field_lookup) that is never on a hot path
/// and never affects correctness -- an occasional bucket collision just
/// costs one extra comparison, not a wrong answer. A plain XOR-of-hashes
/// combination is enough for that; no `is_perfect` opt-in (see
/// pmap::detail::hash_is_perfect_v) is declared or needed here.
struct FieldLookupKeyHash {
    std::size_t operator()(FieldLookupKey k) const noexcept {
        return std::hash<const void*>{}(static_cast<const void*>(k.type)) ^
              (std::hash<const void*>{}(k.field) << 1);
    }
};

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
    /// Defined out-of-line (after Model) so its body can call
    /// Model::record_field_lookup() for LookupCounts -- see that method's
    /// doc comment for why this can't be inline here.
    template <auto Field>
    std::vector<const member_class_t<decltype(Field)>*> find_by_scan_field(
        const member_value_t<decltype(Field)>& value) const;

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
    /// Defined out-of-line (after Model) -- see find_by_scan_field's comment.
    template <auto Field>
    std::vector<const member_class_t<decltype(Field)>*> find_by_cached_field(
        const member_value_t<decltype(Field)>& value) const;

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
        it->second.for_each([&](Id id) {
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
    /// Defined out-of-line (after Model) -- see find_by_scan_field's comment.
    template <auto Field, class F>
    void for_each_referrer(Ref<typename member_value_t<decltype(Field)>::target_type> target,
                           F&& f) const;

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
    /// Defined out-of-line (after Model) -- see find_by_scan_field's comment.
    template <auto Field>
    std::vector<const member_class_t<decltype(Field)>*> find_cached_referrers(
        Ref<typename member_value_t<decltype(Field)>::target_type> target) const;

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
    /// `name`/`data` are optional, caller-supplied labels -- see
    /// Transaction::name()/data().
    Transaction begin(std::string name = "", std::any data = {}) const;

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

    /// Generic ("any type, any field") counterpart of for_each_referrer<Field>:
    /// walks EVERY live object of EVERY type via by_type, and every one of
    /// ITS ref fields via the virtual each_ref() (so no compile-time list of
    /// candidate (Type, Field) pairs is needed), calling
    /// f(referrer_id, field, nullable, referrer_tag) for every edge whose
    /// target is `target`. There is no index for this the way for_each_referrer
    /// has for one declared Field -- referrers_, the structure that WOULD make
    /// this cheap, is the writer's commit_mu_-protected reverse index and is
    /// unreachable from the read side (invariant 7) -- so this is O(total live
    /// objects), strictly for occasional/diagnostic use (see
    /// Transaction::estimate_changes_with_cascades(), its only caller), never a hot path.
    /// Private and friended to Transaction rather than public: exposing an
    /// O(#everything) scan as ordinary public API would invite exactly the
    /// misuse the single-Field scan family already warns against, multiplied
    /// by every type in the model.
    template <class F>
    void for_each_referrer_any(Id target, F&& f) const {
        if (!root_ || !target) return;
        for (const auto& [tag, ids] : root_->by_type) {
            ids.for_each([&](const Id& id) {
                const ObjectBase* o = find_raw(id);
                if (!o) return;
                o->each_ref([&](const void* field, const char*, Id ref_target, bool nullable) {
                    if (ref_target == target) f(o->id, field, nullable, tag);
                });
            });
        }
    }

    /// Forwards to the OWNING Model's Model::record_field_lookup(), if this
    /// Snapshot has an owner (a default-constructed Snapshot's lease_ is
    /// null, and never reaches any of the four call sites that use this --
    /// each already returns early on `!root_`, which is null together with
    /// lease_). Requires the same "must not outlive its Model" precondition
    /// every other Snapshot method already carries -- lease_ already holds
    /// this same Model* for release_version() (see Lease), so this adds no
    /// new lifetime requirement. Defined out-of-line: Model must be a
    /// complete type to call through lease_->m.
    void record_field_lookup(const std::type_info& type, const void* field, bool cached) const;

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
/// (they see the final value either way).
///
/// A given Id (generation included) USUALLY appears at most once per
/// changeset, but not always: a same-transaction create-then-cascade-delete
/// (a deferred local remove, see Transaction::remove_raw) or update()-then-
/// cascade-delete shows up as two separate entries -- e.g. Created then
/// Deleted -- one per event that actually happened during apply, even though
/// the net externally-visible effect is "never existed"/"back to how it
/// looked before this transaction." This is a deliberate audit trail (see
/// the cascade tests asserting both a Created and a Deleted entry for one
/// id), not something a subscriber needs to dedupe by hand -- just don't
/// assume a single lookup by Id tells you everything this changeset did to
/// it. Model::UndoEntry::actions is NOT built this way: it collapses to at
/// most one action per id (see Model::collapse_undo_actions()), since a
/// naive replay of every action would resurrect something that was never
/// visible or restore the wrong intermediate value.
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
/// reap_backlog(), exhausted_slots(), or set_pre_commit() from the hook
/// (self-deadlock on the non-recursive commit_mu_); snapshot() is fine, and
/// yields the still-current PRE-commit version, since the transaction being
/// inspected has not published yet.
///
/// The Transaction parameter is the SAME `txn` passed to try_commit() --
/// its id() is the correlation key shared with PreTransactionsFn and
/// PostCommitFn for this same attempt (see Transaction::id()). By this
/// point, apply has already MOVED `txn`'s local_created_/local_updated_
/// contents out (same "spent" state a non-Committed CommitResult leaves
/// behind) -- don't call create()/update()/remove() on it, and don't expect
/// to see the original objects through it. Read-only accessors remain
/// valid and meaningful, though: id(), base_version(), remove_intents(),
/// and pending_changes() (a separate log, untouched by apply -- still shows
/// the transaction's own local-id view of what it asked for, distinct from
/// this hook's `changes_` parameter, which is the fully-resolved,
/// real-id, cascade-included view).
///
/// Unchanged by, and unaware of, PreTransactionsFn below: if a
/// pre-transactions phase ran first, this hook still runs at exactly the
/// same point (after the MAIN transaction's own apply) seeing exactly the
/// same thing (the main transaction's own resolved changeset) -- it has no
/// way to tell whether pre-transactions ran, and doesn't need to.
using PreCommitFn = std::function<bool(Model&, const Transaction&, const std::vector<Change>&)>;

/// Registered once via Model::set_pre_transactions, called on every
/// try_commit() attempt (if installed) IMMEDIATELY after commit_mu_ is
/// acquired -- before the main Transaction is touched in ANY way: not
/// conflict-checked, not applied. This is the ONLY point at which
/// Model::run_pre_transaction() may be called; use it to run zero or
/// more OTHER Transactions first, atomically with respect to every other
/// writer (commit_mu_ never releases in between) and atomically with
/// respect to the main transaction (which hasn't started yet).
///
/// Each call to run_pre_transaction() is a REAL, independent commit
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
///
/// The Transaction parameter is the MAIN transaction this try_commit()
/// attempt is about to process -- completely untouched at this point (not
/// conflict-checked, not applied), so every accessor is fully populated:
/// id() (the correlation key shared with PreCommitFn/PostCommitFn for this
/// same attempt, see Transaction::id()), base_version(), pending_changes(),
/// remove_intents(). A hook can use these to decide WHICH pre-transactions
/// to run, not just to log an id. It must not call any of txn's mutating
/// methods (create()/update()/remove()) -- inspect it, don't build on it;
/// run_pre_transaction() only ever applies a SEPARATE Transaction
/// built via model.begin() inside the hook, never this one.
using PreTransactionsFn = std::function<void(Model&, const Transaction&)>;

/// Registered once via Model::set_post_commit, called once per non-empty
/// try_commit() attempt (if installed) -- but, unlike PreCommitFn and
/// PreTransactionsFn, strictly AFTER commit_mu_ has been released, with the
/// attempt's own final CommitResult (whatever it was -- Committed, Conflict,
/// Vetoed, Invalid, or PrecommitConflict; check `status` first, same as any
/// other CommitResult).
///
/// Because the lock is already gone by the time this runs, this is the one
/// hook it is SAFE to call try_commit()/begin()/snapshot()/current_version()
/// from -- there is nothing left to self-deadlock against (contrast with
/// PreCommitFn and PreTransactionsFn, which run WHILE commit_mu_ is held and
/// must never do any of that). A hook that always chains another commit is
/// still the caller's own infinite-recursion risk to manage, same as any
/// unconditional retry loop -- the framework doesn't need to guard against
/// it, since it's no different from calling try_commit() in a `while(true)`
/// by hand.
///
/// The trade-off for running unlocked: other commits (from other threads, or
/// a chained one this hook itself starts) may have ALREADY landed by the
/// time this runs -- the CommitResult it receives describes the state of the
/// world as of the moment this attempt finished, not as of "now." If you
/// need to observe or act on state atomically with respect to this
/// transaction, do it inside PreCommitFn (still locked) instead; use
/// PostCommitFn for side effects that don't need that atomicity -- logging,
/// metrics, waking up other work, or kicking off a follow-up transaction.
///
/// The Transaction parameter is the same `txn` try_commit() was called
/// with -- its id() is the same correlation key PreTransactionsFn/
/// PreCommitFn saw for this attempt (see Transaction::id()). Like
/// PreCommitFn's, it is "spent" by now (local overlay moved from during
/// apply, whether or not the attempt actually published) -- id(),
/// base_version(), and pending_changes() remain valid; don't call its
/// mutating methods.
using PostCommitFn = std::function<void(Model&, const Transaction&, const CommitResult&)>;

/// One delivery to a subscriber: a consistent state plus what changed since
/// the previous delivery -- read the changed objects out of THIS update's
/// own snapshot, never a fresh Model::snapshot() (which may already be
/// newer, and would tear the "state matches changes" pairing).
struct Update {
    Snapshot snapshot;  ///< state as of these changes; pins its version until dropped
    /// Every id that changed since the last delivery. Shared (not owned
    /// per-Update): publish_now() hands the SAME changeset to every
    /// subscriber, so this is a shared_ptr<const ...> rather than a
    /// std::vector<Change> -- one commit with N subscribers builds the
    /// changeset once instead of copying it N times. Read-only by
    /// construction (const), so sharing is safe even though each
    /// subscriber's queue is drained on its own thread.
    std::shared_ptr<const std::vector<Change>> changes;
    bool coalesced = false;  ///< true if intermediate versions were folded away
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
    /// `name`/`data` are optional, caller-supplied labels -- see
    /// Transaction::name()/data() -- copied onto this commit's UndoEntry/
    /// UndoSummary (if any) once it publishes.
    Transaction begin(std::string name = "", std::any data = {});

    /// Start a transaction based on an explicit, already-held Snapshot (e.g.
    /// one you're also using for analysis, so you want the transaction's view
    /// to match exactly).
    Transaction begin(Snapshot base, std::string name = "", std::any data = {});

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

    /// Identical to try_commit() in every observable way EXCEPT one: this
    /// commit is never added to the undo list (see UndoEntry/undo_list_).
    /// Not just "captured, then immediately discarded" -- the per-object
    /// pre-image capture (a clone() per changed/removed object; see
    /// pending_undo_'s own comment) never happens at all, so this is the
    /// call to reach for when that extra clone cost matters and you know
    /// you'll never want to undo this particular commit. Existing
    /// undo_list_ entries are still pruned as usual if this commit
    /// conflicts with them (see publish_now()) -- only the ADDITION of a
    /// new entry for THIS commit is skipped.
    CommitResult try_commit_without_undo(Transaction& txn);

    // ---- bulk load (EXCLUSIVE ACCESS ONLY -- read this before using) -------
    //
    // try_commit()'s per-object undo log -- log() every mutation's exact
    // inverse, so a conflicted/invalid/vetoed attempt can unwind cleanly --
    // is what makes multi-writer OCC safe. It is also, unavoidably, an
    // O(items in the transaction) memory cost: every create logs its own
    // slot-write inverse (set_slot), its own slot-allocation inverse
    // (alloc_slot), one inverse per outgoing ref edge (add_out_refs), and a
    // cheap per-id UndoAction (pending_undo_) -- none of it released until
    // the WHOLE transaction resolves. (The four WHOLE-MAP-HANDLE indexes --
    // by_type_/by_field_/by_cached_field_/by_cached_reference_ -- log their
    // pre-attempt handle only ONCE per attempt per key touched, not once per
    // object touching that key; see log_by_type_once()'s doc comment. That
    // used to dominate this cost -- N objects of one type meant N separate
    // captures of the SAME by_type_ handle -- which is why this used to be
    // measured far higher than it is now.) Put 200,000 creates in one
    // Transaction and you still retain 200,000 sets of the remaining
    // per-object closures simultaneously -- measured at ~1.8x the
    // steady-state per-object cost of a bulk load (default build,
    // n=200,000; see performance_tests.cpp's
    // bulk_load_avoids_the_single_transaction_undo_log_memory_blowup for the
    // up-to-date number). That is not a bug in try_commit(); it is the price
    // of a guarantee (clean rollback) that a genuine bulk load does not
    // need, because a bulk load either replaces the ENTIRE model or doesn't
    // run at all -- there is no partial-failure state worth rolling back TO.
    //
    // begin_bulk()/commit_bulk_without_undo() trade that guarantee away, deliberately
    // and only here, for exactly that case: wipe the whole Model and load a
    // fresh graph in one shot, at close to the steady-state per-object
    // cost, with no per-object undo logging at all.
    //
    // THE PRECONDITION, and why it is load-bearing rather than advisory:
    // commit_bulk_without_undo() requires that NO OTHER THREAD is doing ANYTHING with
    // this Model for the ENTIRE begin_bulk()..commit_bulk_without_undo() window --
    // holding a Snapshot, holding a Transaction, calling snapshot() or
    // subscribe(), or having an undrained Subscription queue (a queued
    // Update pins a Snapshot too). This is checked -- commit_bulk_without_undo() asserts
    // live_ is empty, both before it starts wiping and again immediately
    // before it publishes -- but the assert is a tripwire, not a lock:
    // commit_mu_ is held throughout for internal consistency with every
    // other commit_mu_-protected member, but snapshot()/begin() deliberately
    // never take commit_mu_ (invariant 10, and what keeps them
    // contention-free against try_commit() in the first place), so nothing
    // here can BLOCK a concurrent snapshot() call the way a lock would.
    // If one slips in between the precondition check and the actual wipe,
    // that thread's Snapshot points at a Root whose Chunks this call is
    // actively deleting objects out from under -- a real, silent
    // use-after-free, not a logical inconsistency the model can detect
    // after the fact. This is the deliberate cost of "no locks, no log, no
    // other overhead": there is no cheaper way to get that memory profile
    // that still lets other threads touch the Model at the same time. Use
    // this only at startup, before any Snapshot/Subscription has ever been
    // handed out, or during a maintenance window where every other thread
    // has already been quiesced.
    //
    // Scope, deliberately narrow: create() only (no update/remove/peek --
    // there is nothing meaningful to update/remove/peek before the wipe
    // installs anything), and the wipe is unconditional (this always
    // replaces the ENTIRE model; it is not a bulk-insert-into-existing-data
    // tool). PreCommitFn/PreTransactionsFn/PostCommitFn do NOT fire for a
    // bulk commit -- they exist to gate/observe ordinary deltas, and a
    // wholesale replacement isn't one.

    /// Start building a bulk load. Touches nothing yet -- exactly like
    /// begin(), building is free of shared state -- so nothing has
    /// happened to the Model just because this was called. See the section
    /// comment above for the full contract; this is only the builder half.
    BulkTransaction begin_bulk();

    /// Wipes the Model and installs everything created on `txn` as the
    /// model's entire new content, in one atomic publish -- or, if any
    /// object's Ref<>/Opt<> fails to resolve WITHIN this batch (the model
    /// is empty afterward, so there is nothing else for a ref to resolve
    /// against), rejects as CommitStatus::Invalid with NOTHING touched:
    /// the whole batch is validated before any mutation begins, since
    /// there is no undo log to unwind a partial failure with. Never
    /// returns Conflict (nothing else can be racing this, by the
    /// exclusive-access precondition) or Vetoed (no hook runs). See the
    /// section comment above for the precondition this REQUIRES -- calling
    /// this while any other thread holds a Snapshot of this Model is
    /// undefined behavior, not a checked error.
    ///
    /// UNLIKE an ordinary Transaction (see define_keys()'s own doc comment
    /// and Model::validate_field_key_uniqueness), a define_keys() value
    /// duplicated within this batch is NOT rejected here -- only Ref<>/Opt<>
    /// integrity is validated upfront. Extending that check to this path
    /// would mean a second whole-batch validation pass, the same idea as
    /// the ref-integrity one above; not done today, so a batch that
    /// violates define_keys()'s uniqueness contract still silently
    /// overwrites the earlier entry (see add_field_keys_no_log()'s own
    /// comment in the .cpp).
    CommitResult commit_bulk_without_undo(BulkTransaction& txn);

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

    /// Install (or clear, with {}) the post-commit hook. See PostCommitFn.
    /// Same swap-under-commit_mu_ contract as set_pre_commit/
    /// set_pre_transactions -- but the hook it installs runs UNLOCKED; see
    /// PostCommitFn for what that changes.
    void set_post_commit(PostCommitFn fn) {
        std::lock_guard lk(commit_mu_);
        post_commit_ = std::move(fn);
    }
    /// The writer's view of the LATEST state (generation-checked, like
    /// Snapshot::find_raw but against spine_, which may be mid-commit).
    /// This is what validate() checks against -- "re-validated against
    /// latest, not just base" falls out of using peek() here.
    /// Untyped read of the model's CURRENT writer-side state by Id -- the
    /// same "latest, not just base" view validate() itself reads (this is
    /// the ONLY public forwarder to the private peek(Id); peek_as<T> below
    /// is built on top of it, not a second independent caller of peek()).
    /// Callable ONLY from inside a running PreCommitFn or PreTransactionsFn,
    /// which run WHILE the calling thread already holds commit_mu_
    /// (invariant 7): that lock is what makes reading spine_ here safe, and
    /// it is NOT recursive, so this must never itself try to acquire it.
    /// Calling this from anywhere else is a data race on spine_ that only
    /// TSan is likely to catch -- there is no assert guarding it, the same
    /// trust-the-caller convention peek(Id) itself already relies on.
    ///
    /// This is what makes a truly type-agnostic hook possible across a
    /// model with many object types: ObjectBase::each_ref() is a public
    /// virtual method every Object<Derived> overrides to dispatch through
    /// that type's own define_references() (see Object<Derived>::each_ref,
    /// model.h ~line 727) -- so `peek_raw(id)->each_ref(fn)` walks every
    /// outgoing Ref<>/Opt<> field of WHATEVER type `id` happens to be, with
    /// no per-type dispatch table in the caller at all.
    const ObjectBase* peek_raw(Id id) const;

    /// Typed, checked convenience over peek_raw() -- same contract, same
    /// safety reasoning, just a checked downcast for a caller that already
    /// knows (or wants to assert) the concrete type. This is what lets a
    /// pre-commit hook do more than see WHICH ids changed (Change::id/
    /// kind/tag): a Created or Updated object's actual field values are
    /// otherwise unreachable from inside the hook, since by this point
    /// apply has already moved the contents out of the Transaction's own
    /// local_created_/local_updated_ overlay (see PreCommitFn's doc
    /// comment) -- Transaction::peek_as<T> reads exactly that now-emptied
    /// overlay, so it cannot help here.
    template <class T>
    const T* peek_as(Id id) const {
        const ObjectBase* o = peek_raw(id);
        return (o && o->tag() == type_tag<T>()) ? static_cast<const T*>(o) : nullptr;
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
    ///
    /// Why the in_pre_transactions_phase_ check is an assert (crash) rather
    /// than a checked failure (an error CommitResult, or a no-op): this
    /// function does no locking of its own -- it goes straight into
    /// commit_pretransaction_locked(), which requires commit_mu_ ALREADY held by the
    /// caller (see its own doc comment). The assert is the only thing
    /// standing between "called during the one window where try_commit()
    /// guarantees the lock is held" and "called from anywhere else" --
    /// e.g. a stray call from ordinary application code holding no lock at
    /// all, racing every other thread currently inside try_commit(). That
    /// is not a logic error with a graceful fallback; it is invariant 7's
    /// commit_mu_-protected state (spine_, referrers_, by_type_, ...) being
    /// mutated with no synchronization whatsoever -- corruption or a
    /// use-after-free, the exact failure mode this whole design exists to
    /// rule out, and one a returned error status could not prevent (the
    /// corrupting writes already happened by the time there's a status to
    /// check). Failing loudly and immediately, every single time (asserts
    /// are never compiled out in this project -- no preset defines
    /// NDEBUG), turns a would-be race into a guaranteed, obvious crash at
    /// the call site instead of a silent, load-dependent one discovered
    /// later. See CLAUDE.md's "prefer failing loudly."
    CommitResult run_pre_transaction(Transaction& txn);

    /// Identical to run_pre_transaction() in every observable way EXCEPT
    /// one: this pre-transaction's commit is never added to the undo list
    /// (see Model::try_commit_without_undo()'s own doc comment -- same
    /// reasoning, same "skip the clone() entirely, not just discard it"
    /// behavior, applied to a pre-transaction instead of the main one).
    /// Existing undo_list_ entries are still pruned as usual if this
    /// pre-transaction's commit conflicts with them.
    CommitResult run_pre_transaction_without_undo(Transaction& txn);

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
    /// Serviced by the background reaper, so this counts its backlog --
    /// exactly the same quantity Diagnostics::reap_backlog reports, just
    /// without diagnostics()'s cost of also copying spine/index sizes and
    /// the live-snapshot/subscriber sections. retired_ is commit_mu_-
    /// protected (unlike the single-writer sibling, where it was safe to
    /// read unsynchronized from the one writer thread that owned it -- here
    /// any thread may call this while another holds commit_mu_, so the read
    /// needs the same lock apply() does).
    std::size_t reap_backlog() const {
        std::lock_guard lk(commit_mu_);
        return reap_backlog_.load(std::memory_order_relaxed) + retired_.size();
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

    /// Every field looked up at least once via find_by_scan_field/
    /// find_by_cached_field/for_each_referrer/find_referrers/
    /// find_cached_referrers, for the life of this Model -- see
    /// Model::lookup_diagnostics(). `stats` is a direct, unformatted copy of
    /// Model's own internal counters (the atomics read once, into plain
    /// values -- nothing here is live or updates after the copy): no
    /// string work happens until to_string() actually needs one, which is
    /// the only consumer that does. There is no general way in this
    /// project to recover a field's OWN name from a bare FieldLookupKey
    /// (member_class_t/field_tag carry no string) -- to_string() demangles
    /// `type` for the declaring TYPE's name, and falls back to `field`'s
    /// raw address to disambiguate two looked-up fields on the same type
    /// that were never given a name (see FieldKeyReader::key()/
    /// RefIndexReader::index()'s own `name` parameter).
    class LookupDiagnostics {
    public:
        std::unordered_map<FieldLookupKey, LookupCounts, FieldLookupKeyHash> stats;

        /// Human-readable table, sorted by total calls (descending) so the
        /// fields that matter most for a caching decision sort to the top.
        std::string to_string() const;
    };

    /// A point-in-time readout of internal state, for observability and
    /// debugging -- nothing here is part of the model's published data, and
    /// nothing here is versioned. See diagnostics()'s own doc comment for
    /// exactly what each field means and how stale it can be.
    struct Diagnostics {
        std::uint64_t version = 0;              ///< == current_version() at the moment of the call
        std::uint64_t transactions_begun = 0;    ///< next_txn_id_ - 1: Transactions minted so far via
                                                 ///< begin(), committed or not -- NOT a commit count

        /// Live object count for one type, plus its demangled name. The name
        /// comes from a live representative object (ObjectBase::type()) --
        /// TypeTag alone (see model::TypeTag) carries no name, so a type
        /// whose every object has since been deleted (its by_type_ entry
        /// still exists, just empty) has no representative to name it.
        struct TypeCount {
            std::string type_name;  ///< "<no live object of this type>" if live_count == 0
            std::size_t live_count = 0;
        };
        std::size_t live_object_count = 0;  ///< sum of live_by_type[*].live_count
        std::vector<TypeCount> live_by_type;

        std::size_t chunk_count = 0;      ///< spine_.size() -- allocated Chunks, kChunkSize slots each
        std::size_t slots_allocated = 0;  ///< next_slot_ -- high-water mark of ever-allocated slots
        std::size_t slots_free = 0;       ///< free_slots_.size() -- recycled slots ready for reuse
        std::size_t slots_exhausted = 0;  ///< == exhausted_slots()

        std::size_t key_indexed_fields = 0;               ///< by_field_.size()
        std::size_t cached_value_indexed_fields = 0;       ///< by_cached_field_.size()
        std::size_t cached_reference_indexed_fields = 0;   ///< by_cached_reference_.size()
        std::size_t reverse_index_targets = 0;  ///< referrers_.size() -- slots with >=1 referrer
        std::size_t reverse_index_edges = 0;    ///< total referrer edges, summed across all targets

        bool pre_transactions_hook_installed = false;  ///< see set_pre_transactions()
        bool pre_commit_hook_installed = false;        ///< see set_pre_commit()

        std::size_t live_snapshot_versions = 0;   ///< live_.size() -- distinct pinned versions
        std::size_t live_snapshot_refs = 0;       ///< total Snapshots + open Transaction bases pinned
        std::uint64_t reclamation_watermark = 0;  ///< oldest version anything still needs;
                                                  ///< == version (above) if nothing is pinned
        std::size_t reap_backlog = 0;             ///< == reap_backlog()

        std::size_t subscriber_count = 0;  ///< live Subscriptions (see subscribe())

        /// One (version, change count) pair per commit still retained for
        /// try_commit()'s id-overlap conflict check -- NOT the changes
        /// themselves, so this stays cheap regardless of how large any one
        /// commit was. Oldest first. Pruned the same way changelog_ is
        /// (prune_changelog): once no live Transaction base / Snapshot could
        /// still need an entry, it's gone -- so seeing fewer entries on a
        /// later call is normal, not a bug.
        std::vector<std::pair<std::uint64_t, std::size_t>> retained_commit_history;

        /// Cumulative try_commit()/try_commit_without_undo() outcome counts
        /// since this Model was constructed -- monotonically increasing,
        /// never reset, never pruned (unlike retained_commit_history above,
        /// which only covers the still-live changelog window). This is the
        /// number to watch for whether writers are thrashing against each
        /// other at this Model's target scale (many writer threads racing
        /// try_commit()): a rising commits_conflicted relative to
        /// commits_succeeded means real contention, not a bug. Does NOT
        /// include run_pre_transaction()/run_pre_transaction_without_undo()
        /// calls (their outcome is folded into the enclosing try_commit()
        /// call's own PrecommitConflict, exactly once) or
        /// commit_bulk_without_undo() (a single-writer, exclusive-access
        /// path with no OCC contention to observe).
        std::uint64_t commits_succeeded = 0;
        std::uint64_t commits_conflicted = 0;
        std::uint64_t commits_vetoed = 0;
        std::uint64_t commits_invalid = 0;
        std::uint64_t commits_precommit_conflicted = 0;
    };

    /// Builds a Diagnostics readout. Not a hot-path call: takes commit_mu_
    /// long enough to copy writer-private state (spine/index sizes,
    /// referrers_, changelog_ summaries), THEN (fully released first)
    /// ver_mu_ for live_, THEN (also released first) subs_mu_ for the
    /// subscriber count -- one at a time, never nested, the same
    /// "sequential, not nested" trick release_version() uses to stay outside
    /// the commit_mu_ -> ver_mu_ -> reap_mu_ order instead of risking a new
    /// cycle (invariant 10). Because the three sections are three separate
    /// critical sections rather than one held together, the result is an
    /// approximation, not a single atomic instant -- e.g. `version` and
    /// `live_snapshot_versions` could each reflect a slightly different
    /// moment if a commit lands in between. That's the right tradeoff here:
    /// a diagnostics call should never hold up a live commit for longer than
    /// copying one of these sections takes.
    Diagnostics diagnostics() const;

    /// Records one call to a cached-or-not field-lookup function -- called
    /// only from Snapshot::find_by_scan_field/find_by_cached_field/
    /// for_each_referrer/find_cached_referrers (via Snapshot::
    /// record_field_lookup), never directly. Public because Snapshot is not
    /// a friend and there is no reason to make it one for this: the method
    /// is safe to call from anywhere, on any live Model, from any thread.
    ///
    /// One atomic increment per call, globally visible immediately -- no
    /// per-thread staging and no explicit flush step, unlike an earlier
    /// design this replaced: that one kept counts in a thread_local map so
    /// the actual lookup call never touched an atomic, but a count from a
    /// thread that never called flush_lookup_stats() itself was invisible to
    /// every OTHER thread's diagnostics() call, indefinitely -- wrong for
    /// "how has this Model been used, across every thread, for its whole
    /// life," which is the actual question this exists to answer.
    ///
    /// `type` is `typeid(ClassT)` at the call site -- a stable, process-wide
    /// address (see FieldLookupKeyHash) -- deliberately NOT demangled here:
    /// that string work happens only in lookup_diagnostics()/
    /// LookupDiagnostics::to_string(), which run rarely, off this path. The
    /// lock taken here is a std::shared_mutex, held SHARED for the
    /// (steady-state, after the first call for any given field) case where
    /// the entry already exists, and exclusively only to insert a field
    /// never seen before on this Model -- see the .cpp for why that split
    /// is safe.
    void record_field_lookup(const std::type_info& type, const void* field, bool cached) const;

    /// This field's cached-vs-uncached call counts, for the life of this
    /// Model -- zero if Field was never looked up via find_by_scan_field/
    /// find_by_cached_field/for_each_referrer/find_referrers/
    /// find_cached_referrers. The direct way to ask "is define_cached_
    /// fields()/define_cached_references() paying for itself on THIS field"
    /// without needing lookup_diagnostics()'s enumerate-everything list.
    template <auto Field>
    LookupCounts lookup_stats() const {
        using ClassT = member_class_t<decltype(Field)>;
        return lookup_stats_raw(typeid(ClassT), field_tag<Field>());
    }

    /// Untyped counterpart of lookup_stats<Field>() -- same "_raw" naming as
    /// create_raw/update_raw/remove_raw/peek_raw/find_by_key_raw for the
    /// untyped body behind a typed template. Also what lookup_diagnostics()
    /// builds its report from.
    LookupCounts lookup_stats_raw(const std::type_info& type, const void* field) const;

    /// Every field looked up at least once, for the life of this Model, each
    /// labeled with its declaring type's demangled name (see
    /// record_field_lookup's comment on why demangling happens here and
    /// nowhere hotter). Order is unspecified; LookupDiagnostics::to_string()
    /// sorts it for display. Deliberately NOT part of Model::Diagnostics/
    /// diagnostics(): that call is documented as a bounded, point-in-time
    /// snapshot of a handful of small, fixed-size counters, and this is an
    /// unbounded, separately-locked (field_lookup_mu_) collection that can
    /// grow for the life of the Model -- folding it in would make every
    /// diagnostics() call pay for a query nobody asked for.
    LookupDiagnostics lookup_diagnostics() const;

    /// One step of a committed transaction's inverse, captured for free at
    /// the exact point apply already reads the relevant pre-image pointer
    /// (see apply_create/apply_update/clone_for_cascade_null/remove_raw in
    /// model.cpp). `Recreate`'s object gets a genuinely NEW id when
    /// resurrected -- generation never recycles (invariant 5) -- so its
    /// `id` here is the OLD one, informational only.
    struct UndoAction {
        enum class Kind : std::uint8_t { Recreate, Remove, RestoreUpdate };  // mirrors ChangeKind's style
        Kind kind;
        Id id;  ///< Remove/RestoreUpdate: the real id to act on.
               ///< Recreate: the OLD id (about to become stale).

        /// The object's value as of just before the change this action
        /// undoes -- a clone (Recreate, RestoreUpdate), null for Remove.
        /// Named to match Transaction::peek_before(), the read-side analog
        /// of the same idea ("the value... before any local edit") -- NOT
        /// `snapshot`, which would collide with the unrelated `Snapshot`
        /// class (a whole-model, point-in-time view) this single object's
        /// clone has nothing to do with.
        std::unique_ptr<ObjectBase> previous_value;
    };

    /// One committed transaction's full inverse recipe. `touched` is every
    /// id that commit's OWN changes_ mentioned (Created ∪ Updated ∪
    /// Deleted) -- deliberately broader than just the ids apply_undo()
    /// would need to act on, so a later commit touching ANY of them
    /// invalidates this entry (see the undo list's own comment, next to
    /// undo_list_, for why the broad reading was chosen over a narrower
    /// one that would only watch RestoreUpdate targets).
    struct UndoEntry {
        /// The version this commit PUBLISHED as -- `r->version` at the
        /// point it was handed to readers (model.cpp's try_commit()), the
        /// same numbering space as Snapshot::version(), changelog_ entries,
        /// and diagnostics().retained_commit_history. This is the key
        /// take_undo() looks up by, and it's deliberately NOT
        /// Transaction::id(): id() is minted at begin(), in attempt-
        /// construction order, while `version` is minted at publish, in
        /// commit order -- under multi-writer racing those orders diverge
        /// (a transaction that begins first can lose a race, retry, and
        /// publish after one that began later). Using `version` is what
        /// lets a caller compare an undo entry against a Snapshot they're
        /// holding ("is this newer or older than what I'm looking at?");
        /// id() carries no such relationship to publish order and would
        /// make the field a bare opaque token instead.
        std::uint64_t version;

        /// This commit's inverse: at most ONE UndoAction per id it touched --
        /// collapsed (see Model::collapse_undo_actions()) to the NET effect
        /// even when that id's own resolved changeset shows up more than
        /// once (a same-transaction create-then-cascade-delete, or an
        /// update()/cascade-null followed by a cascade delete of the same
        /// object -- see Change's own doc comment for why changes_ itself is
        /// allowed to report an id twice; actions never is). apply_undo()
        /// doesn't care what order these run in (it pre-mints every
        /// Recreate's new id in its own first pass before remapping
        /// anything) -- so no ordering is promised here.
        std::vector<UndoAction> actions;

        /// Every id this commit's OWN changes_ mentioned (Created ∪
        /// Updated ∪ Deleted) -- see the struct comment above for why this
        /// is broader than just the ids apply_undo() would act on. A later
        /// commit that touches ANY id in this set invalidates the whole
        /// entry (the loop in try_commit() that erases conflicting undo_list_
        /// entries), not just the actions that reference it.
        std::unordered_set<Id, IdHash> touched;

        std::string name;  ///< copied from the committing Transaction::name() (see publish_now())
        std::any data;     ///< copied from the committing Transaction::data() (see publish_now())

        /// The committing Transaction::id() -- purely informational, NOT a
        /// lookup key (take_undo() still keys on `version`, for the reasons
        /// in that member's comment). This is what lets a caller correlate
        /// an entry back to the specific attempt that produced it, e.g.
        /// matching it against PostCommitFn logging keyed by id() -- 0 for
        /// paths with no Transaction to draw an id from (commit_bulk_without_undo()),
        /// though those never produce undo data in the first place.
        std::uint64_t txn_id = 0;

        /// Sum, over `actions`, of `sizeof(UndoAction)` plus
        /// `previous_value->byte_size()` for the ones that carry a clone
        /// (Remove actions don't -- see UndoAction::previous_value). Computed
        /// once, in publish_now(), at the same point `actions` reaches its
        /// final (collapsed) contents -- NOT recomputed by take_undo() or
        /// apply_undo(). Same "approximate, not exact" caveat as
        /// ObjectBase::byte_size(): touched/name/data's own bytes aren't
        /// included, so this undercounts by their size. What
        /// set_max_undo_memory_bytes() budgets against.
        std::size_t approx_bytes = 0;
    };

    /// Lightweight, copyable summary of one UndoEntry still in the list --
    /// what list_undo() returns, so browsing what's available doesn't force
    /// copying every entry's (potentially many) snapshot clones. `data` is
    /// the one exception to "lightweight": it's a copy of whatever the
    /// committing Transaction's own data() held, which is only as cheap to
    /// copy as the caller made it -- an empty std::any (the default) is
    /// free, but a caller that passed something expensive pays for that
    /// copy on every list_undo() call, not just once.
    struct UndoSummary {
        /// Copied verbatim from UndoEntry::version -- see that member's
        /// comment for why it's the commit's PUBLISHED version (the same
        /// number CommitResult::snapshot.version() returned for this
        /// commit), not Transaction::id(). Pass this straight to
        /// take_undo() to retrieve the full entry.
        std::uint64_t version;

        /// UndoEntry::actions.size(), so a caller can gauge an entry's
        /// size (e.g. for display, or to skip take_undo()-ing something
        /// huge) without copying the vector of UndoActions -- each of
        /// which owns a full object clone -- just to count them.
        std::size_t action_count;

        std::string name;  ///< copied from UndoEntry::name; see that member's comment
        std::any data;     ///< copied from UndoEntry::data; see that member's comment

        /// Copied verbatim from UndoEntry::txn_id -- see that member's
        /// comment. Informational only; take_undo() still keys on `version`.
        std::uint64_t txn_id = 0;

        /// Copied verbatim from UndoEntry::approx_bytes -- see that member's
        /// comment. A cheap (already-computed, just copied) way to gauge an
        /// entry's memory footprint from list_undo(), the same role
        /// action_count plays for its shape.
        std::size_t approx_bytes = 0;
    };

    /// Every undo entry still in the list, oldest first. See UndoEntry's
    /// own comment for what "still in the list" means -- an entry a later
    /// commit conflicted with is silently gone by the time this is called,
    /// same as a pruned changelog_ entry.
    std::vector<UndoSummary> list_undo() const;

    /// Removes and returns ownership of the entry for `version`, or
    /// nullopt if no such entry is in the list (never committed with undo
    /// data, already taken, or pruned by a later conflicting commit).
    /// Ownership transfer, not a copy: UndoEntry holds move-only
    /// unique_ptr<ObjectBase> clones. Once taken, this entry is the
    /// caller's alone -- it will never be pruned out from under them (it's
    /// no longer in undo_list_ to prune), and calling apply_undo() with it
    /// is safe to retry as many times as they like (apply_undo clones
    /// internally; it never consumes the entry it's given).
    std::optional<UndoEntry> take_undo(std::uint64_t version);

    /// Drops every entry, to reclaim memory (per-entry snapshot clones are
    /// proportional to that commit's OWN change size, but a long-running
    /// Model with many commits and nobody ever calling take_undo() would
    /// otherwise retain all of them forever -- this is that explicit,
    /// caller-driven reclaim, the same "give the primitive, let the caller
    /// bound it" tradeoff Subscription's queue depth already makes).
    void clear_undo_list();

    /// Caps how many entries undo_list_ is allowed to hold. Enforced lazily,
    /// only at the moment a new entry would be appended (publish_now()): if
    /// the list is already at (or over, having been shrunk since the last
    /// add) the cap, the OLDEST entries are dropped first -- same "oldest
    /// first" ordering list_undo() documents -- until there is room for the
    /// one about to be added. Does not retroactively prune when THIS call
    /// LOWERS a positive cap; a list already over the new limit only
    /// shrinks the next time a commit would add to it.
    ///
    /// n == 0 is the one exception, and it IS immediate, not lazy: it means
    /// "keep no undo history at all," enforced right here by clearing
    /// undo_list_ (and undo_touch_index_ along with it) on the spot, not
    /// just going forward. This isn't just tidiness -- it turns "the cap is
    /// 0" and "undo_list_ is empty" into the SAME fact for as long as the
    /// cap stays 0 (nothing else can ever add to undo_list_ while
    /// max_undo_list_size_ == 0 -- apply_transaction_contents() folds this
    /// cap into keep_undo before publish_now() is ever reached, so
    /// pending_undo_ stays empty too), which is what lets publish_now()
    /// skip its entire prune-and-append step outright when the cap is 0,
    /// instead of running the conflict-prune check (see undo_touch_index_)
    /// on every single commit for a list that's provably always empty
    /// anyway. A 0 cap also skips the per-object pre-image clone during
    /// apply (apply_update's baseline->clone(), remove_raw's
    /// victim->clone(), clone_for_cascade_null's cur->clone()) -- no
    /// per-object clone for a commit whose undo data would just be
    /// discarded.
    ///
    /// Read (and, for n == 0, mutated) under commit_mu_ (already held at
    /// that point), same as every other commit_mu_-protected member.
    /// Default is unbounded (matching every version of this Model before
    /// this method existed). Takes commit_mu_, same locking contract as
    /// set_pre_commit/set_pre_transactions/set_post_commit.
    void set_max_undo_list_size(std::size_t n) {
        std::lock_guard lk(commit_mu_);
        max_undo_list_size_ = n;
        if (n == 0) {
            undo_list_.clear();
            undo_touch_index_.clear();
            total_undo_bytes_ = 0;
        }
    }

    /// Same idea as set_max_undo_list_size(), budgeted by UndoEntry::
    /// approx_bytes instead of entry count -- the two caps are independent
    /// and both apply (publish_now() evicts oldest-first against whichever
    /// one, or both, the incoming entry would violate). n == 0 gets the same
    /// immediate-clear treatment (and the same clone-skipping benefit,
    /// folded into keep_undo alongside max_undo_list_size_ -- see
    /// apply_transaction_contents()) as set_max_undo_list_size(0).
    ///
    /// Eviction only ever removes OTHER, already-published entries -- never
    /// the one currently being added. So a single commit whose own
    /// approx_bytes exceeds `n` is still appended in full, leaving
    /// total_undo_bytes_ temporarily over budget; eviction resumes (this
    /// entry included, once it's no longer the newest) the next time
    /// anything would be added. The cap bounds steady-state memory, not any
    /// one commit's worst case -- an undo entry is never the reason a commit
    /// fails, only ever something garbage-collected around one.
    ///
    /// Default is unbounded (numeric_limits::max()), matching
    /// max_undo_list_size_'s default. Same locking contract: takes
    /// commit_mu_.
    void set_max_undo_memory_bytes(std::size_t n) {
        std::lock_guard lk(commit_mu_);
        max_undo_bytes_ = n;
        if (n == 0) {
            undo_list_.clear();
            undo_touch_index_.clear();
            total_undo_bytes_ = 0;
        }
    }

    /// Builds a fresh Transaction from `entry` and commits it: mints a new
    /// local id for every Recreate action first (mirrors the pre-mint
    /// pass, so victim-to-victim edges among resurrected objects remap
    /// correctly regardless of order -- see UndoRemapper), then applies
    /// every action, remapping each captured snapshot's own ref fields
    /// (old real id -> new local id, for anything ALSO being recreated
    /// this same call) before writing it. Takes `entry` by const reference
    /// and clones every action's snapshot again rather than consuming it,
    /// so a Conflict/Invalid result never destroys the caller's only copy
    /// -- they can inspect what beat them (Model::snapshot()) and retry.
    /// Not called from inside any hook: this calls begin()/try_commit()
    /// itself, exactly like any other ordinary application code would.
    CommitResult apply_undo(const UndoEntry& entry);

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
        /// Same shared_ptr instance publish_now() also hands every
        /// subscriber's Update -- one changeset, not a second copy just for
        /// the changelog. See Update::changes.
        std::shared_ptr<const std::vector<Change>> changes;
    };

    // All of the following run only under commit_mu_ (invariant 7), except
    // release_version / reaper_loop / enqueue_retired, which have their own
    // locking noted below.

    /// First touch of a chunk per attempt clones it (tracked in dirty_);
    /// later writes hit the clone in place. Published chunks stay immutable.
    Chunk* cow(std::uint32_t chunk_index);

    /// Pop the LIFO free list (retiring generation-exhausted slots as they
    /// surface -- see Id's width note), else mint a fresh slot. Undo-logged.
    std::uint32_t alloc_slot();

    /// nullopt == valid. `pending`, when non-null, names real ids this
    /// attempt has minted but not yet installed (see the pre-mint pass in
    /// apply_transaction_contents): a Ref<> to one of those is valid --
    /// either every create in the attempt installs, or the whole attempt
    /// rolls back, so "minted" is as good as "installed" for integrity.
    std::optional<IntegrityError> validate(const ObjectBase* o,
                                           const std::unordered_set<Id, IdHash>* pending
                                           = nullptr) const;

    /// One entry per id in txn.local_updated_ (every entry is a real,
    /// non-null clone, so this is unconditional): that object's baseline
    /// define_keys() fields, via baseline->each_field_key(), keyed by slot
    /// index. Pure data collection, no validation -- run once, up front, by
    /// apply_transaction_contents, so BOTH validate_field_key_uniqueness()
    /// (which needs the pre-update value to tell "unchanged"/"vacated" from
    /// a genuine new claim) and, later, apply_update() (which forwards the
    /// same entry to reconcile_field_keys(), see its `old_keys_hint`
    /// parameter) can read it without either one walking a given baseline's
    /// each_field_key() more than this one time.
    ///
    /// The return type, outside in:
    ///   std::unordered_map<K, V> -- one entry per updated object; K/V below.
    ///     K = std::uint32_t                the updated object's SLOT index
    ///                                      (Id::index with the generation
    ///                                      stripped) -- the exact key
    ///                                      txn.local_updated_ itself uses,
    ///                                      so a caller already iterating
    ///                                      that map (apply_transaction_
    ///                                      contents's update loop) can look
    ///                                      an entry up with no translation.
    ///     V = std::vector<std::pair<F, S>> that object's baseline
    ///                                      define_keys() fields, ONE PAIR
    ///                                      PER DECLARED FIELD -- a
    ///                                      linear-scan vector, not a nested
    ///                                      map, because a type has a
    ///                                      handful of define_keys() fields
    ///                                      at most (same tradeoff as
    ///                                      reconcile_out_refs()'s own
    ///                                      old_targets vector).
    ///       F = const void*                the field's identity --
    ///                                      field_tag<Field>(), exactly what
    ///                                      each_field_key()/FieldKeyReader::
    ///                                      key() hand out; an opaque
    ///                                      address, compared, never
    ///                                      dereferenced.
    ///       S = std::string                that field's value, in
    ///                                      to_field_key()'s canonical
    ///                                      string form, AS OF THE BASELINE
    ///                                      (before this transaction's own
    ///                                      edit) -- what by_field_ indexed
    ///                                      this object under prior to the
    ///                                      update. Canonical string, not
    ///                                      the field's native type, for
    ///                                      two reasons: it lets one
    ///                                      vector hold define_keys()
    ///                                      fields of different types
    ///                                      (std::string, int, ...) with
    ///                                      no variant, and it lets
    ///                                      validate_field_key_uniqueness()
    ///                                      tell old from new with a
    ///                                      plain == against the new
    ///                                      key's own to_field_key() form.
    std::unordered_map<std::uint32_t, std::vector<std::pair<const void*, std::string>>>
    collect_update_baseline_field_keys(const Transaction& txn) const;

    /// Read-only, whole-transaction pass over every define_keys() field this
    /// transaction's creates/updates touch, run once at the very start of
    /// apply_transaction_contents -- BEFORE anything mutates -- so a failure
    /// here needs no rollback. Two things a per-object, apply-order check
    /// cannot get right on its own (txn.local_updated_ is an unordered_map;
    /// apply order is not call order):
    ///   - A key claimed by two different objects in the SAME transaction
    ///     (two creates, or a create and an update) is always rejected.
    ///   - A key an update is VACATING this same transaction never blocks
    ///     another object's claim on it, no matter which one applies first
    ///     -- this is what makes "A moves off key K, B moves onto K, same
    ///     transaction" a legal swap. reconcile_field_keys()'s own
    ///     conditional erase (only erase the old key if this object still
    ///     holds it) is what keeps the actual mutation order-safe once this
    ///     check has already proven the final state collision-free.
    /// A collision against another object's CURRENTLY COMMITTED key is
    /// always CommitStatus::Invalid, never Conflict: unlike Ref<> integrity,
    /// key uniqueness isn't part of this design's object-write-set OCC (see
    /// CLAUDE.md) -- there is no "existed at base" test that distinguishes a
    /// real race from a caller who should have checked find_by_key() first,
    /// so this doesn't attempt one.
    ///
    /// `old_keys` is collect_update_baseline_field_keys()'s own output,
    /// passed in rather than recomputed -- keyed by slot, one entry per id
    /// in txn.local_updated_. This function only READS it (to tell
    /// "unchanged"/"vacated" from a new claim); it does not itself walk any
    /// baseline's each_field_key() -- that traversal belongs entirely to the
    /// collection step, so this function is pure validation, nothing more.
    std::optional<IntegrityError> validate_field_key_uniqueness(
        const Transaction& txn,
        const std::unordered_map<std::uint32_t, std::vector<std::pair<const void*, std::string>>>&
            old_keys) const;

    // Reverse-index (referrers_) maintenance: add/drop an object's whole
    // outgoing edge set (create/delete), or diff before->after (update).
    // Every edit is undo-logged -- a rollback that missed one would leave a
    // phantom or missing edge for a LATER cascade to resolve against.
    void add_out_refs(const ObjectBase* o);
    void drop_out_refs(const ObjectBase* o);
    void reconcile_out_refs(const ObjectBase* before, const ObjectBase* after);

    // Same trio for the unique key index (by_field_)...
    void add_field_keys(const ObjectBase* o);
    void drop_field_keys(const ObjectBase* o);
    /// `old_keys_hint`, when non-null, is `before`'s already-computed
    /// define_keys() field/value list -- so this skips walking
    /// before->each_field_key() itself. Populated only by
    /// validate_field_key_uniqueness() for an ordinary Transaction::update()
    /// (apply_update passes it through); every OTHER caller (the cascade-null
    /// branch in remove_raw, which reconciles a survivor's fields against a
    /// baseline validate_field_key_uniqueness never saw) passes nullptr and
    /// gets the original behavior, deriving old_keys from `before` itself.
    void reconcile_field_keys(
        const ObjectBase* before, const ObjectBase* after,
        const std::vector<std::pair<const void*, std::string>>* old_keys_hint = nullptr);

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

    /// Log this key's pre-attempt value into the undo log, but only on the
    /// FIRST touch of that key (type tag / field tag) this attempt -- mirrors
    /// cow()'s dirty_ (first-touch-clones-the-chunk) trick, applied to the
    /// four whole-map-handle indexes instead of a Chunk. Why this is safe:
    /// rollback_apply() replays log() closures in REVERSE (undo_.rbegin() ->
    /// rend()), and each closure OVERWRITES the map wholesale -- so of N
    /// captures taken across N edits to the same key in one attempt, only the
    /// FIRST (executed LAST during rollback) has any effect on the final
    /// restored state; every later capture is silently clobbered before
    /// rollback finishes. Logging on every touch of a type/field a large
    /// transaction edits thousands of times was therefore pure waste: one
    /// closure (alloc + capture + eventual invocation) per touch, when the
    /// attempt only ever needed the very first one. These dirty_by_*_ sets
    /// are cleared alongside dirty_ at the end of every attempt
    /// (rollback_apply()/publish_now()) -- same attempt-scoped lifetime.
    void log_by_type_once(TypeTag tag);
    void log_by_field_once(const void* field);
    void log_by_cached_field_once(const void* field);
    void log_by_cached_reference_once(const void* field);

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
    // The log is proportional to the changes made, never to model size.
    //
    // UndoOp (a closed tagged union, not std::function<void()>): profiling
    // basic_record_bench found this log's PUSH side -- not replay, which only
    // runs on the rare rollback path -- among the largest costs of a commit
    // in its own right: every alloc_slot()/set_slot()/changes_ push (i.e.
    // every single create/update/remove, not just once per attempt like
    // log_by_type_once et al.) built a std::function closure, and most of
    // those closures capture enough state (e.g. set_slot's [this, slot,
    // prev_obj, prev_gen]) to exceed libstdc++'s small-object buffer and
    // heap-allocate -- allocated, then almost always simply discarded
    // wholesale on the (overwhelmingly common) successful-commit path
    // (undo_.clear() below). A closed variant of small, POD-ish "what to
    // undo" structs removes that allocation entirely: pushing one is just
    // writing its fields into the vector's next slot, same as any other
    // value type.
    //
    // This does give up the ORIGINAL reason closures were chosen here (see
    // git history): a hand-written closure at the write site can't drift
    // from what it undoes, while a Kind/payload pair COULD, in principle,
    // stop matching its apply_undo_op() case somewhere else in this file.
    // std::variant (rather than a raw union or an enum + void*) is what
    // keeps that risk small: every alternative is a distinct, named,
    // strongly-typed struct, log()/apply_undo_op() are both exhaustively
    // checked by the compiler (a missing overload is a compile error, a
    // std::visit is never partial), and each Kind is still defined
    // immediately next to the handful of log() call sites that produce it.
    // GenericUndo (a std::function<void()> fallback) is kept for the four
    // log_by_*_once() closures: those capture a whole PersistentMap/Set
    // `prev` by move and fire at most once per DISTINCT index touched per
    // attempt (see log_by_type_once's own comment) -- bounded by the number
    // of declared fields/types, never by object count -- so they were never
    // the cost this change targets, and giving each of the four its own
    // dedicated, differently-shaped Kind would add real complexity for a
    // closure that was already cheap relative to model size.
    struct PushFreeSlot {
        std::uint32_t slot;
    };
    struct DecExhaustedSlots {};
    struct DecNextSlot {};
    struct RestoreSlot {
        std::uint32_t slot;
        const ObjectBase* prev_obj;
        std::uint32_t prev_gen;
    };
    struct ReferrersPopBack {
        std::uint32_t key;
    };
    struct ReferrersPushEdge {
        std::uint32_t key;
        RefEdge edge;
    };
    struct PopChanges {};
    struct DeleteObject {
        ObjectBase* obj;
    };
    struct PopRetired {};
    struct PopFreeSlot {};
    struct RestoreReferrersBucket {
        std::uint32_t key;
        std::vector<RefEdge> saved;
    };
    struct GenericUndo {
        std::function<void()> fn;
    };
    using UndoOp = std::variant<PushFreeSlot, DecExhaustedSlots, DecNextSlot, RestoreSlot,
                                ReferrersPopBack, ReferrersPushEdge, PopChanges, DeleteObject,
                                PopRetired, PopFreeSlot, RestoreReferrersBucket, GenericUndo>;

    template <class Op>
    void log(Op op) {
        undo_.emplace_back(std::move(op));
    }

    // One overload per UndoOp alternative -- see rollback_apply()'s use of
    // std::visit, which is exhaustive at compile time (a Kind added to the
    // variant above without a matching overload here is a compile error,
    // not a silent no-op).
    void apply_undo_op(PushFreeSlot op) { free_slots_.push_back(op.slot); }
    void apply_undo_op(DecExhaustedSlots) { --exhausted_slots_; }
    void apply_undo_op(DecNextSlot) { --next_slot_; }
    void apply_undo_op(RestoreSlot op) {
        const std::uint32_t cc = op.slot >> kChunkBits, ii = op.slot & kChunkMask;
        Chunk* c2 = cow(cc);
        c2->obj[ii] = op.prev_obj;
        c2->gen[ii] = op.prev_gen;
    }
    void apply_undo_op(ReferrersPopBack op) {
        auto it = referrers_.find(op.key);
        if (it != referrers_.end()) {
            it->second.pop_back();  // exact inverse of the push_back it undoes
            if (it->second.empty()) referrers_.erase(it);
        }
    }
    void apply_undo_op(ReferrersPushEdge op) { referrers_[op.key].push_back(op.edge); }
    void apply_undo_op(PopChanges) { changes_.pop_back(); }
    void apply_undo_op(DeleteObject op) { delete op.obj; }
    void apply_undo_op(PopRetired) { retired_.pop_back(); }
    void apply_undo_op(PopFreeSlot) { free_slots_.pop_back(); }
    void apply_undo_op(RestoreReferrersBucket op) { referrers_[op.key] = std::move(op.saved); }
    void apply_undo_op(GenericUndo op) { op.fn(); }

    /// Point a slot at an object (or null) with a new generation, through
    /// cow(); logs the exact inverse (previous object + generation).
    void set_slot(std::uint32_t slot, const ObjectBase* obj, std::uint32_t gen);

    // ---- commit_bulk_without_undo() internals only -- see the section comment above
    // Model::begin_bulk() before reaching for these anywhere else --------
    //
    // Exact behavioral twins of set_slot()/add_out_refs()/add_field_keys()/
    // add_cached_fields()/add_cached_references() above, minus the log()
    // call each one makes. That single omission is the entire point: it is
    // the act of LOGGING (capturing and retaining the prior state so it can
    // be replayed) that makes a huge, single Transaction expensive, not the
    // index mutation itself -- see commit_bulk_without_undo()'s own comment for the
    // measured cost this avoids. Calling these from anywhere a failure
    // might need to be unwound is a correctness bug: nothing here is
    // undo-able, on purpose.
    //
    // Deliberately NOT merged into the logged versions behind a `bool
    // should_log` parameter, even though each pair's body is otherwise
    // identical: a stray `true` at the one call site that must never log
    // would silently reintroduce the exact per-object undo-log retention
    // this whole mechanism exists to avoid, with no signal at the call site
    // and nothing for the type system to catch. Separate names mean
    // "commit_bulk_without_undo() calls the _no_log ones" is a static, grep-able fact
    // instead of something you have to trace a boolean through.
    void set_slot_no_log(std::uint32_t slot, const ObjectBase* obj, std::uint32_t gen);
    void add_out_refs_no_log(const ObjectBase* o);
    void add_field_keys_no_log(const ObjectBase* o);
    void add_cached_fields_no_log(const ObjectBase* o);
    void add_cached_references_no_log(const ObjectBase* o);

    // ---- try_commit() internals (ALL require commit_mu_ already held) ------
    // apply_* return nullopt on success, or the integrity violation that
    // aborted this attempt -- the caller must then rollback_apply().
    // `remap` arrives at apply_create FULLY minted (the pre-mint pass in
    // apply_transaction_contents), and `pending` is the set of its values:
    // real ids minted this attempt whose slots may not be installed yet.
    // `keep_undo` (default true, so every pre-existing call site is
    // unaffected) gates ONLY the pending_undo_ capture -- try_commit_
    // without_undo() is the one caller that passes false, all the way down
    // this whole chain, so it skips the per-object clone() entirely rather
    // than capturing it and throwing it away.
    std::optional<IntegrityError> apply_create(std::unique_ptr<ObjectBase> o,
                                               std::unordered_map<std::uint32_t, Id>& remap,
                                               const std::unordered_set<Id, IdHash>& pending,
                                               bool keep_undo = true);
    /// `old_field_keys_hint`, when non-null, is the target's baseline
    /// define_keys() field/value list, ALREADY computed by
    /// collect_update_baseline_field_keys() (see its own doc comment) --
    /// forwarded straight through to reconcile_field_keys() so this update
    /// doesn't walk baseline->each_field_key() a second time. nullptr for
    /// any caller (there are none today besides apply_transaction_contents's
    /// own update loop) that hasn't already computed it.
    std::optional<IntegrityError> apply_update(
        std::unique_ptr<ObjectBase> clone, std::unordered_map<std::uint32_t, Id>& remap,
        const std::vector<std::pair<const void*, std::string>>* old_field_keys_hint = nullptr,
        bool keep_undo = true);
    /// Cascade BFS, called from try_commit()'s apply phase. `work` is EVERY
    /// remove() intent this transaction resolves (deferred local removes AND
    /// real remove_intents_), taken BY VALUE and used directly as the BFS's
    /// own frontier (the caller's vector becomes this one -- see the .cpp:
    /// there is no separate internal copy) -- fed to the SAME BFS in one
    /// call, one shared `visited` set, instead of one remove_raw() call per
    /// intent. Pass an rvalue (the call site does) and the caller's buffer is
    /// moved into `work` with zero allocation; passing an lvalue still
    /// compiles but copies, same as any other pass-by-value parameter.
    /// Correctness is unaffected by the batching: each id is still visited
    /// (and its cascade resolved) exactly once regardless of which seed's
    /// fan-out reaches it first, exactly as if a later intent's BFS had
    /// found it already-gone via peek_raw() -- the only thing batching
    /// changes is that finding avoids a second, separate BFS setup (a fresh
    /// `work`/`visited` allocation and a redundant referrers_ lookup) to
    /// discover it. The RELATIVE order of resulting Change/UndoAction
    /// entries between two INDEPENDENT intents' cascades is unspecified
    /// either way (see Change's own doc comment) -- only the phase order
    /// relative to creates/updates (this always runs after both; see
    /// apply_transaction_contents) and the per-id order within one id's own
    /// cascade (still guaranteed by `visited`) are load-bearing.
    std::vector<Id> remove_raw(std::vector<Id> work, bool keep_undo = true);
    // remove_raw's helper for a NULLABLE referrer: clone + install, so the
    // caller can null_ref() the field that pointed at the victim, then
    // reconcile immediately. See the .cpp for why reconciliation must happen
    // in the caller, after null_ref() -- not in here.
    ObjectBase* clone_for_cascade_null(Id id, bool keep_undo = true);

    /// The conflict check: every slot this transaction updated or intends to
    /// remove, tested against last_write_version_ (was that slot touched by
    /// ANY commit after this transaction's base?). Read-only; runs before
    /// anything is applied, so a Conflict here costs no rollback. Creates
    /// can't conflict (fresh slots) and aren't checked.
    std::vector<Id> check_id_overlap(const Transaction& txn) const;

    /// Drop changelog entries at or below the live watermark -- no open
    /// Transaction (its base pins a version in live_) can need them again.
    void prune_changelog();

    void rollback_apply();  // unwinds one failed try_commit() apply attempt

    // ---- shared by try_commit() and run_pre_transaction() -----------
    // Both publish a Transaction as a real, independent commit while already
    // holding commit_mu_; apply_transaction_contents/classify_apply_failure/
    // check_and_apply/publish_now are that shared machinery, factored out so
    // a pre-transaction gets EXACTLY the same conflict/apply/publish behavior
    // as the main transaction, with nothing bespoke to keep in sync.

    /// The creates-then-updates-then-removes apply loop (see try_commit()'s
    /// phase-order comment for why that order matters), finishing with
    /// collapse_undo_actions() (when keep_undo) so pending_undo_ has at most
    /// one action per id before the caller ever sees it. Returns nullopt on
    /// success; the caller must then check changes_ and eventually publish
    /// or, on failure, pass the returned error to classify_apply_failure().
    /// Also where `keep_undo` gets ANDed against `max_undo_list_size_ != 0`
    /// (a local fold, not threaded back to either caller's own keep_undo --
    /// see set_max_undo_list_size()'s doc comment for why publish_now()
    /// doesn't need it threaded back).
    std::optional<IntegrityError> apply_transaction_contents(
        Transaction& txn, std::unordered_map<std::uint32_t, Id>& remap, bool keep_undo = true);

    /// Collapses pending_undo_ so a given id has AT MOST ONE UndoAction, even
    /// though changes_ (left untouched -- see Change's own doc comment) may
    /// report that same id more than once. Multiple pending_undo_ entries for
    /// one id arise from ordinary sequences within a single transaction: a
    /// fresh create() immediately cascade-touched by removing something it
    /// points at (Remove, then RestoreUpdate or Recreate), an update()/
    /// cascade-null followed by a cascade DELETE of that same object
    /// (RestoreUpdate, then Recreate), or several cascade-nulls of different
    /// fields on one referrer (RestoreUpdate, RestoreUpdate, ...). Naively
    /// replaying every action apply_undo() would otherwise resurrect
    /// something that was never actually visible (a Recreate outliving an
    /// earlier Remove) or restore an intermediate, WRONG value (a later
    /// action capturing this same transaction's own earlier edit instead of
    /// the true pre-transaction baseline) -- apply_undo() makes no promise
    /// about the order actions run in, so leaving duplicates in is not safe
    /// even though they'd often cancel out by luck. Called once, from the
    /// tail of apply_transaction_contents() (guarded by keep_undo there, the
    /// same way every push into pending_undo_ already is), right after the
    /// create/update/remove loop and before that function returns success --
    /// never mid-apply and never on a failed attempt, so it has no
    /// interaction with the log()-based rollback mechanism at all. Reads
    /// changes_ (unmodified) only to recover, per id, which UndoAction was
    /// pushed first -- see the "index-parallel" assert in the .cpp.
    void collapse_undo_actions();

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
                                                 std::unordered_map<std::uint32_t, Id>& remap,
                                                 bool keep_undo = true);

    /// The publish tail: version bump, new Root, atomic store under ver_mu_,
    /// subscriber notify, retirees handed to the reaper, changelog append,
    /// scratch cleared. Always succeeds -- by the time it's called, nothing
    /// left to reject. Consumes changes_ (member scratch) into the result.
    /// `undo_name`/`undo_data`/`undo_txn_id` are copied onto this commit's
    /// UndoEntry, if it gets one (see pending_undo_'s own comment) --
    /// callers with no Transaction to draw them from (commit_bulk_without_undo(),
    /// which never produces undo data at all) pass empty/zero defaults.
    /// `keep_undo` gates only whether THIS commit gets added as a new entry
    /// -- existing undo_list_ entries this commit conflicts with are pruned
    /// regardless (see try_commit_without_undo()'s own doc comment for why
    /// that split is deliberate).
    CommitResult publish_now(std::unordered_map<std::uint32_t, Id> remap, std::string undo_name = "",
                             std::any undo_data = {}, bool keep_undo = true,
                             std::uint64_t undo_txn_id = 0);

    /// check_and_apply() + publish_now(), with NO veto-hook seam -- used only
    /// by run_pre_transaction()/run_pre_transaction_without_undo(), which
    /// must never invoke pre_commit_ for a pre-transaction (that hook is
    /// reserved for the main transaction). try_commit() does NOT call this:
    /// the main transaction still needs the veto seam between apply and
    /// publish, so it inlines check_and_apply() + publish_now() itself.
    /// Requires commit_mu_ already held. `keep_undo` is exactly
    /// publish_now()'s own parameter, passed straight through.
    CommitResult commit_pretransaction_locked(Transaction& txn, bool keep_undo = true);

    /// The body of try_commit()/try_commit_without_undo() that actually
    /// needs commit_mu_: everything from the pre-transactions phase through
    /// check_and_apply(), the pre_commit_ veto, and publish_now(). Factored
    /// out so try_commit_core() can copy post_commit_ out (see PostCommitFn)
    /// and invoke it AFTER the std::lock_guard wrapping this call has gone
    /// out of scope, instead of while still holding commit_mu_.
    CommitResult commit_main_locked(Transaction& txn, bool keep_undo = true);

    /// Shared body of try_commit()/try_commit_without_undo(): the empty-
    /// transaction fast path, then commit_mu_ + post_commit_ copy-out +
    /// commit_main_locked() + post_commit_ invoked (unlocked) afterward.
    /// The ONLY difference between the two public entry points is which
    /// `keep_undo` value they pass here -- factored into one place rather
    /// than duplicated so the post_commit_-outside-the-lock ordering (see
    /// commit_main_locked's own comment) can't drift between two copies.
    CommitResult try_commit_core(Transaction& txn, bool keep_undo);

    /// Shared body of run_pre_transaction()/run_pre_transaction_without_
    /// undo(): the two asserts, commit_pretransaction_locked(), and
    /// recording precommit_failed_/precommit_failure_ on anything other
    /// than Committed. The ONLY difference between the two public entry
    /// points is which `keep_undo` value they pass here -- same reasoning
    /// as try_commit_core().
    CommitResult run_pre_transaction_core(Transaction& txn, bool keep_undo);

    /// EXPERIMENTAL: backs the Node/Leaf allocations of by_type_ (see
    /// apply_create's by_type_[tag] seeding in model.cpp, the only current
    /// user) instead of routing them through glibc's general-purpose
    /// malloc/free -- see pmap::detail::TrieCore::mem_'s own comment for the
    /// measured cost this targets.
    ///
    /// synchronized_pool_resource, NOT unsynchronized: this was the ONE
    /// thing that looked safe here and wasn't. commit_mu_ (invariant 7)
    /// only serializes ALLOCATION (every insert()/erase() during apply) --
    /// it says nothing about DEALLOCATION, which happens whenever a Node's
    /// shared_ptr refcount hits zero, and that can be triggered by ANY
    /// thread: a reader dropping its Snapshot (and, with it, the last
    /// reference to some old Root's copy of by_type_) NEVER takes
    /// commit_mu_ at all -- that lock-free-reads guarantee is invariant 10,
    /// not optional. So a writer's allocate() (under commit_mu_) and a
    /// reader's deallocate() (under no lock whatsoever) can and do run
    /// concurrently against this SAME pool's internal freelist. Confirmed
    /// by TSan: a real data race between a Node destructor (triggered by
    /// by_type_[tag]'s old value being overwritten/dropped) and a
    /// concurrent allocate_shared<Node> on a different thread, in
    /// performance_tests.cpp's four-writer-threads stress test -- caught
    /// only once the ordering bug that had left the pool completely
    /// unused (see apply_create's own comment) was fixed and the pool
    /// started actually being exercised. unsynchronized_pool_resource is
    /// therefore unsafe here NO MATTER how tightly it's scoped (Model-
    /// owned or process-wide, it doesn't matter -- see persistent_map.h's
    /// TrieCore::mem_ comment, which used to claim otherwise). Paying for
    /// synchronization here is the price of NOT slowing down the
    /// lock-free read path with a matching lock -- the alternative would
    /// be routing every Root/Snapshot teardown through commit_mu_ too,
    /// which is a strictly worse trade.
    ///
    /// MUST still be declared before every member that could still
    /// reference a Node/Leaf allocated from it -- root_, by_type_,
    /// by_field_, by_cached_field_, by_cached_reference_ below -- because
    /// std::pmr::polymorphic_allocator stores only a raw memory_resource*
    /// inside the shared_ptr control block std::allocate_shared builds
    /// (see TrieCore::mem_), so this pool must outlive every Node/Leaf
    /// ever allocated from it, thread-safety aside. Members are destroyed
    /// in REVERSE declaration order, so declaring node_pool_ FIRST makes
    /// it the LAST thing ~Model() tears down -- by which point
    /// root_/by_type_/etc. (declared after, so destroyed first) have
    /// already released every Node/Leaf they held. This is exactly the
    /// precondition ~Model() already documents (every Snapshot/
    /// Transaction/View must already be gone), just extended one step
    /// further to this pool. Do not move this member, and do not add a
    /// new Node/Leaf-holding member ABOVE it.
    std::pmr::synchronized_pool_resource node_pool_;

    // ---- read/publish path -------------------------------------------------
    // root_ is atomic so snapshot() acquires the current version with a lock-free
    // load -- no shared mutex on the hot read path. Version bookkeeping (for the
    // reclamation watermark) still needs a short critical section, but it is
    // split onto its own small mutex (ver_mu_) so it never contends with a
    // commit's apply work or a reader's actual traversal.
    std::atomic<std::shared_ptr<const Root>> root_;

    /// Mints Transaction::id(). Deliberately its own plain atomic, NOT
    /// commit_mu_-protected: Transaction construction (begin()) must stay
    /// fully lock-free with respect to try_commit() (invariant 10), so this
    /// counter can't live behind the same lock version_/next_slot_/etc. do.
    /// Starts at 1 so 0 is free to mean "no transaction" wherever useful.
    std::atomic<std::uint64_t> next_txn_id_{1};

    /// Cumulative try_commit()/try_commit_without_undo() outcomes, one
    /// counter per CommitStatus, since Model construction -- never reset,
    /// never decremented. Plain atomics rather than commit_mu_-protected:
    /// record_commit_outcome() is called from try_commit_core() both
    /// inside AND outside the commit_mu_ critical section (the
    /// empty-transaction fast path returns Committed before commit_mu_ is
    /// ever taken), so a lock-free counter is the only shape that covers
    /// both call sites without extending the locked region just for
    /// bookkeeping. See Diagnostics' commits_* fields and
    /// record_commit_outcome().
    std::atomic<std::uint64_t> commits_succeeded_{0};
    std::atomic<std::uint64_t> commits_conflicted_{0};
    std::atomic<std::uint64_t> commits_vetoed_{0};
    std::atomic<std::uint64_t> commits_invalid_{0};
    std::atomic<std::uint64_t> commits_precommit_conflicted_{0};

    void record_commit_outcome(CommitStatus status) noexcept;

    mutable std::mutex ver_mu_;
    std::map<std::uint64_t, int>
        live_;  ///< live snapshot (incl. txn base) versions; min = watermark

    mutable std::mutex subs_mu_;                       ///< guards subs_ only; publish copies the
                                                      ///< list out so pushes run without it held
                                                      ///< (mutable: diagnostics() locks it from a
                                                      ///< const method, same as commit_mu_/ver_mu_)
    std::vector<std::shared_ptr<Subscription>> subs_;  ///< every live subscriber; shared_ptr so a
                                                       ///< subscriber outliving shutdown() is safe

    // ---- field lookup stats -------------------------------------------------
    // Entirely independent of everything above: not touched by try_commit(),
    // not part of published state, never read back into a decision the model
    // itself makes. See record_field_lookup()'s doc comment for the design
    // (a shared_mutex, held shared on the steady-state path) and why it
    // replaced an earlier thread_local-plus-flush design.

    // FieldLookupKey/FieldLookupKeyHash live at namespace scope, not here --
    // see FieldLookupKey's own doc comment for why.

    /// The atomic mirror of the public LookupCounts: one instance per
    /// looked-up field, held inside field_lookup_counts_. Split into atomics
    /// (rather than storing LookupCounts directly) is what lets the
    /// steady-state increment in record_field_lookup() take field_lookup_mu_
    /// only SHARED -- concurrent lookups bump their own counter without
    /// serializing against each other, only against a never-seen-before
    /// field's insert. LookupCounts itself stays a plain (non-atomic) value
    /// type because it's also the return type callers see (lookup_stats(),
    /// LookupDiagnostics::stats), copied out once, after the atomics are read.
    struct FieldLookupCounters {
        std::atomic<std::uint64_t> cached{0};
        std::atomic<std::uint64_t> uncached{0};
    };
    mutable std::shared_mutex field_lookup_mu_;  ///< guards field_lookup_counts_'s STRUCTURE
                                                 ///< only (inserting a field never seen before
                                                 ///< on this Model) -- the atomics inside an
                                                 ///< entry already present are updated with
                                                 ///< only a SHARED lock held, so every call
                                                 ///< after the first-ever one for a given field
                                                 ///< contends with inserts only, never with
                                                 ///< other readers.
    mutable std::unordered_map<FieldLookupKey, FieldLookupCounters, FieldLookupKeyHash>
        field_lookup_counts_;

    // ---- background reaper --------------------------------------------------
    // Retired objects are handed to a dedicated thread rather than freed inline
    // in try_commit(), so a large cascade never stalls a commit, and destructors
    // run off both the committing thread and any reader thread.
    std::thread reaper_;               ///< started by the ctor, joined by the dtor
    std::mutex reap_mu_;               ///< guards everything below except reap_backlog_
    std::condition_variable reap_cv_;  ///< wakes the reaper: work arrived, or stopping
    std::condition_variable reap_done_cv_;  ///< wakes wait_for_reclamation(): a pass finished
    std::deque<std::pair<std::uint64_t, const ObjectBase*>> reap_queue_;
    ///< ^ (version at which each object became invisible, object); freed once
    ///< the live watermark reaches that version. ALWAYS sorted ascending by
    ///< version: every batch enqueue_retired() appends carries version_+1 of
    ///< the commit that produced it, and commits are serialized (commit_mu_)
    ///< with version_ strictly increasing -- so freeable entries are always
    ///< exactly the front prefix. A deque (not vector), so reaper_loop() can
    ///< pop that prefix in O(1) per entry instead of re-scanning the whole
    ///< queue every pass -- see its own comment.
    std::atomic<std::size_t> reap_backlog_{0};  ///< reaper backlog, cheaply answer "how far behind is the background reaper right now?" 
                                                ///< — e.g. to monitor whether cascade-heavy churn is outpacing reclamation — without
                                                ///< paying reap_mu_ contention or reap_queue_.size()'s cost under load. It's kept as a
                                                ///< separate atomic specifically so that check doesn't need reap_mu_ at all
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
    // First-touch-this-attempt tracking for the four whole-map-handle
    // indexes' undo capture -- see log_by_type_once()'s doc comment. Same
    // "attempt-scoped, cleared alongside dirty_" lifetime as dirty_ itself.
    std::unordered_set<TypeTag> dirty_by_type_;
    std::unordered_set<const void*> dirty_by_field_;
    std::unordered_set<const void*> dirty_by_cached_field_;
    std::unordered_set<const void*> dirty_by_cached_reference_;
    // The writer's working copies of Root's three indexes -- same persistent
    // structures, so publishing them into a new Root is a cheap map copy.
    std::unordered_map<TypeTag, pmap::PersistentSet<Id, IdHash>> by_type_;
    std::unordered_map<const void*, pmap::PersistentMap<std::string, Id, pmap::StringHash>> by_field_;
    std::unordered_map<const void*,
                       pmap::PersistentMap<std::string, pmap::PersistentSet<Id, IdHash>,
                                          pmap::StringHash>>
        by_cached_field_;
    std::unordered_map<const void*,
                       pmap::PersistentMap<Id, pmap::PersistentSet<Id, IdHash>, IdHash>>
        by_cached_reference_;

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
    std::vector<UndoAction> pending_undo_;  ///< scratch: this attempt's inverse, parallel to changes_ --
                                            ///< cleared on rollback (rollback_apply), moved into
                                            ///< undo_list_ on publish (publish_now)
    std::vector<std::pair<std::uint64_t, const ObjectBase*>> retired_;
    ///< ^ this attempt's retirees, same shape as reap_queue_: handed to the
    ///< reaper on publish, drained back out by the undo log on rollback
    PreCommitFn pre_commit_;  ///< empty = no hook; swapped only under commit_mu_ (set_pre_commit)
    std::deque<ChangelogEntry>
        changelog_;  ///< for try_commit()'s conflict check; see prune_changelog

    /// Dense, slot-indexed conflict index: last_write_version_[slot] is the
    /// version() of the most recent commit that created/updated/removed/
    /// cascade-nulled that slot (0 if never touched), last_write_id_[slot]
    /// the Id (generation included) of that write. check_id_overlap() uses
    /// this instead of scanning changelog_, so its cost is O(this
    /// transaction's own write set) rather than O(every change committed
    /// by every writer since this transaction's base) -- the latter grows
    /// with contention itself (more concurrent writers => a transaction's
    /// base falls further behind by the time it reaches try_commit()),
    /// which is exactly backwards for a design whose whole point is many
    /// writer threads. Grown lazily in publish_now() (indices only ever
    /// need to reach the highest slot touched so far); never shrunk, since
    /// a recycled slot's old entry is harmless -- see check_id_overlap's
    /// version comparison, which only cares whether it exceeds a specific
    /// txn's base_version(). Writer-private and NOT part of Root: unlike
    /// spine_/by_type_/etc. it is never handed to a reader, so it costs
    /// nothing per-commit to "publish" (there is nothing to copy).
    std::vector<std::uint64_t> last_write_version_;
    std::vector<Id> last_write_id_;

    /// One entry per successful commit that had any undo data (see
    /// UndoEntry), oldest first. Unlike changelog_ (pruned by the
    /// reclamation watermark -- a LIFETIME concern), an entry here is
    /// pruned by CONFLICT: publish_now() drops any existing entry whose
    /// own `touched` set intersects the just-published commit's, on the
    /// theory that this is the broadest, simplest-to-state definition of
    /// "a later commit invalidated this undo" -- even a Remove action
    /// (idempotent-safe against most later changes) or a Recreate action
    /// (whose old id can never be touched again) gets pruned this way, a
    /// deliberately conservative choice over a narrower one that would
    /// only watch RestoreUpdate targets. Otherwise unbounded: nothing
    /// caps its size automatically, matching this project's existing "give
    /// the primitive, let the caller bound it" pattern (Subscription's
    /// queue depth) -- see clear_undo_list() and set_max_undo_list_size().
    ///
    /// A std::list, not a std::vector: undo_touch_index_ below stores
    /// iterators into this container across arbitrary later insertions and
    /// erasures elsewhere in it, which only a node-based container can
    /// promise stay valid for (a vector's iterators are invalidated by any
    /// insert/erase that isn't at the very end). size() is still O(1) (C++11
    /// guarantee), so the cap check in publish_now() is unaffected; nothing
    /// here ever needs random access, only forward iteration and erase-by-
    /// iterator, both of which a list gives in O(1).
    std::list<UndoEntry> undo_list_;

    /// Reverse index for the conflict-prune step in publish_now(): every id
    /// CURRENTLY touched by some entry still in undo_list_ maps to THAT
    /// entry's iterator. Lets a commit that touched ids {a, b, c} find
    /// exactly which retained entries conflict in O(|{a,b,c}|) expected
    /// time -- one hash lookup per id THIS commit touched -- instead of the
    /// O(|undo_list_|) scan-every-entry approach that used to run on every
    /// single commit regardless of how much (or how little) history was
    /// actually affected.
    ///
    /// At most ONE entry can be indexed under a given id at any moment: any
    /// commit that touches id X always prunes whichever entry currently
    /// touches X (via this very index) before it can possibly add a new
    /// entry of its own -- so a later entry claiming X necessarily replaces
    /// the only earlier one that could have claimed it, never coexists with
    /// it. That's what keeps this a plain Id -> iterator map rather than a
    /// multimap.
    ///
    /// Kept in exact lockstep with undo_list_ by every mutation site
    /// (publish_now()'s prune-and-append, take_undo(), clear_undo_list()) --
    /// see untrack_undo_entry(), the one place entries are ever removed from
    /// this index, so "an entry left undo_list_" and "its ids left this
    /// index" can never drift apart.
    std::unordered_map<Id, std::list<UndoEntry>::iterator, IdHash> undo_touch_index_;

    /// Removes every id in `e.touched` from undo_touch_index_, and unwinds
    /// `e.approx_bytes` from total_undo_bytes_ -- called immediately before
    /// `e` itself is erased from undo_list_ (publish_now, take_undo), never
    /// after: an id left indexed (or a byte count left counted) while
    /// pointing at an already-erased list node is a dangling iterator the
    /// next lookup would dereference, or a leaked count nothing will ever
    /// re-subtract. Doesn't touch undo_list_ itself; the caller does that
    /// part. clear_undo_list() and set_max_undo_(list_size|memory_bytes)(0)
    /// wipe undo_list_/undo_touch_index_/total_undo_bytes_ wholesale instead
    /// of calling this per-entry -- equivalent, cheaper when discarding
    /// everything at once.
    void untrack_undo_entry(const UndoEntry& e);

    /// Cap enforced by set_max_undo_list_size(); default (max()) means
    /// unbounded, preserving the historical behavior for anyone who never
    /// calls the setter. Read and enforced only in publish_now(), at the
    /// point a new entry would be appended -- see that setter's own
    /// comment for why this is lazy (checked on add) rather than applied
    /// retroactively when the cap changes.
    std::size_t max_undo_list_size_ = std::numeric_limits<std::size_t>::max();

    /// Cap enforced by set_max_undo_memory_bytes(), against total_undo_bytes_
    /// below -- same lazy, checked-on-add, oldest-first-eviction discipline
    /// as max_undo_list_size_, just budgeted by UndoEntry::approx_bytes
    /// instead of by count. Default (max()) means unbounded.
    std::size_t max_undo_bytes_ = std::numeric_limits<std::size_t>::max();

    /// Running sum of UndoEntry::approx_bytes over every entry currently in
    /// undo_list_ -- kept in lockstep with undo_list_ the same way
    /// undo_touch_index_ is (see untrack_undo_entry()), so
    /// set_max_undo_memory_bytes()'s cap check in publish_now() is an O(1)
    /// comparison instead of an O(|undo_list_|) re-sum on every commit.
    std::size_t total_undo_bytes_ = 0;

    PreTransactionsFn pre_transactions_;  ///< empty = no hook; swapped only under commit_mu_
                                          ///< (set_pre_transactions)
    bool in_pre_transactions_phase_ = false;  ///< guards run_pre_transaction(): true only
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

    PostCommitFn post_commit_;  ///< empty = no hook; swapped only under commit_mu_
                                ///< (set_post_commit). The MEMBER is commit_mu_-protected like
                                ///< every other hook here -- but try_commit() copies it out while
                                ///< still holding the lock and invokes that copy only after
                                ///< releasing it; see commit_main_locked() and PostCommitFn.

    // Undo log and the objects created this attempt (which rollback_apply()
    // must delete, since they were never published and nothing else owns them).
    // Scratch: cleared at the start of every try_commit() attempt.
    std::vector<UndoOp> undo_;
    std::vector<const ObjectBase*> txn_created_;
};

// ---------------------------------------------------------------------------
// Snapshot lookup functions -- defined here, not in the class body, because
// each needs Model::record_field_lookup(), and Model isn't a complete type
// until the closing brace above. Snapshot::record_field_lookup ITSELF is
// defined in model.cpp, not here: it also needs Snapshot::Lease complete,
// and Lease's own definition lives there, not in this header (see Lease's
// forward declaration).
// ---------------------------------------------------------------------------

template <auto Field>
std::vector<const member_class_t<decltype(Field)>*> Snapshot::find_by_scan_field(
    const member_value_t<decltype(Field)>& value) const {
    using ClassT = member_class_t<decltype(Field)>;
    std::vector<const ClassT*> out;
    record_field_lookup(typeid(ClassT), field_tag<Field>(), /*cached=*/false);
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

template <auto Field>
std::vector<const member_class_t<decltype(Field)>*> Snapshot::find_by_cached_field(
    const member_value_t<decltype(Field)>& value) const {
    using ClassT = member_class_t<decltype(Field)>;
    std::vector<const ClassT*> out;
    record_field_lookup(typeid(ClassT), field_tag<Field>(), /*cached=*/true);
    if (!root_) return out;
    auto it = root_->by_cached_field.find(field_tag<Field>());
    if (it == root_->by_cached_field.end()) return out;
    const pmap::PersistentSet<Id, IdHash>* bucket = it->second.get(to_field_key(value));
    if (!bucket) return out;
    bucket->for_each([&](Id id) {
        if (const ClassT* p = cast<ClassT>(find_raw(id))) out.push_back(p);
    });
    return out;
}

template <auto Field, class F>
void Snapshot::for_each_referrer(Ref<typename member_value_t<decltype(Field)>::target_type> target,
                                 F&& f) const {
    using ClassT = member_class_t<decltype(Field)>;
    record_field_lookup(typeid(ClassT), field_tag<Field>(), /*cached=*/false);
    for_each<ClassT>([&](const ClassT& o) {
        if ((o.*Field).raw() == target.raw()) f(o);
    });
}

template <auto Field>
std::vector<const member_class_t<decltype(Field)>*> Snapshot::find_cached_referrers(
    Ref<typename member_value_t<decltype(Field)>::target_type> target) const {
    using ClassT = member_class_t<decltype(Field)>;
    std::vector<const ClassT*> out;
    record_field_lookup(typeid(ClassT), field_tag<Field>(), /*cached=*/true);
    if (!root_) return out;
    auto it = root_->by_cached_reference.find(field_tag<Field>());
    if (it == root_->by_cached_reference.end()) return out;
    const pmap::PersistentSet<Id, IdHash>* bucket = it->second.get(target.raw());
    if (!bucket) return out;
    bucket->for_each([&](Id id) {
        if (const ClassT* p = cast<ClassT>(find_raw(id))) out.push_back(p);
    });
    return out;
}

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
///   PrecommitConflict a Model::run_pre_transaction() call inside the
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
    /// is only unique within the transaction that minted it). Use to_real()
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
    ///
    /// Named to_real(), not resolve(): "resolve" already means one thing in
    /// this header -- Snapshot::resolve() (an unchecked, never-null
    /// dereference to `const T&`/`const T*`, trusting the published
    /// invariant), which Transaction::peek()'s doc contrasts itself
    /// against. This is neither: no dereference happens here at all, just a
    /// local-id -> real-id translation (Ref<T> in, Ref<T> out) -- the same
    /// translation RefRemapper::translate() performs on the writer side,
    /// during apply, via local_remap's table (named translate(), not
    /// resolve(), for exactly this same reason). See RefRemapper's own
    /// comment.
    template <class T>
    Ref<T> to_real(Ref<T> local) const {
        if (!is_local(local.raw())) return local;
        auto it = local_remap.find(local.raw().index);
        return Ref<T>(it == local_remap.end() ? Id{} : it->second);
    }
    /// Opt<T> form. Null passes through as null.
    template <class T>
    Opt<T> to_real(Opt<T> local) const {
        if (!local || !is_local(local.raw())) return local;
        auto it = local_remap.find(local.raw().index);
        return Opt<T>(it == local_remap.end() ? Id{} : it->second);
    }
};

// ---------------------------------------------------------------------------
// BulkTransaction -- the builder half of Model::begin_bulk()/commit_bulk_without_undo()
// ---------------------------------------------------------------------------

/// The bulk-load builder: create() plus a narrow, local-only update() --
/// still no remove()/peek(), and no base to read against (there is nothing
/// meaningful to read -- commit_bulk_without_undo() wipes the model before installing
/// this). See Model::begin_bulk()'s section comment for the full contract,
/// including the exclusive-access precondition commit_bulk_without_undo() requires.
///
/// Building one touches no shared state, exactly like Transaction -- create()
/// just mints a local id (see is_local()) and stores the object locally.
/// Objects created here can reference each other freely, in any order,
/// through the Ref<T>/Opt<T> returned by create(), the same way same-
/// transaction local creates work on an ordinary Transaction; commit_bulk_without_undo()
/// remaps every one of those local ids to its real id in one pass. A forward
/// reference -- pointing at an object this batch hasn't created yet -- can be
/// fixed up after the fact with update() instead of predicting a future
/// local id by hand. Not copyable (owns unique_ptrs to not-yet-installed
/// objects); movable.
///
/// Deliberately NOT a subclass of Transaction (in either direction), despite
/// the create()-side resemblance. Transaction's local-id scheme tolerates
/// holes (a same-transaction create-then-remove cancels the create outright,
/// see Transaction::remove()); this type's create() uses objects_.size() as
/// the next local index precisely BECAUSE it never needs to leave a hole --
/// unifying the two would force one scheme onto the other. More importantly,
/// if this were a base of Transaction, Model::commit_bulk_without_undo(BulkTransaction&)
/// would silently accept a Transaction upcast and wipe/reload the whole
/// model from just its local_created_, discarding its base_, local_updated_,
/// and remove_intents_ without a warning. Keeping them unrelated types means
/// that mistake doesn't compile.
class BulkTransaction {
public:
    BulkTransaction(BulkTransaction&&) = default;
    BulkTransaction& operator=(BulkTransaction&&) = default;
    BulkTransaction(const BulkTransaction&) = delete;
    BulkTransaction& operator=(const BulkTransaction&) = delete;

    /// Takes ownership and returns a LOCAL id -- usable immediately as a
    /// Ref<T>/Opt<T> target for any other object created on this SAME
    /// BulkTransaction, before or after this call. Meaningless outside it;
    /// see CommitResult::to_real() for translating one into its real,
    /// post-commit form once commit_bulk_without_undo() succeeds.
    template <class T>
    Ref<T> create(std::unique_ptr<T> o) {
        static_assert(std::is_base_of_v<ObjectBase, T>, "T must derive from Object<T>");
        // objects_.size() doubles as the next local index: unlike
        // Transaction, there is no remove() to cancel an earlier entry and
        // leave a hole, so "how many objects exist so far" and "the next
        // free local index" are always the same number.
        const Id local_id{kLocalIdBit | static_cast<std::uint32_t>(objects_.size()), 1};
        o->id = local_id;
        objects_.push_back(std::move(o));
        return Ref<T>(local_id);
    }

    /// Mutable pointer to an object already create()'d on this SAME
    /// BulkTransaction -- lets a forward reference get fixed up once the
    /// object it points to exists, instead of predicting a future local id
    /// by hand. Returns the exact object create() already owns: no clone,
    /// no baseline to track, since there is no base() here to diff against
    /// (unlike Transaction::update()). Null if `r` isn't a local id minted
    /// by this same batch, or is out of range.
    template <class T>
    T* update(Ref<T> r) {
        return static_cast<T*>(update_raw(r.raw()));
    }
    template <class T>
    T* update(Opt<T> r) {
        return static_cast<T*>(update_raw(r.raw()));
    }

    /// Untyped counterpart of update<T> -- same reasoning as Transaction's
    /// create_raw/update_raw/remove_raw. objects_'s index IS the local id's
    /// low bits (see create()), so this is a direct bounds-checked lookup
    /// -- no clone-on-first-touch bookkeeping like Transaction::update_raw(),
    /// because every entry here is already a local, not-yet-installed
    /// object owned outright by this batch.
    ObjectBase* update_raw(Id id) {
        if (!is_local(id)) return nullptr;
        const std::uint32_t idx = id.index & ~kLocalIdBit;
        return idx < objects_.size() ? objects_[idx].get() : nullptr;
    }

    /// How many objects are pending. Diagnostic; commit_bulk_without_undo() doesn't need
    /// it, but a caller sanity-checking a large generated batch might.
    std::size_t size() const noexcept { return objects_.size(); }

private:
    friend class Model;

    /// Only Model::begin_bulk() constructs one.
    explicit BulkTransaction(Model* m) : model_(m) {}

    Model* model_ = nullptr;  ///< asserted against cross-model misuse in commit_bulk_without_undo()
    std::vector<std::unique_ptr<ObjectBase>> objects_;  ///< index == local id's low bits
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

    /// Unique for the lifetime of the Model that built this Transaction
    /// (minted from Model::next_txn_id_, a plain atomic counter -- NOT
    /// commit_mu_-protected, so begin() stays fully lock-free; see invariant
    /// 10). Rides along across the move that returns a Transaction from
    /// begin(), so it stays stable for the object's whole life. This is the
    /// correlation key for PreTransactionsFn/PreCommitFn/PostCommitFn: all
    /// three receive the SAME Transaction (by const reference) for one
    /// try_commit() attempt, so id() is how a hook (or a caller comparing
    /// its own logging) recognizes "these three calls are about the same
    /// attempt." A pre-transaction built with model.begin() inside
    /// PreTransactionsFn gets its OWN, different id -- there is no need for
    /// a separate "parent id": the hook already has both ids in scope
    /// (the main txn's, from its own parameter; the pre-transaction's, from
    /// the object it just built) without any extra API. If this transaction
    /// commits and produces undo data, this id is also copied onto
    /// UndoEntry::txn_id/UndoSummary::txn_id -- informational there too
    /// (take_undo() still keys on the commit's published version, not this).
    std::uint64_t id() const noexcept { return id_; }

    /// Caller-supplied label, set once at begin() and read-only from here on
    /// -- empty unless the caller passed one. Purely descriptive (never
    /// compared or dispatched on by the model itself); copied onto this
    /// commit's UndoEntry/UndoSummary (if any), so a later list_undo() call
    /// can show something more meaningful than a bare version number.
    const std::string& name() const noexcept { return name_; }

    /// Caller-supplied payload, set once at begin() and read-only from here
    /// on -- empty (std::any{}) unless the caller passed one. Opaque to the
    /// model; same copy-onto-UndoEntry/UndoSummary treatment as name().
    const std::any& data() const noexcept { return data_; }

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
        return Ref<T>(create_raw(std::move(o)));
    }

    /// Untyped counterpart of create<T> -- for a caller building a
    /// Transaction generically across many object types, without a
    /// per-type dispatch table (same reasoning as Model::peek_raw
    /// alongside peek_as<T>). Returns the bare local Id instead of a typed
    /// Ref<T>, since there is no T to name here. Used by Model::apply_undo
    /// to recreate a cascade-deleted object without knowing its type.
    Id create_raw(std::unique_ptr<ObjectBase> o) {
        const Id local_id{kLocalIdBit | next_local_id_++, 1};
        o->id = local_id;
        const TypeTag tag = o->tag();
        local_created_.push_back(std::move(o));
        pending_changes_.push_back({local_id, ChangeKind::Created, tag});
        return local_id;
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
        return static_cast<T*>(update_raw(r.raw()));
    }
    template <class T>
    T* update(Opt<T> r) {
        return static_cast<T*>(update_raw(r.raw()));
    }

    /// Untyped counterpart of update<T> -- same reasoning as create_raw.
    /// Used by Model::apply_undo, both for a just-recreated local object
    /// (remapping its own ref fields) and for restoring a still-real
    /// object's pre-image (RestoreUpdate actions). Also update<T>'s own
    /// body: local id -> the already-owned local_created_ entry (create()-
    /// then-update() in the same txn, no new clone needed); real id already
    /// touched this txn -> the existing clone; otherwise clone base()'s
    /// value into local_updated_, remember the pre-edit baseline
    /// (peek_before()) and record the pending change. Null for a masked
    /// (remove()-intended) or nonexistent id. The clone happens AT MOST
    /// ONCE per id per transaction -- repeated calls on the same id return
    /// the SAME clone, so writes accumulate.
    ObjectBase* update_raw(Id id) {
        if (is_local(id)) {
            const std::uint32_t idx = id.index & ~kLocalIdBit;
            // Masked like peek_raw: a deferred-removed local can't be
            // written to any more than a remove-intended real id can.
            if (std::find(local_remove_intents_.begin(), local_remove_intents_.end(), idx) !=
                local_remove_intents_.end())
                return nullptr;
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
    /// narrow benefit). The removed object ITSELF is masked immediately,
    /// real or local.
    ///
    /// If `r` is a local id (created earlier in this same transaction,
    /// never committed) and nothing else pending references it, the create
    /// is simply cancelled outright: nothing was ever published, so there
    /// is nothing to resolve at commit time. If another pending object DOES
    /// reference it, the remove defers instead: the create still installs
    /// during apply and is then removed by the same commit-time cascade BFS
    /// a committed id gets -- so an Opt<> referrer is nulled and a Ref<>
    /// referrer cascades, identical semantics whether the target ever
    /// committed or not. (The referenced-or-not decision is taken at THIS
    /// call: a ref added to an already-cancelled local id afterward is a
    /// build bug and still rejects as Invalid at commit.)
    template <class T>
    void remove(Ref<T> r) {
        remove_raw(r.raw());
    }
    template <class T>
    void remove(Opt<T> r) {
        remove_raw(r.raw());
    }

    /// Untyped counterpart of remove<T> -- same reasoning as create_raw.
    /// (Model::remove_raw already exists with unrelated semantics -- the
    /// cascade BFS, on a different class -- no collision.) Also remove<T>'s
    /// own body: for a real id, just records the intent (remove_intents_);
    /// no cascade work happens here, see the class-level remove() doc for
    /// why. For a LOCAL id, the choice is made here, once, based on what
    /// the transaction holds RIGHT NOW:
    ///   - unreferenced: cancel the create outright -- drop the owned
    ///     object and scrub it from pending_changes() -- since nothing was
    ///     ever published for anything to reference;
    ///   - referenced by another pending object: DEFER (record the local
    ///     index in local_remove_intents_) -- the create still installs at
    ///     apply time and is then fed, via its freshly minted real id, to
    ///     the exact same cascade BFS a committed remove gets. Its Created
    ///     entry stays in pending_changes(), because it genuinely will be
    ///     created (and then deleted) by the commit.
    void remove_raw(Id id) {
        if (is_local(id)) {
            const std::uint32_t idx = id.index & ~kLocalIdBit;
            if (idx >= local_created_.size() || !local_created_[idx])
                return;  // never created here, or already cancelled: nothing to do
            if (std::find(local_remove_intents_.begin(), local_remove_intents_.end(), idx) !=
                local_remove_intents_.end())
                return;  // already deferred: remove() is idempotent
            if (locally_referenced(id)) {
                local_remove_intents_.push_back(idx);
                return;
            }
            local_created_[idx].reset();  // cancel locally
            auto rm = [&](const Change& c) { return c.id == id; };
            pending_changes_.erase(
                std::remove_if(pending_changes_.begin(), pending_changes_.end(), rm),
                pending_changes_.end());
            return;
        }
        remove_intents_.insert(id);
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
    ///
    /// Deliberately NOT named/shaped like Snapshot::resolve(), even for a
    /// non-nullable Ref<T>: resolve() returns `const T&`, unchecked, and
    /// trusts the writer's invariant that a Ref inside a PUBLISHED,
    /// validated snapshot can never dangle. Nothing here has been through
    /// validate() yet -- a transaction being built can be temporarily
    /// inconsistent (a Ref pointing at something not yet created, or since
    /// locally removed) right up until try_commit() -- so peek() returns
    /// `const T*`, checked, and can legitimately be null even for a field
    /// that will end up non-nullable once committed. Don't assume peek()'s
    /// null-ness tells you anything about how the FIELD is declared; it
    /// only tells you what this transaction can currently see.
    template <class T>
    const T* peek(Ref<T> r) const {
        const ObjectBase* o = peek_raw(r.raw());
        return (o && o->tag() == type_tag<T>()) ? static_cast<const T*>(o) : nullptr;
    }
    template <class T>
    const T* peek(Opt<T> r) const {
        const ObjectBase* o = peek_raw(r.raw());
        return (o && o->tag() == type_tag<T>()) ? static_cast<const T*>(o) : nullptr;
    }

    /// Untyped counterpart of peek<T>/peek_as<T> -- same reasoning as
    /// create_raw/update_raw/remove_raw: resolves this transaction's
    /// current local view of `id` -- this transaction's own pending edit
    /// if there is one, else base()'s committed value, else null. Masked
    /// (null) by a pending remove() intent regardless of what base() would
    /// say. No tag check here -- that's what makes it reusable for a
    /// caller that doesn't know (or wants to check itself) the concrete
    /// type; peek<T>/peek_as<T> add the checked cast on top.
    const ObjectBase* peek_raw(Id id) const {
        if (is_local(id)) {
            const std::uint32_t idx = id.index & ~kLocalIdBit;
            // A deferred local remove masks its target exactly like a real
            // remove intent does below -- "removed as far as this
            // transaction can tell", even though the create still installs
            // (and is then removed) at apply time.
            if (std::find(local_remove_intents_.begin(), local_remove_intents_.end(), idx) !=
                local_remove_intents_.end())
                return nullptr;
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
        return base_.find_raw(id);
    }

    /// Safe typed read from an untyped Id -- e.g. a Change::id out of
    /// pending_changes(). Checks the object's actual type before casting
    /// (same as Snapshot::find<T>); null if dead/masked, or a type mismatch.
    template <class T>
    const T* peek_as(Id id) const {
        const ObjectBase* o = peek_raw(id);
        return (o && o->tag() == type_tag<T>()) ? static_cast<const T*>(o) : nullptr;
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

    /// Estimate this transaction's full effect, with no lock taken and
    /// nothing mutated: pending_changes() (creates/updates -- already known
    /// exactly, included verbatim) PLUS an estimated cascade resolution of
    /// remove_intents(). For each pending remove, a read-only walk of
    /// base() (see Snapshot::for_each_referrer_any) finds every current
    /// referrer and estimates a cascaded Deleted (non-nullable) or a
    /// null'd Updated (nullable) -- the same resolution remove_raw() does
    /// for real at apply time, just computed here against base() instead
    /// of the writer's referrers_ (commit_mu_-protected, unreachable from
    /// Transaction-building code -- invariant 10), and so at O(total live
    /// objects) per pending remove rather than referrers_'s near-O(1)
    /// lookup. Meant for occasional, interactive use (e.g. "this will
    /// affect N other objects, are you sure?") -- not a loop, and entirely
    /// independent of try_commit(): call it any number of times, or never,
    /// on a Transaction you may or may not go on to commit.
    ///
    /// This is an ESTIMATE, not a guarantee of what a later try_commit()
    /// call will actually do -- shaped the same as CommitResult::changes
    /// for easy comparison, but:
    ///   - it says nothing about ACCEPTANCE, only CONTENT: it doesn't (and
    ///     from here, cannot) run check_id_overlap(), so it cannot tell you
    ///     whether a real commit would be accepted as Conflict-free;
    ///   - it's computed against base(), which can go stale the instant
    ///     another writer commits something that would change the
    ///     cascade's shape;
    ///   - DEFERRED LOCAL removes (a remove() of a still-referenced local
    ///     create, see remove_raw) are not cascade-estimated at all: their
    ///     victim still appears as Created (accurate -- it will be), but
    ///     the same-commit Deleted for it, and any fan-out into other
    ///     pending objects, only shows up in CommitResult::changes.
    std::vector<Change> estimate_changes_with_cascades() const;

private:
    friend class Model;

    /// Only Model::begin() (and Snapshot::begin(), through it) constructs
    /// one -- a Transaction is meaningless without a Model to commit to and
    /// a pinned base to read through. Mints this Transaction's id() here,
    /// via m->next_txn_id_ (a plain atomic -- no lock taken), so id() is
    /// stable and unique from construction, before try_commit() is ever
    /// called.
    Transaction(Model* m, Snapshot base, std::string name = "", std::any data = {})
        : model_(m),
          base_(std::move(base)),
          id_(m->next_txn_id_.fetch_add(1, std::memory_order_relaxed)),
          name_(std::move(name)),
          data_(std::move(data)) {}

    /// Does any OTHER pending object (a live local create, or an update
    /// clone) hold a Ref<>/Opt<> whose target is `local`? Purely
    /// transaction-local -- walks this transaction's own overlay via
    /// each_ref(), touching no shared state (invariant 10 intact). The
    /// victim's own fields are excluded: a self-loop dies with its owner.
    bool locally_referenced(Id local) const {
        bool found = false;
        auto scan = [&](const ObjectBase* o) {
            if (!o || found || o->id == local) return;
            o->each_ref([&](const void*, const char*, Id target, bool) {
                if (target == local) found = true;
            });
        };
        for (const auto& up : local_created_) scan(up.get());
        for (const auto& [slot, up] : local_updated_) {
            (void)slot;
            scan(up.get());
        }
        return found;
    }

    Model* model_ = nullptr;  ///< where try_commit() applies this; asserted against
                              ///< cross-model misuse in try_commit()
    Snapshot base_;  ///< pins base_version_ in Model::live_, same as any reader's snapshot
    std::uint64_t id_ = 0;  ///< see id(); minted once, in the constructor, from
                            ///< model_'s next_txn_id_ -- moves along with the rest of this
                            ///< object via the defaulted move ctor/assignment
    std::string name_;  ///< see name()
    std::any data_;     ///< see data()

    std::vector<std::unique_ptr<ObjectBase>>
        local_created_;  ///< index == local id's low bits;
                         ///< null entry == cancelled (see remove_raw)
    std::unordered_map<std::uint32_t, std::unique_ptr<ObjectBase>>
        local_updated_;  ///< keyed by real slot index
    std::unordered_map<std::uint32_t, const ObjectBase*>
        update_baseline_;                            ///< points into base_'s Root,
                                                     ///< kept alive by base_ itself
    std::unordered_set<Id, IdHash> remove_intents_;  ///< real ids only -- see remove_raw
    std::vector<std::uint32_t> local_remove_intents_;  ///< DEFERRED local removes (bare local
                                                       ///< indexes): creates that install at apply
                                                       ///< time and are then cascade-removed, see
                                                       ///< remove_raw. Usually empty; linear
                                                       ///< std::find is fine at this size.
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
