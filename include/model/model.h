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

#include "model/persistent_map.h"

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
#include <stdexcept>
#include <string>
#include <thread>
#include <type_traits>
#include <typeinfo>
#include <unordered_map>
#include <unordered_set>
#include <vector>

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
struct Id {
    std::uint32_t index = 0;
    std::uint32_t gen = 0;  ///< 0 means null

    explicit operator bool() const noexcept { return gen != 0; }
    friend bool operator==(Id, Id) noexcept = default;
};

struct IdHash {
    std::size_t operator()(Id id) const noexcept {
        return (static_cast<std::size_t>(id.gen) << 32) | id.index;
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

inline bool is_local(Id id) noexcept { return (id.index & kLocalIdBit) != 0; }

// ---------------------------------------------------------------------------
// Type tags -- a cheap, RTTI-free checked downcast
// ---------------------------------------------------------------------------

using TypeTag = const void*;

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
    using target_type = T;

    Ref() = default;  ///< null until assigned; validate() rejects it at create time
    explicit Ref(Id id) noexcept : id_(id) {}

    Id raw() const noexcept { return id_; }
    explicit operator bool() const noexcept { return static_cast<bool>(id_); }

    friend bool operator==(Ref, Ref) noexcept = default;

private:
    Id id_{};
};

/// A nullable reference to a T. Deleting the target does NOT delete the holder --
/// the field is nulled instead.
template <class T>
class Opt {
public:
    using target_type = T;

    Opt() = default;
    explicit Opt(Id id) noexcept : id_(id) {}

    /// A non-null Ref is always a valid Opt. Not the reverse.
    Opt(Ref<T> r) noexcept : id_(r.raw()) {}

    void reset() noexcept { id_ = Id{}; }

    Id raw() const noexcept { return id_; }
    explicit operator bool() const noexcept { return static_cast<bool>(id_); }

    friend bool operator==(Opt, Opt) noexcept = default;

private:
    Id id_{};
};

// ---------------------------------------------------------------------------
// Object model
// ---------------------------------------------------------------------------

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
/// Throws Model::IntegrityError if a local id has no entry in the table --
/// which means it pointed at a local object that was never created, or one
/// created-then-removed within the same transaction (see
/// Transaction::remove(), the create-then-uncreate case): a genuine
/// transaction-building bug, not a concurrency conflict, so it is NOT
/// classified as a Conflict by try_commit() (bad_target is left null, since
/// there is no real Id to check against the transaction's base -- see
/// Model::validate's use of IntegrityError::bad_target for the contrast).
struct RefRemapper {
    const std::unordered_map<std::uint32_t, Id>& table;  // local Id::index -> real Id

    Id resolve(Id id) const;  // defined after Model::IntegrityError below

    template <class T>
    void operator()(const void*, const char*, Ref<T>& r) const {
        if (is_local(r.raw())) r = Ref<T>(resolve(r.raw()));
    }
    template <class T>
    void operator()(const void*, const char*, Opt<T>& r) const {
        if (r.raw() && is_local(r.raw())) r = Opt<T>(resolve(r.raw()));
    }
};

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
template <class D, class = void>
struct has_define_references : std::false_type {};
template <class D>
struct has_define_references<
    D, std::void_t<decltype(D::define_references(std::declval<const D&>(), std::declval<const RefReader&>()))>>
    : std::true_type {};

template <class D, class = void>
struct has_define_keys : std::false_type {};
template <class D>
struct has_define_keys<
    D, std::void_t<decltype(D::define_keys(std::declval<const D&>(), std::declval<const FieldKeyReader&>()))>>
    : std::true_type {};

/// Demangles a typeid name for use as a diagnostic label (Object<Derived>::type()).
/// Itanium ABI (GCC/Clang) only; passed through unchanged elsewhere (already
/// readable on MSVC). Defined in model.cpp so <cxxabi.h> doesn't leak into every
/// translation unit that includes this header.
std::string demangle_type_name(const std::type_info& ti);
}  // namespace detail

class ObjectBase {
public:
    Id id;

    virtual ~ObjectBase() = default;

    virtual ObjectBase* clone() const = 0;
    virtual TypeTag tag() const noexcept = 0;

    /// Diagnostic label only (IntegrityError messages) -- never compared or
    /// used for dispatch; tag()/type_tag<T>() is what the model actually
    /// checks types against. Object<Derived> derives this from typeid()
    /// automatically; there is nothing to override.
    virtual const char* type() const = 0;

    virtual void each_ref(const RefFn&) const {}
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
///         v.key<&Order::computed_key>(s.computed_key());  // find_by_key<&Order::computed_key>("ord:O1")
///     }
///
/// Each indexed field is assumed unique within its type; a duplicate value
/// silently overwrites the earlier entry. For "give me every match, not just
/// one," use the slow Snapshot::find_all() predicate scan instead.
template <class Derived>
class Object : public ObjectBase {
public:
    ObjectBase* clone() const override {
        return new Derived(static_cast<const Derived&>(*this));
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

    void each_field_key(const FieldKeyFn& fn) const override {
        if constexpr (detail::has_define_keys<Derived>::value)
            Derived::define_keys(static_cast<const Derived&>(*this), FieldKeyReader{fn});
    }
};

// ---------------------------------------------------------------------------
// Storage
// ---------------------------------------------------------------------------

inline constexpr std::uint32_t kChunkBits = 8;
inline constexpr std::uint32_t kChunkSize = 1u << kChunkBits;  // 256 slots
inline constexpr std::uint32_t kChunkMask = kChunkSize - 1;

/// The last usable generation. A slot that reaches this is retired rather than
/// reused, because the next bump would wrap to 0 -- the null id -- and a stale
/// handle would alias the new object. See Model::alloc_slot.
inline constexpr std::uint32_t kGenMax = 0xFFFFFFFFu;

/// Copy-on-write leaf. Raw pointers on purpose: copying a chunk must be a memcpy
/// with zero atomics. shared_ptr here would mean kChunkSize atomic RMWs per
/// dirtied chunk, which is what kills this design at scale.
struct Chunk {
    const ObjectBase* obj[kChunkSize] = {};
    std::uint32_t gen[kChunkSize] = {};
};

struct Root {
    std::uint64_t version = 0;
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
    /// the only lookup-by-value index in the model; there is no separate
    /// mandatory "primary key" index.
    std::unordered_map<const void*, pmap::PersistentMap<Id>> by_field;
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
    Snapshot() = default;

    std::uint64_t version() const noexcept { return root_ ? root_->version : 0; }
    explicit operator bool() const noexcept { return root_ != nullptr; }

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
    /// value across two objects means the later write wins -- use find_all()
    /// for "every match." Null if the key is absent, or the field wasn't
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
    /// not a runtime static_assert.
    template <auto Field, class F>
    void for_each_referrer(Ref<typename member_value_t<decltype(Field)>::target_type> target, F&& f) const {
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
    void for_each_referrer_view(Ref<typename member_value_t<decltype(Field)>::target_type> target, F&& f) const;

    /// View-returning form of find_referrers.
    template <auto Field>
    std::vector<View<member_class_t<decltype(Field)>>> find_referrers_view(
        Ref<typename member_value_t<decltype(Field)>::target_type> target) const;

    // ---- views ------------------------------------------------------------
    //
    // A View<T> is an object bound to the snapshot it came from. Traversal
    // through a view always uses *that* snapshot, so you cannot accidentally
    // resolve a v40 object's reference against a v44 root -- a mistake the raw
    // resolve() API happily compiles. See View below.

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

    /// Untyped escape hatch -- for change events, which are heterogeneous. The
    /// key is only unique within `field`'s own keyspace (see field_tag), so
    /// the field must be supplied; there is no single global keyspace to
    /// search without it.
    const ObjectBase* find_raw(Id id) const noexcept;
    const ObjectBase* find_by_key_raw(const void* field, const std::string& key) const;

private:
    friend class Model;
    friend class Transaction;
    struct Lease;

    /// One virtual call plus a pointer compare. No RTTI, no dynamic_cast.
    template <class T>
    static const T* cast(const ObjectBase* o) noexcept {
        if (!o || o->tag() != type_tag<T>()) return nullptr;
        return static_cast<const T*>(o);
    }

    std::shared_ptr<const Root> root_;
    std::shared_ptr<Lease> lease_;
};

// ---------------------------------------------------------------------------
// Change events
// ---------------------------------------------------------------------------

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
    Id id;
    ChangeKind kind;
    TypeTag tag;
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
/// so far and returns CommitStatus::Vetoed. Throwing instead propagates to
/// the caller of try_commit(), leaving the Transaction's local state
/// untouched (nothing shared was published) -- catch it, fix up, and retry.
///
/// Read-only by convention: the hook sees the resolved changeset and can
/// peek_as<T> any of it via the Model reference, but must not call
/// try_commit() itself or otherwise begin a new transaction -- this runs
/// inside an existing try_commit(), which already holds the commit lock.
using PreCommitFn = std::function<bool(Model&, const std::vector<Change>&)>;

struct Update {
    Snapshot snapshot;
    std::vector<Change> changes;
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
    explicit Subscription(std::size_t depth) : cap_(depth) {}

    bool wait(Update& out);       ///< blocks; false once the model shuts down
    bool try_drain(Update& out);  ///< non-blocking

private:
    friend class Model;

    void push(Update u);
    void collapse(Update tail);
    void close();

    std::mutex m_;
    std::condition_variable cv_;
    std::deque<Update> q_;
    std::size_t cap_;
    bool closed_ = false;
};

// ---------------------------------------------------------------------------
// Model
// ---------------------------------------------------------------------------

class Model {
public:
    Model();
    ~Model();

    Model(const Model&) = delete;
    Model& operator=(const Model&) = delete;

    // ---- read side (any thread) ---------------------------------------------

    Snapshot snapshot();  ///< O(1)
    std::shared_ptr<Subscription> subscribe(std::size_t queue_depth = 8);
    void shutdown();      ///< wakes blocked subscribers so their threads can exit

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
    /// A conflicting or vetoed attempt leaves `txn` fully consumed either way
    /// (its local overlay has already been moved from during apply) --
    /// discard it and begin() a fresh Transaction to retry.
    CommitResult try_commit(Transaction& txn);

    /// Install (or clear, with {}) the pre-commit hook. See PreCommitFn.
    void set_pre_commit(PreCommitFn fn) { pre_commit_ = std::move(fn); }

    /// Thrown by try_commit()'s apply phase when a Transaction's own creates
    /// or updates would violate referential integrity. Most such failures are
    /// classified as a Conflict by try_commit() (see bad_target) rather than
    /// escaping as this exception -- it only ever reaches a caller for a
    /// target that never existed even at the transaction's own base, which is
    /// a genuine transaction-building bug, not contention.
    struct IntegrityError : std::runtime_error {
        IntegrityError(const std::string& msg, Id bad_target = Id{})
            : std::runtime_error(msg), bad_target(bad_target) {}
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

    const ObjectBase* peek(Id id) const;
    Chunk* cow(std::uint32_t chunk_index);
    std::uint32_t alloc_slot();
    void validate(const ObjectBase* o) const;  // throws IntegrityError
    void add_out_refs(const ObjectBase* o);
    void drop_out_refs(const ObjectBase* o);
    void reconcile_referrer_edges(const ObjectBase* before, const ObjectBase* after);
    void add_field_keys(const ObjectBase* o);
    void drop_field_keys(const ObjectBase* o);
    void reconcile_field_keys(const ObjectBase* before, const ObjectBase* after);
    void retire(const ObjectBase* o);
    void release_version(std::uint64_t v);
    void reaper_loop();          ///< body of the background reaper thread
    void enqueue_retired(std::vector<std::pair<std::uint64_t, const ObjectBase*>> batch);

    // Undo log. Every mutation of writer-private state during a try_commit()
    // attempt pushes its inverse here; rollback_apply() runs the inverses in
    // reverse if that attempt fails; a successful try_commit() clears the log.
    // Kept as closures so each write op records its own undo inline, which is
    // far harder to get out of sync than a parallel variant type. The log is
    // proportional to the changes made, never to model size.
    void log(std::function<void()> undo) { undo_.push_back(std::move(undo)); }
    void set_slot(std::uint32_t slot, const ObjectBase* obj, std::uint32_t gen);

    // ---- try_commit() internals (ALL require commit_mu_ already held) ------
    void apply_create(std::unique_ptr<ObjectBase> o, std::unordered_map<std::uint32_t, Id>& remap);
    void apply_update(std::unique_ptr<ObjectBase> clone, std::unordered_map<std::uint32_t, Id>& remap);
    std::vector<Id> remove_raw(Id id);  // cascade BFS, called from try_commit()'s apply phase
    // remove_raw's helper for a NULLABLE referrer: clone + install, so the
    // caller can null_ref() the field that pointed at the victim, then
    // reconcile immediately. See the .cpp for why reconciliation must happen
    // in the caller, after null_ref() -- not in here.
    ObjectBase* clone_for_cascade_null(Id id);
    std::vector<Id> check_id_overlap(const Transaction& txn) const;
    void prune_changelog();
    void rollback_apply();  // unwinds one failed try_commit() apply attempt

    // ---- read/publish path -------------------------------------------------
    // root_ is atomic so snapshot() acquires the current version with a lock-free
    // load -- no shared mutex on the hot read path. Version bookkeeping (for the
    // reclamation watermark) still needs a short critical section, but it is
    // split onto its own small mutex (ver_mu_) so it never contends with a
    // commit's apply work or a reader's actual traversal.
    std::atomic<std::shared_ptr<const Root>> root_;

    mutable std::mutex ver_mu_;
    std::map<std::uint64_t, int> live_;  ///< live snapshot (incl. txn base) versions; min = watermark

    std::mutex subs_mu_;
    std::vector<std::shared_ptr<Subscription>> subs_;

    // ---- background reaper --------------------------------------------------
    // Retired objects are handed to a dedicated thread rather than freed inline
    // in try_commit(), so a large cascade never stalls a commit, and destructors
    // run off both the committing thread and any reader thread.
    std::thread reaper_;
    std::mutex reap_mu_;
    std::condition_variable reap_cv_;
    std::condition_variable reap_done_cv_;
    std::vector<std::pair<std::uint64_t, const ObjectBase*>> reap_queue_;
    std::atomic<std::size_t> retired_pending_{0};  ///< reaper backlog, for observability
    std::uint64_t reap_done_round_ = 0;            ///< bumped after each reap pass
    bool dirty_reap_ = false;                      ///< a reap pass is due
    bool reaper_stop_ = false;

    // ---- commit-lock-protected state ----------------------------------------
    // Touched ONLY by whichever thread currently holds commit_mu_, only from
    // inside try_commit(). This is exactly the old single-writer-thread state
    // (referrers_, by_type_, ...), unchanged in shape -- it's now protected by
    // an actual mutex instead of "single-threaded by convention," which is
    // strictly safer, not a rewrite. See CLAUDE.md invariant 7 and the lock
    // order rule (commit_mu_ -> ver_mu_ -> reap_mu_, never reversed).
    mutable std::mutex commit_mu_;
    std::uint64_t version_ = 0;
    std::vector<std::shared_ptr<const Chunk>> spine_;
    std::unordered_set<std::uint32_t> dirty_;
    std::unordered_map<TypeTag, pmap::PersistentMap<Id>> by_type_;
    std::unordered_map<const void*, pmap::PersistentMap<Id>> by_field_;
    std::unordered_map<std::uint32_t, std::vector<RefEdge>> referrers_;
    std::vector<std::uint32_t> free_slots_;
    std::uint32_t next_slot_ = 0;
    std::size_t exhausted_slots_ = 0;
    std::vector<Change> changes_;  ///< scratch: this attempt's resolved changeset
    std::vector<std::pair<std::uint64_t, const ObjectBase*>> retired_;
    PreCommitFn pre_commit_;
    std::deque<ChangelogEntry> changelog_;  ///< for try_commit()'s conflict check; see prune_changelog

    // Undo log and the objects created this attempt (which rollback_apply()
    // must delete, since they were never published and nothing else owns them).
    // Scratch: cleared at the start of every try_commit() attempt.
    std::vector<std::function<void()>> undo_;
    std::vector<const ObjectBase*> txn_created_;
};

// ---------------------------------------------------------------------------
// Multi-writer commit results
// ---------------------------------------------------------------------------

enum class CommitStatus { Committed, Conflict, Vetoed };
enum class ConflictReason { IdSetOverlap, RefIntegrity };

struct ConflictInfo {
    ConflictReason reason;
    std::vector<Id> ids;  ///< the specific id(s) that conflicted
};

struct CommitResult {
    CommitStatus status;
    Snapshot snapshot;                     ///< valid only if status == Committed
    std::vector<Change> changes;           ///< FULL resolved changeset, incl. cascade deletes;
                                            ///< empty unless status == Committed
    std::optional<ConflictInfo> conflict;  ///< set only if status == Conflict

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

    /// Translates a Ref<T> obtained from Transaction::create() BEFORE this
    /// try_commit() call into its real, post-commit handle. A ref that was
    /// never local (read from base(), or returned by Transaction::update())
    /// passes through unchanged. Only meaningful when status == Committed --
    /// a local id from a Conflict/Vetoed attempt was never installed
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

inline Id RefRemapper::resolve(Id id) const {
    auto it = table.find(id.index);
    if (it == table.end())
        throw Model::IntegrityError(
            "Ref<>/Opt<> points at a local id that was never created, or was removed "
            "within the same transaction before it was ever committed");
    return it->second;
}

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
    Transaction(Transaction&&) = default;
    Transaction& operator=(Transaction&&) = default;
    Transaction(const Transaction&) = delete;
    Transaction& operator=(const Transaction&) = delete;

    /// The committed version this transaction is building on top of.
    const Snapshot& base() const noexcept { return base_; }
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
        if (it == update_baseline_.end() || it->second->id != id || it->second->tag() != type_tag<T>())
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

    Transaction(Model* m, Snapshot base) : model_(m), base_(std::move(base)) {}

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

    void remove_impl(Id id) {
        if (is_local(id)) {
            const std::uint32_t idx = id.index & ~kLocalIdBit;
            if (idx < local_created_.size()) local_created_[idx].reset();  // cancel locally
            auto rm = [&](const Change& c) { return c.id == id; };
            pending_changes_.erase(std::remove_if(pending_changes_.begin(), pending_changes_.end(), rm),
                                   pending_changes_.end());
            return;
        }
        remove_intents_.insert(id);
    }

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

    Model* model_ = nullptr;
    Snapshot base_;  ///< pins base_version_ in Model::live_, same as any reader's snapshot

    std::vector<std::unique_ptr<ObjectBase>> local_created_;  ///< index == local id's low bits;
                                                               ///< null entry == cancelled (see remove_impl)
    std::unordered_map<std::uint32_t, std::unique_ptr<ObjectBase>> local_updated_;  ///< keyed by real slot index
    std::unordered_map<std::uint32_t, const ObjectBase*> update_baseline_;  ///< points into base_'s Root,
                                                                              ///< kept alive by base_ itself
    std::unordered_set<Id, IdHash> remove_intents_;  ///< real ids only -- see remove_impl
    std::uint32_t next_local_id_ = 0;
    std::vector<Change> pending_changes_;
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
    View(const Snapshot& s, const T& obj) noexcept : s_(&s), o_(&obj) {}
    View(Snapshot&&, const T&) = delete;  // never bind to a temporary snapshot

    const T& operator*() const noexcept { return *o_; }
    const T* operator->() const noexcept { return o_; }

    /// Follow a non-nullable field. Always yields a view.
    template <class U>
    View<U> operator[](Ref<U> T::*field) const noexcept {
        return View<U>(*s_, s_->resolve(o_->*field));
    }

    /// Follow a nullable field. Empty if the target was cascaded away.
    template <class U>
    std::optional<View<U>> operator[](Opt<U> T::*field) const noexcept {
        if (const U* p = s_->resolve(o_->*field)) return View<U>(*s_, *p);
        return std::nullopt;
    }

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
    const Snapshot* s_;
    const T* o_;
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
void Snapshot::for_each_referrer_view(Ref<typename member_value_t<decltype(Field)>::target_type> target,
                                      F&& f) const {
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

}  // namespace model
