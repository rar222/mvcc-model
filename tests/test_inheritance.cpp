#include <cstdint>
#include <string>

#include "model/model.h"
#include "test_harness.h"
#include "test_helpers.h"

using namespace model;

// Model types are ordinary C++ classes: `class Foo final : public
// model::Object<Foo>` is the ONLY inheritance the machinery itself requires
// (see model.h's Object<Derived> doc comment). Nothing stops a user type
// from ALSO inheriting from its own, non-model base classes -- for shared
// fields/behavior across several model types -- as long as the final type
// stays copyable/copy-assignable (clone()/assign_from() need that) and
// derives from model::Object<Derived> somewhere in its bases so the
// ObjectBase*-returning lookup APIs (find_by_key, find_referrers, ...) can
// legally downcast to it.
//
// This file exercises that: two plain (non-model) mixins, Timestamped and
// Labeled, both virtually inheriting a common Entity base -- the textbook
// diamond -- and two model types, Asset and Gizmo, that each inherit from
// BOTH mixins (so from Entity twice-over, collapsed to one shared subobject
// by virtual inheritance) plus model::Object<Derived>. That's two unrelated
// inheritance graphs meeting in one class: the diamond among Entity/
// Timestamped/Labeled, and the ordinary Object<Derived> CRTP base -- proof
// the model doesn't care what else Derived derives from.
//
// One real constraint falls out of this and is deliberately respected
// below, not "fixed": a field used with the model's `<&Type::field>`
// template APIs (define_keys/define_references/find_by_key/find_referrers/
// ...) must be declared DIRECTLY on the concrete Object<Derived> type, not
// inherited from a plain mixin. Those APIs downcast a `const ObjectBase*` to
// `member_class_t<decltype(Field)>*` (see model.h, e.g. Snapshot::resolve's
// `static_cast<const T*>(p)`), and member_class_t<Field> is deduced from
// Field's pointer-to-member type -- for a field declared in a base that
// ISN'T itself an ObjectBase, that cast target isn't related to ObjectBase
// by inheritance and the cast is simply ill-formed. asset_key()/gizmo_key()
// below are the sanctioned way around it: a computed key, declared on the
// concrete type itself, that reads an inherited field -- the same pattern
// Order::computed_key() already uses for a stored (not inherited) field.
namespace {

/// The shared virtual base at the bottom of the diamond. Plain data, no
/// model machinery -- ordinary C++ inheritance, nothing Object<Derived>-
/// specific about it.
struct Entity {
    int schema_version = 1;
};

/// virtual Entity: this and Labeled below both need to inherit from Entity
/// without Asset/Gizmo ending up with two separate Entity subobjects (the
/// diamond problem) -- virtual inheritance is what collapses them back to
/// one, checked explicitly in diamond_base_collapses_to_one_shared_subobject
/// below.
struct Timestamped : virtual Entity {
    std::int64_t created_at = 0;
};

struct Labeled : virtual Entity {
    std::string label;
};

/// Inherits Entity via TWO paths (Timestamped and Labeled) plus
/// model::Object<Asset> -- three base classes, two unrelated inheritance
/// graphs. owner is declared directly here (not in a mixin) specifically so
/// define_references()/define_cached_references() and the find_referrers/
/// find_cached_referrers APIs downcast cleanly -- see the file header.
class Asset final : public Timestamped, public Labeled, public model::Object<Asset> {
public:
    std::int64_t value = 0;
    model::Ref<Account> owner;  ///< non-nullable: deleting the account cascades

    /// Reads label, which lives in the Labeled base -- but the method itself
    /// is declared on Asset, so &Asset::asset_key is a pointer-to-member of
    /// Asset, not of Labeled. That's what keeps it usable with find_by_key.
    std::string asset_key() const { return "asset:" + label; }

    std::string to_string() const {
        auto ref_str = [](model::Id ref_id) {
            return std::to_string(ref_id.index) + ":" + std::to_string(ref_id.gen);
        };
        return "Asset{id=" + ref_str(id) + ", label=" + label + ", created_at=" + std::to_string(created_at) +
               ", schema_version=" + std::to_string(schema_version) + ", value=" + std::to_string(value) +
               ", owner=" + ref_str(owner.raw()) + "}";
    }

    template <class Self, class V>
    static void define_references(Self& s, V&& v) {
        v(model::field_tag<&Asset::owner>(), "owner", s.owner);
    }

    template <class Self>
    static void define_cached_references(Self& s, const model::RefIndexReader& v) {
        (void)s;
        v.index<&Asset::owner>("owner");
    }

    template <class Self>
    static void define_keys(Self& s, const model::FieldKeyReader& v) {
        v.key<&Asset::asset_key>(s.asset_key(), "asset_key");
        // The raw inherited field, declared under ITS OWN tag (&Labeled::label,
        // not &Asset::label -- same pointer-to-member value either way, since
        // label is only ever declared once, in Labeled; see
        // finding_by_the_inherited_label_field_uses_the_untyped_api below for
        // why that tag can only be queried back through find_by_key_raw, not
        // the typed find_by_key<Field>.
        v.key<&Labeled::label>(s.label, "label");
    }
};

/// A second type sharing the SAME diamond mixins (proving the pattern is
/// reusable, not a one-off), with its own Opt<Asset> -- the nullable/
/// cascade-null side of the reference story, exercised through a
/// diamond-derived type this time instead of a plain one.
class Gizmo final : public Timestamped, public Labeled, public model::Object<Gizmo> {
public:
    model::Opt<Asset> linked;  ///< nullable: deleting the asset nulls this instead of killing Gizmo

    std::string gizmo_key() const { return "giz:" + label; }

    std::string to_string() const {
        const model::Id linked_id = linked.raw();
        return "Gizmo{id=" + std::to_string(id.index) + ":" + std::to_string(id.gen) + ", label=" + label +
               ", linked=" + (linked ? std::to_string(linked_id.index) + ":" + std::to_string(linked_id.gen)
                                     : "null") +
               "}";
    }

    template <class Self, class V>
    static void define_references(Self& s, V&& v) {
        v(model::field_tag<&Gizmo::linked>(), "linked", s.linked);
    }

    template <class Self>
    static void define_keys(Self& s, const model::FieldKeyReader& v) {
        v.key<&Gizmo::gizmo_key>(s.gizmo_key(), "gizmo_key");
        // Same &Labeled::label tag Asset registers above -- NOT a
        // per-type copy of it. field_tag<Field> is keyed on the pointer-to-
        // member VALUE, and Labeled::label only has one address; Asset and
        // Gizmo both naming &Labeled::label means they share one entry in
        // Root::by_field, keyed by that single tag, with entries from BOTH
        // types living in it side by side. See
        // finding_by_the_inherited_label_field_can_return_either_diamond_
        // type below for what that implies: one raw lookup can hand back
        // either type, and a label live on one type blocks that SAME label
        // on the other.
        v.key<&Labeled::label>(s.label, "label");
    }
};

Ref<Asset> make_asset(Model& m, const std::string& label, Ref<Account> owner, std::int64_t created_at = 0) {
    Transaction txn = m.begin();
    auto a = std::make_unique<Asset>();
    a->label = label;
    a->owner = owner;
    a->created_at = created_at;
    const Ref<Asset> local = txn.create(std::move(a));
    const CommitResult res = commit_ok(m, txn);
    return res.to_real(local);
}

Ref<Gizmo> make_gizmo(Model& m, const std::string& label, Opt<Asset> linked = Opt<Asset>{}) {
    Transaction txn = m.begin();
    auto g = std::make_unique<Gizmo>();
    g->label = label;
    g->linked = linked;
    const Ref<Gizmo> local = txn.create(std::move(g));
    const CommitResult res = commit_ok(m, txn);
    return res.to_real(local);
}

}  // namespace

// Virtual inheritance is what makes this a true diamond rather than two
// independent Entity copies: Timestamped and Labeled each virtually inherit
// Entity, so Asset (and Gizmo) end up with exactly ONE Entity subobject,
// reachable through either base. Verified by pointer identity, not just by
// "it compiles" -- a non-virtual `struct Timestamped : Entity` here would
// still compile (Asset would just have two separate Entity subobjects, one
// per path) but this check would then fail.
TEST(diamond_base_collapses_to_one_shared_subobject) {
    Asset a;
    a.schema_version = 42;

    const Entity* via_timestamped = static_cast<Timestamped*>(&a);
    const Entity* via_labeled = static_cast<Labeled*>(&a);
    CHECK(via_timestamped == via_labeled);  // same object, not two copies

    // ...and since it's the SAME subobject, a write through one path is
    // visible through the other -- there's only one schema_version to see.
    static_cast<Timestamped&>(a).schema_version = 7;
    CHECK_EQ(static_cast<Labeled&>(a).schema_version, 7);
}

// A model type may inherit from arbitrary extra (non-model) base classes as
// long as it's still copyable -- Object<Derived>::clone() and assign_from()
// go through Asset's compiler-generated copy constructor/assignment, which
// correctly handles the virtual Entity base (copies it once, not twice, not
// zero times). Round-tripping through a real commit is what actually
// exercises clone(): Transaction::create() stores the object as-is, but
// Transaction::update() (further down) clones it, so this test's job is
// just to confirm fields from every base show up correctly after the
// create commits and is read back through a Snapshot.
TEST(model_type_with_diamond_base_creates_and_reads_back_fields_from_every_base) {
    Model m;
    const Ref<Account> acc = make_account(m, "OWNER");
    const Ref<Asset> a = make_asset(m, "ASSET1", acc, /*created_at=*/12345);

    Snapshot s = m.snapshot();
    const Asset* ap = s.find(a);
    CHECK(ap != nullptr);
    CHECK_EQ(ap->label, std::string("ASSET1"));     // from Labeled
    CHECK_EQ(ap->created_at, std::int64_t{12345});  // from Timestamped
    CHECK_EQ(ap->schema_version, 1);                // from Entity, via either mixin
    CHECK_EQ(s.resolve(ap->owner).id, acc.raw());   // Asset's own field, resolves normally
}

// define_keys() on a COMPUTED method (asset_key(), declared on Asset itself)
// that reads a field inherited from a mixin base works exactly like Order::
// computed_key() reading a stored field -- find_by_key doesn't know or care
// that `label` lives in Labeled rather than Asset. Duplicate rejection
// (Model::validate_field_key_uniqueness) applies the same way too.
TEST(model_type_with_diamond_base_supports_keys_computed_from_an_inherited_field) {
    Model m;
    const Ref<Account> acc = make_account(m, "OWNER");
    const Ref<Asset> a1 = make_asset(m, "DUP", acc);

    Snapshot s = m.snapshot();
    CHECK(s.find_by_key<&Asset::asset_key>("asset:DUP") != nullptr);
    CHECK_EQ(s.find_by_key<&Asset::asset_key>("asset:DUP")->id, a1.raw());
    CHECK(s.find_by_key<&Asset::asset_key>("asset:NOPE") == nullptr);

    // Same label as a1 -> rejected, same as two Accounts sharing a name.
    Transaction txn = m.begin();
    auto dup = std::make_unique<Asset>();
    dup->label = "DUP";
    dup->owner = acc;
    txn.create(std::move(dup));
    const CommitResult res = m.try_commit(txn);
    CHECK(res.status == CommitStatus::Invalid);
    CHECK_EQ(m.snapshot().size(), std::size_t{2});  // acc + a1 only
}

// Unlike asset_key() (a method declared ON Asset), `label` above is declared
// in define_keys() under ITS OWN tag -- &Labeled::label, a pointer-to-member
// of Labeled, not of Asset (see the file header: the enclosing class of an
// inherited member's pointer-to-member type is where it's actually
// declared). find_by_key<Field>'s return type is
// `const member_class_t<decltype(Field)>*`, so find_by_key<&Labeled::label>
// would have to return `const Labeled*` -- and since Labeled doesn't derive
// from ObjectBase (it's a plain mixin, unrelated to the model), the
// static_cast<const Labeled*>(ObjectBase*) that implementation needs is
// ill-formed. That's a compile error, not a runtime one, so it can't be
// asserted with CHECK -- the absence of a `find_by_key<&Labeled::label>`
// call anywhere in this file (or a `static_assert(false)` if you try it) IS
// the test. find_by_key_raw is the sanctioned way around it: the untyped
// escape hatch Snapshot::find_by_key<Field> itself is built on (see its own
// doc comment), which returns `const ObjectBase*` and leaves the downcast to
// the caller -- who, unlike the generic template, knows the concrete type is
// Asset (which DOES derive from ObjectBase, via model::Object<Asset>) rather
// than blindly trusting member_class_t.
TEST(finding_by_the_inherited_label_field_uses_the_untyped_api) {
    Model m;
    const Ref<Account> acc = make_account(m, "OWNER");
    const Ref<Asset> a1 = make_asset(m, "ASSET1", acc);
    make_asset(m, "ASSET2", acc);

    // Same NTTP value either way -- there's only one `label`, declared once,
    // in Labeled -- so the tag registered via &Labeled::label in
    // define_keys() is found the same way however it's spelled at the call
    // site.
    CHECK(model::field_tag<&Labeled::label>() == model::field_tag<&Asset::label>());

    Snapshot s = m.snapshot();
    const ObjectBase* raw = s.find_by_key_raw(model::field_tag<&Labeled::label>(), "ASSET1");
    CHECK(raw != nullptr);
    CHECK(raw->tag() == model::type_tag<Asset>());  // the caller's job, not find_by_key_raw's
    const Asset* ap = static_cast<const Asset*>(raw);
    CHECK_EQ(ap->id, a1.raw());
    CHECK_EQ(ap->label, std::string("ASSET1"));

    CHECK(s.find_by_key_raw(model::field_tag<&Labeled::label>(), "NOPE") == nullptr);
}

// The keyspace behind a define_keys() entry belongs to the FIELD TAG, not to
// any one type (see find_by_key_raw's own doc comment: "the key string is
// only unique within field's own keyspace"). Asset and Gizmo both register
// &Labeled::label -- the SAME tag, since it names the one place `label` is
// actually declared -- so Root::by_field holds ONE map for it with entries
// from both types mixed together. That has two consequences neither a
// single-type field could show:
//   1. A raw lookup by that tag can come back as EITHER type -- the caller
//      has to check tag() (or try find<T>) rather than assume one fixed
//      type, unlike find_by_key_raw's other uses in this file where the
//      answer, if any, is always an Asset.
//   2. Uniqueness is enforced ACROSS the two types: an Asset and a Gizmo
//      cannot hold the same label value, even though nothing about Asset or
//      Gizmo individually looks like it should collide with the other.
TEST(finding_by_the_inherited_label_field_can_return_either_diamond_type) {
    Model m;
    const Ref<Account> acc = make_account(m, "OWNER");
    const Ref<Asset> a = make_asset(m, "ASSET-LABEL", acc);
    const Ref<Gizmo> g = make_gizmo(m, "GIZMO-LABEL");

    Snapshot s = m.snapshot();

    const ObjectBase* raw_a = s.find_by_key_raw(model::field_tag<&Labeled::label>(), "ASSET-LABEL");
    CHECK(raw_a != nullptr);
    CHECK(raw_a->tag() == model::type_tag<Asset>());
    CHECK_EQ(static_cast<const Asset*>(raw_a)->id, a.raw());

    const ObjectBase* raw_g = s.find_by_key_raw(model::field_tag<&Labeled::label>(), "GIZMO-LABEL");
    CHECK(raw_g != nullptr);
    CHECK(raw_g->tag() == model::type_tag<Gizmo>());  // a DIFFERENT concrete type, same tag lookup
    CHECK_EQ(static_cast<const Gizmo*>(raw_g)->id, g.raw());

    // Cross-type collision: a Gizmo claiming an already-live Asset's label
    // (or vice versa) is rejected exactly like same-type duplicates are --
    // the two types are competing for entries in the same shared keyspace.
    Transaction txn = m.begin();
    auto dup = std::make_unique<Gizmo>();
    dup->label = "ASSET-LABEL";  // already held by `a`, a totally different type
    txn.create(std::move(dup));
    const CommitResult res = m.try_commit(txn);
    CHECK(res.status == CommitStatus::Invalid);

    Snapshot s2 = m.snapshot();
    CHECK_EQ(s2.size(), std::size_t{3});  // acc, a, g -- the cross-type dup never landed
    // The tag still resolves to the Asset, unchanged -- the Gizmo never
    // got a chance to overwrite or shadow it.
    CHECK(s2.find_by_key_raw(model::field_tag<&Labeled::label>(), "ASSET-LABEL")->tag() ==
          model::type_tag<Asset>());
}

// A duplicate `label` is rejected the same way a duplicate asset_key() is
// (they're 1:1 here, but this drives the rejection through the BASE field's
// own key entry specifically, not the computed one) -- define_keys()'s
// uniqueness check (Model::validate_field_key_uniqueness) doesn't
// distinguish a field declared directly on Derived from one declared via
// `&Labeled::label`; both are just entries keyed by field_tag.
TEST(duplicate_inherited_label_is_rejected_like_any_other_key) {
    Model m;
    const Ref<Account> acc = make_account(m, "OWNER");
    make_asset(m, "SAME", acc);

    Transaction txn = m.begin();
    auto dup = std::make_unique<Asset>();
    dup->label = "SAME";   // collides on the &Labeled::label key
    dup->owner = acc;
    txn.create(std::move(dup));
    const CommitResult res = m.try_commit(txn);
    CHECK(res.status == CommitStatus::Invalid);
    CHECK_EQ(m.snapshot().size(), std::size_t{2});  // acc + the first Asset only
}

// Same reconciliation guarantee invariant 9 already covers for asset_key()
// (see update_through_diamond_base_fields_reconciles_together further down),
// checked here specifically through the &Labeled::label key entry: the old
// value's index slot must be gone and the new value's must be findable,
// straight after one Transaction::update().
TEST(updating_the_inherited_label_field_reconciles_its_own_key_entry) {
    Model m;
    const Ref<Account> acc = make_account(m, "OWNER");
    const Ref<Asset> a = make_asset(m, "OLD", acc);

    update_field(m, a, [](Asset* p) { p->label = "NEW"; });

    Snapshot s = m.snapshot();
    const ObjectBase* old_raw = s.find_by_key_raw(model::field_tag<&Labeled::label>(), "OLD");
    CHECK(old_raw == nullptr);  // stale key entry: gone
    const ObjectBase* new_raw = s.find_by_key_raw(model::field_tag<&Labeled::label>(), "NEW");
    CHECK(new_raw != nullptr);
    CHECK_EQ(static_cast<const Asset*>(new_raw)->id, a.raw());
}

// Asset's non-nullable owner Ref<Account> is declared directly on Asset
// (not a mixin) -- cascade delete must treat it exactly like Order::account
// despite Asset's extra diamond bases: define_references()/each_ref()/
// clone()/RefRemapper are all reached the same way (through the
// ObjectBase virtual interface), unaffected by what else Asset derives from.
TEST(model_type_with_diamond_base_cascades_on_non_nullable_ref) {
    Model m;
    const Ref<Account> acc = make_account(m, "OWNER");
    const Ref<Asset> a = make_asset(m, "ASSET1", acc);

    const std::size_t killed = remove_and_commit(m, acc);

    CHECK_EQ(killed, std::size_t{2});  // acc, and Asset (non-nullable Ref<Account>)
    Snapshot s = m.snapshot();
    CHECK(s.find(acc) == nullptr);
    CHECK(s.find(a) == nullptr);
    CHECK(s.find_by_key<&Asset::asset_key>("asset:ASSET1") == nullptr);  // index cleaned up too
}

// Gizmo's Opt<Asset> is nullable: deleting the linked Asset must null the
// field and leave Gizmo (itself diamond-derived) alive -- the null_ref()/
// RefNuller path, exercised through a type with the same extra bases.
TEST(model_type_with_diamond_base_nulls_on_nullable_ref) {
    Model m;
    const Ref<Account> acc = make_account(m, "OWNER");
    const Ref<Asset> a = make_asset(m, "ASSET1", acc);
    const Ref<Gizmo> g = make_gizmo(m, "GIZMO1", Opt<Asset>(a));

    const std::size_t killed = remove_and_commit(m, a);

    CHECK_EQ(killed, std::size_t{1});  // only the asset; the account and gizmo survive
    Snapshot s = m.snapshot();
    CHECK(s.find(acc) != nullptr);
    const Gizmo* gp = s.find(g);
    CHECK(gp != nullptr);
    CHECK(!gp->linked);  // nulled, not dangling
}

// define_cached_references()/find_cached_referrers on a diamond-derived
// type's own field -- the indexed counterpart of the scan above, same
// downcast concern as find_by_key, satisfied the same way (owner lives on
// Asset itself).
TEST(model_type_with_diamond_base_supports_cached_referrer_lookup) {
    Model m;
    const Ref<Account> acc1 = make_account(m, "OWNER1");
    const Ref<Account> acc2 = make_account(m, "OWNER2");
    const Ref<Asset> a1 = make_asset(m, "A1", acc1);
    const Ref<Asset> a2 = make_asset(m, "A2", acc1);
    make_asset(m, "A3", acc2);

    Snapshot s = m.snapshot();
    const auto for_acc1 = s.find_cached_referrers<&Asset::owner>(acc1);
    CHECK_EQ(for_acc1.size(), std::size_t{2});
    bool saw_a1 = false, saw_a2 = false;
    for (const Asset* ap : for_acc1) {
        if (ap->id == a1.raw()) saw_a1 = true;
        if (ap->id == a2.raw()) saw_a2 = true;
    }
    CHECK(saw_a1);
    CHECK(saw_a2);
    CHECK_EQ(s.find_cached_referrers<&Asset::owner>(acc2).size(), std::size_t{1});
}

// Transaction::update() clones the committed object (ObjectBase::clone(),
// via the copy constructor) for local editing, then reconciles indexes
// against the FINAL written value at apply time (invariant 9). Writing
// through fields from BOTH mixin bases in the same transaction, plus
// Asset's own value field, must all land together and the key index must
// track the new (not old) label.
TEST(update_through_diamond_base_fields_reconciles_together) {
    Model m;
    const Ref<Account> acc = make_account(m, "OWNER");
    const Ref<Asset> a = make_asset(m, "OLD", acc, /*created_at=*/1);

    update_field(m, a, [](Asset* p) {
        p->label = "NEW";       // Labeled
        p->created_at = 999;    // Timestamped
        p->schema_version = 2;  // Entity, via either mixin
        p->value = 100;         // Asset's own
    });

    Snapshot s = m.snapshot();
    const Asset* ap = s.find(a);
    CHECK(ap != nullptr);
    CHECK_EQ(ap->label, std::string("NEW"));
    CHECK_EQ(ap->created_at, std::int64_t{999});
    CHECK_EQ(ap->schema_version, 2);
    CHECK_EQ(ap->value, std::int64_t{100});

    CHECK(s.find_by_key<&Asset::asset_key>("asset:NEW") != nullptr);
    CHECK(s.find_by_key<&Asset::asset_key>("asset:OLD") == nullptr);  // old index entry gone
}
