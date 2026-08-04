#pragma once
//
// Example domain types. These are user code, not part of the model -- they show
// exactly what a type has to provide to live in the store.
//
// The contract (each define_X drives find_by_X and view_by_X; a field you
// don't declare is invisible to that family's lookup):
//   define_references()     optional: list every reference field ONCE, tagged by its own address
//   define_keys()           optional: zero or more fields (or computed methods) for fast
//                           UNIQUE lookup (find_by_key; a create/update that would duplicate
//                           another live object's value is rejected, CommitStatus::Invalid)
//   define_scan_fields()    optional: zero or more fields for UNINDEXED multi-match lookup
//                           (find_by_scan_field; O(#objects) scan per query, zero write-side
//                           cost -- for fields queried rarely)
//   define_cached_fields()  optional: zero or more fields for INDEXED multi-match lookup
//                           (find_by_cached_field; every match kept, O(log n + matches)).
//                           Costs one index entry per object per field, maintained on every
//                           commit -- declare only what's queried often.
//   define_cached_references()  optional: zero or more Ref<>/Opt<> fields -- ALREADY listed
//                           in define_references() -- for INDEXED "who points at this?"
//                           lookup (find_cached_referrers; O(log n + matches) instead of
//                           for_each_referrers's always-available O(#objects) scan). Same
//                           cost model as define_cached_fields; declare only what's queried
//                           often.
//
// (type() also exists, for diagnostic messages, but Object<Derived> derives it
// from typeid() automatically -- there's nothing to override.)
//
// define_references() drives clone(), the reverse index, cascade delete
// (resolved at Model::try_commit() time, not eagerly), and nulling.
// Nullability comes from the field TYPE:
//
//   Ref<T>  non-nullable. Deleting the target CASCADES: this object dies too.
//   Opt<T>  nullable.     Deleting the target NULLS this field; the object lives.
//
// A ref field left out of define_references() is invisible to the model. There is no
// compiler error and no assert -- just a dangling reference in production. Add a
// field, add a line, add a test. Each field passes model::field_tag<&Type::field>()
// as its own identity -- there's no int slot number to hand-assign or keep in
// sync, the same way define_keys() fields identify themselves. The string right
// after it ("account", "parent") is diagnostic-only, for IntegrityError messages;
// get it wrong and you mislabel an error, nothing more.
//
// There is no mandatory identity key. define_keys() opts zero or more fields
// into Snapshot::find_by_key, and the model keeps each one's index entry in
// sync even if the underlying field is reassigned after Transaction::update().
// A define_keys() entry can also name a nullary const method instead of a
// data member -- Account/Order both use this for a computed_key() that
// combines a type prefix with a stored field.

#include <cstdint>
#include <string>

#include "model/model.h"

namespace example {

class Account final : public model::Object<Account> {
public:
    std::string name;
    std::int64_t balance = 0;

    /// Diagnostic only -- e.g. for a subscriber printing a changeset. `id`
    /// (own index:gen) always comes first, so every type's to_string() is
    /// grep-able by id the same way. Refs are printed as raw index:gen, not
    /// resolved: a Ref<T>/Opt<T> has no snapshot pointer to resolve through
    /// (see Ref<T>'s doc comment), and to_string() has no Snapshot parameter
    /// to resolve one with anyway.
    std::string to_string() const {
        return "Account{id=" + std::to_string(id.index) + ":" + std::to_string(id.gen) +
               ", name=" + name + ", balance=" + std::to_string(balance) + "}";
    }

    template <class Self>
    static void define_keys(Self& s, const model::FieldKeyReader& v) {
        v.key<&Account::name>(s.name, "name");
    }

    /// name is ALSO a scan field, so the same field can live in more than
    /// one lookup family -- but since it's ALSO a define_keys() field, no
    /// two live Accounts can actually share a name (a duplicate is rejected
    /// at commit time), so find_by_scan_field on it can only ever return
    /// zero or one match in practice. See Order::qty for a scan field where
    /// genuine multi-match duplicates are legal.
    template <class Self>
    static void define_scan_fields(Self& s, const model::FieldKeyReader& v) {
        v.key<&Account::name>(s.name, "name");
    }
};

class Order final : public model::Object<Order> {
public:
    std::string code;
    model::Ref<Account> account;  ///< deleting the account kills this order
    model::Opt<Order> parent;     ///< deleting the parent nulls this field
    std::int64_t qty = 0;

    std::string computed_key() const { return "ord:" + code; }

    /// Diagnostic only -- see Account::to_string()'s doc comment for why
    /// `id` comes first and refs print as raw index:gen instead of being
    /// resolved.
    std::string to_string() const {
        auto ref_str = [](model::Id ref_id) {
            return std::to_string(ref_id.index) + ":" + std::to_string(ref_id.gen);
        };
        return "Order{id=" + ref_str(id) + ", code=" + code + ", qty=" + std::to_string(qty) +
               ", account=" + ref_str(account.raw()) +
               ", parent=" + (parent ? ref_str(parent.raw()) : "null") + "}";
    }

    template <class Self, class V>
    static void define_references(Self& s, V&& v) {
        v(model::field_tag<&Order::account>(), "account", s.account);
        v(model::field_tag<&Order::parent>(), "parent", s.parent);
    }

    /// account is the classic hot reverse lookup ("every Order for this
    /// Account") -- worth the index. parent is deliberately left OUT: it's
    /// queried rarely enough that for_each_referrers<&Order::parent>(...)'s
    /// O(#orders) scan is the better trade (zero write-side cost) -- the same
    /// "index only what's worth it" tradeoff define_cached_fields() and
    /// define_scan_fields() apply to plain fields, above. Both account and
    /// parent are still declared in define_references() regardless -- that's
    /// what makes them cascade/null correctly; this declaration only adds
    /// the fast lookup on top of account.
    template <class Self>
    static void define_cached_references(Self& s, const model::RefIndexReader& v) {
        (void)s;
        v.index<&Order::account>("account");
    }

    template <class Self>
    static void define_keys(Self& s, const model::FieldKeyReader& v) {
        v.key<&Order::computed_key>(s.computed_key(), "computed_key");
    }

    /// The scan family: queryable via find_by_scan_field / view_by_scan_field
    /// (O(#orders) per query, zero write-side cost). qty appears in BOTH
    /// multi-match families -- scan here, cached below -- to show they're
    /// independent; a real type would usually pick one per field.
    template <class Self>
    static void define_scan_fields(Self& s, const model::FieldKeyReader& v) {
        v.key<&Order::qty>(s.qty, "qty");
        v.key<&Order::computed_key>(s.computed_key(), "computed_key");
    }

    /// qty is deliberately non-unique (many orders share a quantity), so it
    /// goes in a MULTI-match family, not define_keys():
    /// s.find_by_cached_field<&Order::qty>(5) -> every order with qty == 5.
    template <class Self>
    static void define_cached_fields(Self& s, const model::FieldKeyReader& v) {
        v.key<&Order::qty>(s.qty, "qty");
        v.key<&Order::computed_key>(s.computed_key(), "computed_key");
    }
};

}  // namespace example
