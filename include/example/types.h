#pragma once
//
// Example domain types. These are user code, not part of the model -- they show
// exactly what a type has to provide to live in the store.
//
// The declaration contract (each define_X below) is unchanged: a field left out of
// every lookup family is invisible to it (find_by_key on an undeclared field,
// find_by_field/find_referrers on a field declared in neither family -- empty,
// nothing walked). What changed is the READ side: find_by_scan_field/find_by_
// cached_field collapsed into one find_by_field, and for_each_cached_referrers/
// find_cached_referrers into for_each_referrers/find_referrers -- each cache-first,
// scan-fallback. See model.h's Object<Derived> class comment for the full contract.
//   define_references()     optional: list every reference field ONCE, tagged by its own address
//   define_keys()           optional: zero or more fields (or computed methods) for fast
//                           UNIQUE lookup (find_by_key; a create/update that would duplicate
//                           another live object's value is rejected, CommitStatus::Invalid)
//   define_scan_fields()    optional: zero or more fields that make find_by_field reachable
//                           via an UNINDEXED O(#objects) scan when the same field isn't ALSO
//                           in define_cached_fields() -- zero write-side cost, for fields
//                           queried rarely
//   define_cached_fields()  optional: zero or more fields that make find_by_field resolve via
//                           an INDEXED O(log n + matches) lookup instead (every match kept).
//                           Costs one index entry per object per field, maintained on every
//                           commit -- declare only what's queried often; find_by_field always
//                           prefers this over the scan when a field is in both families.
//   define_cached_references()  optional: zero or more Ref<>/Opt<> fields -- ALREADY listed
//                           in define_references() -- that make find_referrers's "who points
//                           at this?" lookup resolve via an INDEXED O(log n + matches) lookup
//                           instead of its always-available O(#objects) scan fallback. Same
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

    /// name is ALSO a scan field (and nothing else -- not in define_cached_
    /// fields()), so find_by_field<&Account::name> exercises the scan-
    /// fallback branch. It's ALSO a define_keys() field, so no two live
    /// Accounts can actually share a name (a duplicate is rejected at commit
    /// time), meaning find_by_field on it can only ever return zero or one
    /// match in practice. See Order::qty for a scan field where genuine
    /// multi-match duplicates are legal.
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

    /// Scan-only twins of qty/account, kept equal to them by whichever
    /// caller sets qty/account -- there's no derived-field mechanism in the
    /// model (see the "Why not compute a key from a joined field" comment in
    /// model.h), so keeping these in sync is the caller's job, same as any
    /// other field. They exist purely so find_by_field/find_referrers's
    /// scan-fallback branch can be exercised on data shaped exactly like
    /// qty/account's cache-hit case, instead of forcing every scan-fallback
    /// test onto a differently-distributed field. account_scan is Opt<>,
    /// not Ref<> like account: it defaults to null and MOST Order-creation
    /// call sites never set it (only the ones exercising the scan-fallback
    /// referrer case do), so it can't be non-nullable -- a Ref<> left unset
    /// would fail commit validation (CLAUDE.md invariant 1) on every one of
    /// those other call sites.
    std::int64_t qty_scan = 0;
    model::Opt<Account> account_scan;

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
        v(model::field_tag<&Order::account_scan>(), "account_scan", s.account_scan);
    }

    /// account is the classic hot reverse lookup ("every Order for this
    /// Account") -- worth the index, so find_referrers<&Order::account>
    /// resolves via Root::by_cached_reference. parent and account_scan are
    /// deliberately left OUT: queried rarely enough that find_referrers's
    /// O(#orders) scan fallback is the better trade (zero write-side cost)
    /// -- the same "index only what's worth it" tradeoff define_cached_
    /// fields()/define_scan_fields() apply to plain fields, above.
    /// account_scan exists specifically to give find_referrers a scan-
    /// fallback case shaped like account's cache-hit case (see its own field
    /// comment); parent is Order's genuinely-nullable, rarely-queried ref.
    /// All three are still declared in define_references() regardless --
    /// that's what makes account CASCADE and parent/account_scan NULL
    /// correctly on their target's delete (see Ref<>/Opt<> nullability
    /// above); this declaration only adds the fast lookup on top of
    /// account.
    template <class Self>
    static void define_cached_references(Self& s, const model::RefIndexReader& v) {
        (void)s;
        v.index<&Order::account>("account");
    }

    template <class Self>
    static void define_keys(Self& s, const model::FieldKeyReader& v) {
        v.key<&Order::computed_key>(s.computed_key(), "computed_key");
    }

    /// The scan-fallback family: makes find_by_field reachable via an
    /// UNINDEXED O(#orders) scan when a field isn't ALSO in define_cached_
    /// fields() (qty_scan isn't, so it stays on the scan; qty also appears
    /// in define_cached_fields() below, so find_by_field<&Order::qty>
    /// resolves via the index instead -- the two declarations aren't a
    /// meaningful redundancy for qty itself, just a fixture the tests use to
    /// prove both paths agree).
    template <class Self>
    static void define_scan_fields(Self& s, const model::FieldKeyReader& v) {
        v.key<&Order::qty>(s.qty, "qty");
        v.key<&Order::computed_key>(s.computed_key(), "computed_key");
        v.key<&Order::qty_scan>(s.qty_scan, "qty_scan");
    }

    /// qty is deliberately non-unique (many orders share a quantity), so it
    /// goes in a MULTI-match family, not define_keys():
    /// s.find_by_field<&Order::qty>(5) -> every order with qty == 5, via the
    /// index (this declaration is what makes it resolve that way instead of
    /// falling back to the scan qty_scan uses).
    template <class Self>
    static void define_cached_fields(Self& s, const model::FieldKeyReader& v) {
        v.key<&Order::qty>(s.qty, "qty");
        v.key<&Order::computed_key>(s.computed_key(), "computed_key");
    }
};

}  // namespace example
