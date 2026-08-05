#pragma once
//
// Example domain types. These are user code, not part of the model -- they show
// exactly what a type has to provide to live in the store.
//
// The declaration contract (each define_X below): a field left out of
// define_fields()/define_references() is invisible to find_by_field/find_referrers
// (find_by_key on an undeclared field is similarly invisible to define_keys()) --
// empty, nothing walked. Each field/reference carries a single LookupType tag
// (Cache or Scan) attached at its one declaration site -- a field can be tagged
// exactly one way, never both. See model.h's Object<Derived> class comment for
// the full contract.
//   define_references()     optional: list every reference field ONCE, tagged by its own
//                           address and a model::LookupType (Cache or Scan) -- Cache backs
//                           find_referrers's "who points at this?" lookup with an INDEXED
//                           O(log n + matches) reverse index (one entry per object,
//                           maintained every commit); Scan leaves it on the always-
//                           available O(#objects) scan fallback (zero write-side cost).
//                           Declare Cache only for what's queried often.
//   define_keys()           optional: zero or more fields (or computed methods) for fast
//                           UNIQUE lookup (find_by_key; a create/update that would duplicate
//                           another live object's value is rejected, CommitStatus::Invalid)
//   define_fields()         optional: zero or more fields, each tagged model::LookupType::
//                           Cache or model::LookupType::Scan, that make find_by_field
//                           reachable -- Cache resolves via an INDEXED O(log n + matches)
//                           lookup (one index entry per object, maintained every commit);
//                           Scan resolves via an UNINDEXED O(#objects) scan (zero write-side
//                           cost). Declare Cache only for what's queried often.
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

    /// name is tagged Scan (nothing else references it as Cache), so
    /// find_by_field<&Account::name> exercises the scan-fallback branch.
    /// It's ALSO a define_keys() field, so no two live Accounts can
    /// actually share a name (a duplicate is rejected at commit time),
    /// meaning find_by_field on it can only ever return zero or one match
    /// in practice. See Order::qty for a scan field where genuine
    /// multi-match duplicates are legal.
    template <class Self>
    static void define_fields(Self& s, const model::LookupFieldReader& v) {
        v.field<&Account::name>(s.name, model::LookupType::Scan, "name");
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

    /// account is the classic hot reverse lookup ("every Order for this
    /// Account") -- worth the index, so it's tagged LookupType::Cache and
    /// find_referrers<&Order::account> resolves via Root::by_cached_
    /// reference. parent and account_scan are deliberately tagged Scan:
    /// queried rarely enough that find_referrers's O(#orders) scan fallback
    /// is the better trade (zero write-side cost) -- the same "index only
    /// what's worth it" tradeoff define_fields() applies to plain fields,
    /// below. account_scan exists specifically to give find_referrers a
    /// scan-fallback case shaped like account's cache-hit case (see its own
    /// field comment); parent is Order's genuinely-nullable, rarely-queried
    /// ref. All three are still declared in define_references() regardless
    /// of tag -- that's what makes account CASCADE and parent/account_scan
    /// NULL correctly on their target's delete (see Ref<>/Opt<> nullability
    /// above); the LookupType only adds (or withholds) the fast reverse
    /// lookup on top of that.
    template <class Self, class V>
    static void define_references(Self& s, V&& v) {
        v(model::field_tag<&Order::account>(), s.account, model::LookupType::Cache, "account");
        v(model::field_tag<&Order::parent>(), s.parent, model::LookupType::Scan, "parent");
        v(model::field_tag<&Order::account_scan>(), s.account_scan, model::LookupType::Scan,
          "account_scan");
    }

    template <class Self>
    static void define_keys(Self& s, const model::FieldKeyReader& v) {
        v.key<&Order::computed_key>(s.computed_key(), "computed_key");
    }

    /// qty is deliberately non-unique (many orders share a quantity), so it
    /// goes in the MULTI-match family, not define_keys():
    /// s.find_by_field<&Order::qty>(5) -> every order with qty == 5.
    /// Tagged Cache, so that resolves via the index; qty_scan is a scan-
    /// only twin kept equal to qty (see its own field comment) so find_by_
    /// field/find_referrers's scan-fallback branch can be exercised on data
    /// shaped exactly like qty's cache-hit case. computed_key is ALSO
    /// declared here (in addition to define_keys() above) purely as a
    /// fixture: the two declarations aren't a meaningful redundancy for
    /// computed_key itself, just a way for tests to prove the cache and key
    /// lookup paths agree.
    template <class Self>
    static void define_fields(Self& s, const model::LookupFieldReader& v) {
        v.field<&Order::qty>(s.qty, model::LookupType::Cache, "qty");
        v.field<&Order::computed_key>(s.computed_key(), model::LookupType::Cache, "computed_key");
        v.field<&Order::qty_scan>(s.qty_scan, model::LookupType::Scan, "qty_scan");
    }
};

}  // namespace example
