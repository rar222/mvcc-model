#pragma once
//
// Account/Order: the tests' own copies of the domain types that used to be
// shared with examples/ via include/example/types.h. Field-for-field
// identical (many tests check exact string formats like "ord:O1" that depend
// on it), but defined here so tests/ has no dependency on include/example/ or
// examples/ at all -- the two are free to diverge without either one's
// tests breaking. See include/example/types.h for the field-by-field
// commentary on what each define_X() does; that commentary isn't repeated
// here to avoid two copies drifting out of sync with each other in meaning
// (only the code itself is deliberately duplicated, not its explanation).

#include <cstdint>
#include <string>

#include "model/model.h"

class Account final : public model::Object<Account> {
public:
    std::string name;
    std::int64_t balance = 0;

    /// Diagnostic only -- see include/example/types.h's Account::to_string()
    /// doc comment for why `id` comes first and refs (none here) would print
    /// as raw index:gen.
    std::string to_string() const {
        return "Account{id=" + std::to_string(id.index) + ":" + std::to_string(id.gen) +
               ", name=" + name + ", balance=" + std::to_string(balance) + "}";
    }

    template <class Self>
    static void define_keys(Self& s, const model::FieldKeyReader& v) {
        v.key<&Account::name>(s.name, "name");
    }

    template <class Self>
    static void define_scan_fields(Self& s, const model::FieldKeyReader& v) {
        v.key<&Account::name>(s.name, "name");
    }
};

class Order final : public model::Object<Order> {
public:
    std::string code;
    model::Ref<Account> account;
    model::Opt<Order> parent;
    std::int64_t qty = 0;

    std::string computed_key() const { return "ord:" + code; }

    /// Diagnostic only -- see include/example/types.h's Order::to_string()
    /// doc comment for why `id` comes first and account/parent print as raw
    /// index:gen instead of being resolved.
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

    template <class Self>
    static void define_cached_references(Self& s, const model::RefIndexReader& v) {
        (void)s;
        v.index<&Order::account>("account");
    }

    template <class Self>
    static void define_keys(Self& s, const model::FieldKeyReader& v) {
        v.key<&Order::computed_key>(s.computed_key(), "computed_key");
    }

    template <class Self>
    static void define_scan_fields(Self& s, const model::FieldKeyReader& v) {
        v.key<&Order::qty>(s.qty, "qty");
        v.key<&Order::computed_key>(s.computed_key(), "computed_key");
    }

    template <class Self>
    static void define_cached_fields(Self& s, const model::FieldKeyReader& v) {
        v.key<&Order::qty>(s.qty, "qty");
        v.key<&Order::computed_key>(s.computed_key(), "computed_key");
    }
};
