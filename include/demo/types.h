#pragma once
//
// Example domain types. These are user code, not part of the model -- they show
// exactly what a type has to provide to live in the store.
//
// The contract:
//   define_references()  optional: list every reference field ONCE, tagged by its own address
//   define_keys()        optional: zero or more fields (or computed methods) for fast lookup
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

namespace demo {

class Account final : public model::Object<Account> {
public:
    std::string name;
    std::int64_t balance = 0;

    template <class Self>
    static void define_keys(Self& s, const model::FieldKeyReader& v) {
        v.key<&Account::name>(s.name);
    }
};

class Order final : public model::Object<Order> {
public:
    std::string code;
    model::Ref<Account> account;  ///< deleting the account kills this order
    model::Opt<Order> parent;     ///< deleting the parent nulls this field
    std::int64_t qty = 0;

    std::string computed_key() const { return "ord:" + code; }

    template <class Self, class V>
    static void define_references(Self& s, V&& v) {
        v(model::field_tag<&Order::account>(), "account", s.account);
        v(model::field_tag<&Order::parent>(), "parent", s.parent);
    }

    template <class Self>
    static void define_keys(Self& s, const model::FieldKeyReader& v) {
        v.key<&Order::computed_key>(s.computed_key());
    }
};

}  // namespace demo
