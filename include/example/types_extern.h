#pragma once
//
// extern template coverage for example::Account / example::Order.
//
// include/example/types.h's Account/Order are ordinary generic-model user
// types: every Snapshot::resolve<T>, Transaction::create<T>,
// Snapshot::find_by_key<Field>, ... they touch is a template, so a TU that
// #includes types.h and calls these implicitly (re)instantiates -- and, at
// -O2, (re)optimizes -- its own private copy of each one. Across many TUs
// that all use Account/Order, that's the same work paid for repeatedly; the
// linker later folds the duplicate weak symbols away, but only after every
// TU already did the work.
//
// This header + examples/types_extern.cpp move that work to ONE place:
// types_extern.cpp explicitly instantiates every (non-callback-taking)
// Snapshot/Transaction/Model/BulkTransaction/View entry point Account or
// Order can appear in, and this header's `extern template` declarations
// tell every OTHER TU "the definition lives elsewhere" -- so it emits a
// call to that one already-compiled copy instead of redoing the work.
//
// This is NOT a blanket win -- see examples/extern_template_demo.cpp's own
// header comment for measured numbers and when it's and isn't worth it.
// Notably excluded: every callback/predicate-taking template (for_each<T,F>,
// for_each_by_scan_field<Field,F>, find_all<T,Pred>, View::for_each_referrer,
// ...). Each call site's lambda is its own distinct type, so there is
// nothing to share across TUs for those regardless of this technique --
// listing them here would just be dead weight.
//
// Must be spelled from inside `namespace model` (not qualified as
// `model::Transaction::create<...>` from outside it) -- GCC and Clang both
// reject a fully-qualified member-template-id used that way as an explicit
// instantiation declarator ("template-id used as a declarator").
//
// Extend this list only for entry points genuinely called from more than
// one TU that includes this header -- check before adding more; an unused
// entry here is pure maintenance cost with no compile-time benefit.

#include "example/types.h"

namespace model {

// ---- Snapshot: Account ----
extern template const example::Account& Snapshot::resolve<example::Account>(Ref<example::Account>) const noexcept;
extern template const example::Account* Snapshot::find<example::Account>(Ref<example::Account>) const noexcept;
extern template const example::Account* Snapshot::find_by_key<&example::Account::name>(const std::string&) const;
extern template std::optional<View<example::Account>> Snapshot::view_by_key<&example::Account::name>(const std::string&) const;
extern template std::vector<const example::Account*> Snapshot::find_by_scan_field<&example::Account::name>(const std::string&) const;
extern template std::vector<View<example::Account>> Snapshot::view_by_scan_field<&example::Account::name>(const std::string&) const;
extern template View<example::Account> Snapshot::view<example::Account>(const example::Account&) const noexcept;
extern template std::optional<View<example::Account>> Snapshot::view<example::Account>(Ref<example::Account>) const;

// ---- Snapshot: Order ----
extern template const example::Order* Snapshot::resolve<example::Order>(Opt<example::Order>) const noexcept;
extern template const example::Order* Snapshot::find<example::Order>(Ref<example::Order>) const noexcept;
extern template const example::Order* Snapshot::find<example::Order>(Opt<example::Order>) const noexcept;
extern template const example::Order* Snapshot::find_by_key<&example::Order::computed_key>(const std::string&) const;
extern template std::optional<View<example::Order>> Snapshot::view_by_key<&example::Order::computed_key>(const std::string&) const;
extern template std::vector<const example::Order*> Snapshot::find_by_scan_field<&example::Order::qty>(const std::int64_t&) const;
extern template std::vector<const example::Order*> Snapshot::find_by_scan_field<&example::Order::computed_key>(const std::string&) const;
extern template std::vector<View<example::Order>> Snapshot::view_by_scan_field<&example::Order::qty>(const std::int64_t&) const;
extern template std::vector<const example::Order*> Snapshot::find_by_cached_field<&example::Order::qty>(const std::int64_t&) const;
extern template std::vector<const example::Order*> Snapshot::find_by_cached_field<&example::Order::computed_key>(const std::string&) const;
extern template std::vector<View<example::Order>> Snapshot::view_by_cached_field<&example::Order::qty>(const std::int64_t&) const;
extern template std::vector<const example::Order*> Snapshot::find_referrers<&example::Order::account>(Ref<example::Account>) const;
extern template std::vector<View<example::Order>> Snapshot::find_referrers_view<&example::Order::account>(Ref<example::Account>) const;
extern template std::vector<const example::Order*> Snapshot::find_cached_referrers<&example::Order::account>(Ref<example::Account>) const;
extern template std::vector<View<example::Order>> Snapshot::view_cached_referrers<&example::Order::account>(Ref<example::Account>) const;
extern template View<example::Order> Snapshot::view<example::Order>(const example::Order&) const noexcept;
extern template std::optional<View<example::Order>> Snapshot::view<example::Order>(Ref<example::Order>) const;

// ---- Transaction: Account ----
extern template Ref<example::Account> Transaction::create<example::Account>(std::unique_ptr<example::Account>);
extern template example::Account* Transaction::update<example::Account>(Ref<example::Account>);
extern template void Transaction::remove<example::Account>(Ref<example::Account>);
extern template bool Transaction::exists<example::Account>(Ref<example::Account>) const;
extern template const example::Account* Transaction::peek<example::Account>(Ref<example::Account>) const;
extern template const example::Account* Transaction::peek_as<example::Account>(Id) const;
extern template const example::Account* Transaction::peek_before<example::Account>(Id) const;

// ---- Transaction: Order ----
extern template Ref<example::Order> Transaction::create<example::Order>(std::unique_ptr<example::Order>);
extern template example::Order* Transaction::update<example::Order>(Ref<example::Order>);
extern template void Transaction::remove<example::Order>(Ref<example::Order>);
extern template void Transaction::remove<example::Order>(Opt<example::Order>);
extern template bool Transaction::exists<example::Order>(Ref<example::Order>) const;
extern template const example::Order* Transaction::peek<example::Order>(Ref<example::Order>) const;
extern template const example::Order* Transaction::peek_as<example::Order>(Id) const;
extern template const example::Order* Transaction::peek_before<example::Order>(Id) const;

// ---- Model ----
extern template const example::Account* Model::peek_as<example::Account>(Id) const;
extern template const example::Order* Model::peek_as<example::Order>(Id) const;
extern template LookupCounts Model::lookup_stats<&example::Order::qty>() const;

// ---- BulkTransaction ----
extern template Ref<example::Account> BulkTransaction::create<example::Account>(std::unique_ptr<example::Account>);
extern template Ref<example::Order> BulkTransaction::create<example::Order>(std::unique_ptr<example::Order>);
extern template example::Account* BulkTransaction::update<example::Account>(Ref<example::Account>);
extern template example::Order* BulkTransaction::update<example::Order>(Ref<example::Order>);

// ---- View ----
extern template View<example::Account> View<example::Order>::operator[]<example::Account>(Ref<example::Account> example::Order::*) const noexcept;
extern template std::optional<View<example::Order>> View<example::Order>::operator[]<example::Order>(Opt<example::Order> example::Order::*) const noexcept;
extern template std::vector<View<example::Order>> View<example::Account>::find_referrers<&example::Order::account>() const;

}  // namespace model
