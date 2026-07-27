#include "example/types_extern.h"

// The one place that actually instantiates the ~50 Snapshot/Transaction/
// Model/BulkTransaction/View entry points declared `extern template` in
// types_extern.h, for Account and Order. Every other TU that includes
// types_extern.h links against these definitions instead of reinstantiating
// (and, at -O2, re-optimizing) its own copy. See types_extern.h's own
// comment for why the set is shaped the way it is.

namespace model {

// ---- Object<T> ----
template class Object<example::Account>;
template class Object<example::Order>;

// ---- Snapshot: Account ----
template const example::Account& Snapshot::resolve<example::Account>(Ref<example::Account>) const noexcept;
template const example::Account* Snapshot::find<example::Account>(Ref<example::Account>) const noexcept;
template const example::Account* Snapshot::find_by_key<&example::Account::name>(const std::string&) const;
template std::optional<View<example::Account>> Snapshot::view_by_key<&example::Account::name>(const std::string&) const;
template std::vector<const example::Account*> Snapshot::find_by_scan_field<&example::Account::name>(const std::string&) const;
template std::vector<View<example::Account>> Snapshot::view_by_scan_field<&example::Account::name>(const std::string&) const;
template View<example::Account> Snapshot::view<example::Account>(const example::Account&) const noexcept;
template std::optional<View<example::Account>> Snapshot::view<example::Account>(Ref<example::Account>) const;

// ---- Snapshot: Order ----
template const example::Order* Snapshot::resolve<example::Order>(Opt<example::Order>) const noexcept;
template const example::Order* Snapshot::find<example::Order>(Ref<example::Order>) const noexcept;
template const example::Order* Snapshot::find<example::Order>(Opt<example::Order>) const noexcept;
template const example::Order* Snapshot::find_by_key<&example::Order::computed_key>(const std::string&) const;
template std::optional<View<example::Order>> Snapshot::view_by_key<&example::Order::computed_key>(const std::string&) const;
template std::vector<const example::Order*> Snapshot::find_by_scan_field<&example::Order::qty>(const std::int64_t&) const;
template std::vector<const example::Order*> Snapshot::find_by_scan_field<&example::Order::computed_key>(const std::string&) const;
template std::vector<View<example::Order>> Snapshot::view_by_scan_field<&example::Order::qty>(const std::int64_t&) const;
template std::vector<const example::Order*> Snapshot::find_by_cached_field<&example::Order::qty>(const std::int64_t&) const;
template std::vector<const example::Order*> Snapshot::find_by_cached_field<&example::Order::computed_key>(const std::string&) const;
template std::vector<View<example::Order>> Snapshot::view_by_cached_field<&example::Order::qty>(const std::int64_t&) const;
template std::vector<const example::Order*> Snapshot::find_referrers<&example::Order::account>(Ref<example::Account>) const;
template std::vector<View<example::Order>> Snapshot::find_referrers_view<&example::Order::account>(Ref<example::Account>) const;
template std::vector<const example::Order*> Snapshot::find_cached_referrers<&example::Order::account>(Ref<example::Account>) const;
template std::vector<View<example::Order>> Snapshot::view_cached_referrers<&example::Order::account>(Ref<example::Account>) const;
template View<example::Order> Snapshot::view<example::Order>(const example::Order&) const noexcept;
template std::optional<View<example::Order>> Snapshot::view<example::Order>(Ref<example::Order>) const;

// ---- Transaction: Account ----
template Ref<example::Account> Transaction::create<example::Account>(std::unique_ptr<example::Account>);
template example::Account* Transaction::update<example::Account>(Ref<example::Account>);
template void Transaction::remove<example::Account>(Ref<example::Account>);
template bool Transaction::exists<example::Account>(Ref<example::Account>) const;
template const example::Account* Transaction::peek<example::Account>(Ref<example::Account>) const;
template const example::Account* Transaction::peek_as<example::Account>(Id) const;
template const example::Account* Transaction::peek_before<example::Account>(Id) const;

// ---- Transaction: Order ----
template Ref<example::Order> Transaction::create<example::Order>(std::unique_ptr<example::Order>);
template example::Order* Transaction::update<example::Order>(Ref<example::Order>);
template void Transaction::remove<example::Order>(Ref<example::Order>);
template void Transaction::remove<example::Order>(Opt<example::Order>);
template bool Transaction::exists<example::Order>(Ref<example::Order>) const;
template const example::Order* Transaction::peek<example::Order>(Ref<example::Order>) const;
template const example::Order* Transaction::peek_as<example::Order>(Id) const;
template const example::Order* Transaction::peek_before<example::Order>(Id) const;

// ---- Model ----
template const example::Account* Model::peek_as<example::Account>(Id) const;
template const example::Order* Model::peek_as<example::Order>(Id) const;
template LookupCounts Model::lookup_stats<&example::Order::qty>() const;

// ---- BulkTransaction ----
template Ref<example::Account> BulkTransaction::create<example::Account>(std::unique_ptr<example::Account>);
template Ref<example::Order> BulkTransaction::create<example::Order>(std::unique_ptr<example::Order>);
template example::Account* BulkTransaction::update<example::Account>(Ref<example::Account>);
template example::Order* BulkTransaction::update<example::Order>(Ref<example::Order>);

// ---- View ----
template View<example::Account> View<example::Order>::operator[]<example::Account>(Ref<example::Account> example::Order::*) const noexcept;
template std::optional<View<example::Order>> View<example::Order>::operator[]<example::Order>(Opt<example::Order> example::Order::*) const noexcept;
template std::vector<View<example::Order>> View<example::Account>::find_referrers<&example::Order::account>() const;

}  // namespace model
