// Large single-transaction benchmark: measures ONLY the try_commit() apply
// phase for one huge Transaction, isolating the per-object index-maintenance
// cost inside apply_transaction_contents() from Transaction-building cost
// (which CLAUDE.md/single_vs_multi_txn_bench.cpp already show is cheap and
// fully parallel -- the bottleneck is apply, which is serialized behind
// commit_mu_, invariant 7).
//
// Order (example/types.h) is used rather than Account because it touches
// every commit_mu_-protected index apply_transaction_contents maintains:
// by_type_, by_field_ (computed_key), by_cached_field_ (qty, computed_key),
// by_cached_reference_ (account), and referrers_ (account: non-nullable,
// parent: nullable) -- so a bulk create/update/remove here exercises the
// same per-object work a large real-world transaction would.
//
// Three scenarios, each timing only the try_commit() call:
//   bulk_create           -- one Transaction, N fresh Orders.
//   bulk_update           -- one Transaction, N pre-existing Orders all
//                             updated (qty changes -> reconcile_* on every
//                             index for every object).
//   bulk_cascading_remove -- one Transaction, N SEPARATE remove() intents
//                             (one per Account), each cascading into its own
//                             Orders via the non-nullable account Ref<>.

#include "example/types.h"
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <memory>
#include <string>
#include <vector>

using namespace model;
using namespace example;
using Clock = std::chrono::steady_clock;

namespace {

double ms_since(Clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

std::vector<Ref<Account>> seed_accounts(Model& m, int n_accounts) {
    Transaction txn = m.begin();
    std::vector<Ref<Account>> out;
    out.reserve(static_cast<std::size_t>(n_accounts));
    for (int i = 0; i < n_accounts; ++i) {
        auto a = std::make_unique<Account>();
        a->name = "Acct" + std::to_string(i);
        out.push_back(txn.create(std::move(a)));
    }
    const CommitResult res = m.try_commit(txn);
    for (auto& r : out) r = res.to_real(r);
    return out;
}

double bench_bulk_create(int n_orders, int n_accounts) {
    Model m;
    auto accts = seed_accounts(m, n_accounts);

    Transaction txn = m.begin();
    for (int i = 0; i < n_orders; ++i) {
        auto o = std::make_unique<Order>();
        o->code = "O" + std::to_string(i);
        o->account = accts[static_cast<std::size_t>(i) % accts.size()];
        o->qty = i % 50;
        txn.create(std::move(o));
    }

    const auto t0 = Clock::now();
    const CommitResult res = m.try_commit(txn);
    const double commit_ms = ms_since(t0);
    std::printf("  bulk_create n=%d: commit=%.1f ms status=%s\n", n_orders, commit_ms,
               res.status == CommitStatus::Committed ? "OK" : "FAIL");
    return commit_ms;
}

double bench_bulk_update(int n_orders, int n_accounts) {
    Model m;
    auto accts = seed_accounts(m, n_accounts);

    std::vector<Ref<Order>> orders;
    orders.reserve(static_cast<std::size_t>(n_orders));
    {
        Transaction seed = m.begin();
        for (int i = 0; i < n_orders; ++i) {
            auto o = std::make_unique<Order>();
            o->code = "O" + std::to_string(i);
            o->account = accts[static_cast<std::size_t>(i) % accts.size()];
            o->qty = i % 50;
            orders.push_back(seed.create(std::move(o)));
        }
        const CommitResult res = m.try_commit(seed);
        for (auto& r : orders) r = res.to_real(r);
    }

    Transaction txn = m.begin();
    for (auto r : orders)
        if (auto* o = txn.update(r)) o->qty += 1000;  // moves every object's cached-field buckets

    const auto t0 = Clock::now();
    const CommitResult res = m.try_commit(txn);
    const double commit_ms = ms_since(t0);
    std::printf("  bulk_update n=%d: commit=%.1f ms status=%s\n", n_orders, commit_ms,
               res.status == CommitStatus::Committed ? "OK" : "FAIL");
    return commit_ms;
}

double bench_bulk_cascading_remove(int n_accounts, int orders_per_account) {
    Model m;
    auto accts = seed_accounts(m, n_accounts);
    {
        Transaction seed = m.begin();
        int id = 0;
        for (auto acct : accts) {
            for (int j = 0; j < orders_per_account; ++j) {
                auto o = std::make_unique<Order>();
                o->code = "O" + std::to_string(id++);
                o->account = acct;
                seed.create(std::move(o));
            }
        }
        m.try_commit(seed);
    }

    Transaction txn = m.begin();
    for (auto acct : accts) txn.remove(acct);  // n_accounts SEPARATE remove() intents

    const auto t0 = Clock::now();
    const CommitResult res = m.try_commit(txn);
    const double commit_ms = ms_since(t0);
    std::printf(
        "  bulk_cascading_remove accounts=%d orders_each=%d (total removed=%zu): commit=%.1f ms "
        "status=%s\n",
        n_accounts, orders_per_account, res.changes.size(), commit_ms,
        res.status == CommitStatus::Committed ? "OK" : "FAIL");
    return commit_ms;
}

template <class Fn>
double best_of(int repeats, Fn&& fn) {
    double best = 1e18;
    for (int r = 0; r < repeats; ++r) best = std::min(best, fn());
    return best;
}

}  // namespace

int main() {
    constexpr int kRepeats = 3;
    constexpr int kOrders = 150000;
    constexpr int kAccounts = 2000;
    constexpr int kCascadeAccounts = 3000;
    constexpr int kOrdersPerAccount = 20;

    std::printf("=== bulk_create: 1 transaction, %d Orders (ref+keyed+cached) ===\n", kOrders);
    const double create_ms = best_of(kRepeats, [] { return bench_bulk_create(kOrders, kAccounts); });

    std::printf("\n=== bulk_update: 1 transaction, %d pre-existing Orders, all updated ===\n",
               kOrders);
    const double update_ms = best_of(kRepeats, [] { return bench_bulk_update(kOrders, kAccounts); });

    std::printf(
        "\n=== bulk_cascading_remove: 1 transaction, %d separate remove() intents, each "
        "cascading into %d Orders ===\n",
        kCascadeAccounts, kOrdersPerAccount);
    const double remove_ms = best_of(
        kRepeats, [] { return bench_bulk_cascading_remove(kCascadeAccounts, kOrdersPerAccount); });

    std::printf("\n=== summary (best of %d, commit-phase only) ===\n", kRepeats);
    std::printf("  bulk_create           (%6d orders):            %9.1f ms\n", kOrders, create_ms);
    std::printf("  bulk_update           (%6d orders):            %9.1f ms\n", kOrders, update_ms);
    std::printf("  bulk_cascading_remove (%dx%d = %6d objects):    %9.1f ms\n", kCascadeAccounts,
               kOrdersPerAccount, kCascadeAccounts * (1 + kOrdersPerAccount), remove_ms);
    return 0;
}
