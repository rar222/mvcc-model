// Per-commit latency of many SMALL (one-object) transactions, each its own
// try_commit() -- the target-scale workload per CLAUDE.md ("Commits: 10-100
// / sec, from any number of writer threads"), as opposed to
// large_txn_bench.cpp's one-huge-transaction scenarios. Written to answer a
// specific question: do the apply_transaction_contents() optimizations in
// that other file (first-touch-only undo logging for the four whole-map-
// handle indexes, splitting validate_field_key_uniqueness's collection out,
// batching remove_raw's cascade BFS across remove() intents) cost anything
// on the SMALL-transaction path, where there's only ever one thing to
// dedupe against -- i.e. where the "first touch" IS the only touch?
//
// Each scenario commits N separate one-object transactions and reports
// total wall time / N -- an average per-commit latency, not a throughput
// figure (single-threaded, so it isolates apply cost the same way
// large_txn_bench.cpp's bulk scenarios do, just at N=1 object per commit
// instead of N objects in one commit).

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

// N separate transactions, each creating exactly ONE Order.
double bench_small_creates(int n_txns) {
    Model m;
    // Uncapped (the default) makes publish_now()'s undo_list_ conflict-prune
    // loop O(entries retained so far) on EVERY commit -- fine for a handful
    // of commits, but with thousands of separate small transactions here it
    // would dominate the measurement with an unrelated, pre-existing
    // O(n^2)-ish cost that has nothing to do with what this file is
    // isolating. Capped at 0, that loop always runs over an empty list.
    m.set_max_undo_list_size(0);
    auto accts = seed_accounts(m, 50);

    const auto t0 = Clock::now();
    for (int i = 0; i < n_txns; ++i) {
        Transaction txn = m.begin();
        auto o = std::make_unique<Order>();
        o->code = "O" + std::to_string(i);
        o->account = accts[static_cast<std::size_t>(i) % accts.size()];
        o->qty = i % 50;
        txn.create(std::move(o));
        const CommitResult res = m.try_commit(txn);
        if (res.status != CommitStatus::Committed) std::printf("  UNEXPECTED FAIL at i=%d\n", i);
    }
    const double total_ms = ms_since(t0);
    std::printf("  small_creates n=%d: total=%.1f ms  (%.3f us/commit)\n", n_txns, total_ms,
               total_ms * 1000.0 / n_txns);
    return total_ms;
}

// Seed N Orders, committed one per transaction; then N MORE separate
// transactions, each updating exactly ONE pre-existing Order.
double bench_small_updates(int n_txns) {
    Model m;
    // Uncapped (the default) makes publish_now()'s undo_list_ conflict-prune
    // loop O(entries retained so far) on EVERY commit -- fine for a handful
    // of commits, but with thousands of separate small transactions here it
    // would dominate the measurement with an unrelated, pre-existing
    // O(n^2)-ish cost that has nothing to do with what this file is
    // isolating. Capped at 0, that loop always runs over an empty list.
    m.set_max_undo_list_size(0);
    auto accts = seed_accounts(m, 50);
    std::vector<Ref<Order>> orders;
    orders.reserve(static_cast<std::size_t>(n_txns));
    for (int i = 0; i < n_txns; ++i) {
        Transaction txn = m.begin();
        auto o = std::make_unique<Order>();
        o->code = "O" + std::to_string(i);
        o->account = accts[static_cast<std::size_t>(i) % accts.size()];
        o->qty = i % 50;
        const Ref<Order> local = txn.create(std::move(o));
        const CommitResult res = m.try_commit(txn);
        orders.push_back(res.to_real(local));
    }

    const auto t0 = Clock::now();
    for (int i = 0; i < n_txns; ++i) {
        Transaction txn = m.begin();
        if (auto* o = txn.update(orders[static_cast<std::size_t>(i)])) o->qty += 1000;
        const CommitResult res = m.try_commit(txn);
        if (res.status != CommitStatus::Committed) std::printf("  UNEXPECTED FAIL at i=%d\n", i);
    }
    const double total_ms = ms_since(t0);
    std::printf("  small_updates n=%d: total=%.1f ms  (%.3f us/commit)\n", n_txns, total_ms,
               total_ms * 1000.0 / n_txns);
    return total_ms;
}

// Seed N Orders (parent=null, nothing references any of them, so removing
// one is a single-victim cascade of size 1 -- exactly one remove() intent
// per transaction, the shape that isolates remove_raw's single-seed cost).
double bench_small_removes(int n_txns) {
    Model m;
    // Uncapped (the default) makes publish_now()'s undo_list_ conflict-prune
    // loop O(entries retained so far) on EVERY commit -- fine for a handful
    // of commits, but with thousands of separate small transactions here it
    // would dominate the measurement with an unrelated, pre-existing
    // O(n^2)-ish cost that has nothing to do with what this file is
    // isolating. Capped at 0, that loop always runs over an empty list.
    m.set_max_undo_list_size(0);
    auto accts = seed_accounts(m, 50);
    std::vector<Ref<Order>> orders;
    orders.reserve(static_cast<std::size_t>(n_txns));
    for (int i = 0; i < n_txns; ++i) {
        Transaction txn = m.begin();
        auto o = std::make_unique<Order>();
        o->code = "O" + std::to_string(i);
        o->account = accts[static_cast<std::size_t>(i) % accts.size()];
        const Ref<Order> local = txn.create(std::move(o));
        const CommitResult res = m.try_commit(txn);
        orders.push_back(res.to_real(local));
    }

    const auto t0 = Clock::now();
    for (int i = 0; i < n_txns; ++i) {
        Transaction txn = m.begin();
        txn.remove(orders[static_cast<std::size_t>(i)]);
        const CommitResult res = m.try_commit(txn);
        if (res.status != CommitStatus::Committed) std::printf("  UNEXPECTED FAIL at i=%d\n", i);
    }
    const double total_ms = ms_since(t0);
    std::printf("  small_removes n=%d: total=%.1f ms  (%.3f us/commit)\n", n_txns, total_ms,
               total_ms * 1000.0 / n_txns);
    return total_ms;
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
    constexpr int kTxns = 20000;

    std::printf("=== small_creates: %d separate transactions, 1 Order each ===\n", kTxns);
    const double create_ms = best_of(kRepeats, [] { return bench_small_creates(kTxns); });

    std::printf("\n=== small_updates: %d separate transactions, 1 pre-existing Order each ===\n",
               kTxns);
    const double update_ms = best_of(kRepeats, [] { return bench_small_updates(kTxns); });

    std::printf(
        "\n=== small_removes: %d separate transactions, 1 unreferenced Order each ===\n", kTxns);
    const double remove_ms = best_of(kRepeats, [] { return bench_small_removes(kTxns); });

    std::printf("\n=== summary (best of %d, total wall time for %d single-object commits) ===\n",
               kRepeats, kTxns);
    std::printf("  small_creates:   %9.1f ms  (%.3f us/commit)\n", create_ms,
               create_ms * 1000.0 / kTxns);
    std::printf("  small_updates:   %9.1f ms  (%.3f us/commit)\n", update_ms,
               update_ms * 1000.0 / kTxns);
    std::printf("  small_removes:   %9.1f ms  (%.3f us/commit)\n", remove_ms,
               remove_ms * 1000.0 / kTxns);
    return 0;
}
