#include "demo/types.h"
#include <cassert>
#include <chrono>
#include <cstdio>
#include <memory>
#include <string>
#include <thread>
#include <vector>
using namespace model; using namespace demo;

// Commit latency vs model size: should stay flat, since apply is proportional
// to the changeset, not to n_objects (the persistent secondary index derives
// in O(log32 n), and the reverse index is per-target, not a full scan).
double bench_latency(int n_objects) {
    Model m;
    Transaction seed = m.begin();
    std::vector<Ref<Account>> accts;
    for (int k = 0; k < 2000; ++k) {
        auto a = std::make_unique<Account>();
        a->name = "A" + std::to_string(k);
        accts.push_back(seed.create(std::move(a)));
    }
    const CommitResult seed_res = m.try_commit(seed);
    for (auto& r : accts) r = seed_res.to_real(r);  // local ids -> real, post-commit

    int id = 0;
    for (int i = 0; i < n_objects; i += 1000) {
        Transaction txn = m.begin();
        for (int j = 0; j < 1000 && i + j < n_objects; ++j) {
            auto o = std::make_unique<Order>();
            o->code = "O" + std::to_string(id++);
            o->account = accts[(i + j) % 2000];
            txn.create(std::move(o));
        }
        m.try_commit(txn);
    }

    Snapshot s = m.snapshot();
    std::vector<Ref<Order>> some;
    {
        int c = 0;
        s.for_each<Order>([&](const Order& o) {
            if (c++ < 10) some.push_back(Ref<Order>(o.id));
        });
    }

    auto t0 = std::chrono::steady_clock::now();
    const int iters = 200;
    for (int it = 0; it < iters; ++it) {
        Transaction txn = m.begin();
        for (auto r : some)
            if (auto* o = txn.update(r)) o->qty = it;
        m.try_commit(txn);
    }
    auto t1 = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::micro>(t1 - t0).count() / iters;
}

// try_commit() throughput vs. concurrent writer-thread count. commit_mu_
// fully serializes the apply step model-wide -- transaction *building* is
// genuinely parallel, but *applying* is not. This sweep exists to show that
// plainly: throughput should NOT scale linearly with thread count. That is
// the documented scope boundary (see CLAUDE.md), not a bug to chase.
struct ThroughputResult {
    int threads;
    double commits_per_sec;
};

ThroughputResult bench_throughput(int n_threads, int commits_per_thread) {
    Model m;
    // Each thread updates its own disjoint account, so conflicts are rare and
    // this measures commit_mu_'s serialization, not conflict-retry overhead.
    Transaction seed = m.begin();
    std::vector<Ref<Account>> accts;
    for (int i = 0; i < n_threads; ++i) {
        auto a = std::make_unique<Account>();
        a->name = "T" + std::to_string(i);
        accts.push_back(seed.create(std::move(a)));
    }
    const CommitResult seed_res = m.try_commit(seed);
    for (auto& r : accts) r = seed_res.to_real(r);  // local ids -> real, post-commit

    auto t0 = std::chrono::steady_clock::now();
    std::vector<std::thread> workers;
    for (int t = 0; t < n_threads; ++t) {
        workers.emplace_back([&, t] {
            for (int i = 0; i < commits_per_thread; ++i) {
                Transaction txn = m.begin();
                if (Account* a = txn.update(accts[static_cast<std::size_t>(t)])) a->balance = i;
                m.try_commit(txn);
            }
        });
    }
    for (auto& w : workers) w.join();
    auto t1 = std::chrono::steady_clock::now();

    const double secs = std::chrono::duration<double>(t1 - t0).count();
    const double total_commits = static_cast<double>(n_threads) * commits_per_thread;
    return {n_threads, total_commits / secs};
}

int main() {
    for (int n : {10000, 100000, 500000}) {
        double us = bench_latency(n);
        std::printf("n=%7d objects:  %8.1f us per (10-mutation txn + try_commit)\n", n, us);
    }
    std::printf("\ncommit latency should stay roughly flat across n (changeset-proportional).\n");

    std::printf("\ntry_commit throughput vs. writer-thread count (commit_mu_ serializes apply):\n");
    for (int threads : {1, 2, 4, 8}) {
        ThroughputResult r = bench_throughput(threads, 2000);
        std::printf("threads=%2d:  %9.1f commits/sec\n", r.threads, r.commits_per_sec);
    }
    std::printf(
        "\nExpect sub-linear scaling -- apply is fully serialized, so more writer threads "
        "means more contention on commit_mu_, not more parallel apply work.\n");
    return 0;
}
