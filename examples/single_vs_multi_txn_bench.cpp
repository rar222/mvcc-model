// Does creating 1,000,000 objects go faster as one single-threaded
// Transaction, or as 4 threads each building and committing their own
// 250,000-object Transaction?
//
// The intuitive guess is "4 threads should parallelize the work" -- but
// measure it (default preset, RelWithDebInfo -O2, asserts on) and the
// single transaction wins:
//
//   1 thread,  1 Transaction,  1,000,000 creates:            ~18.1 s
//   4 threads, 250,000 creates each (own Transaction):       ~21.3 s
//
// Why: per-thread BUILD time is tiny (tens to ~150 ms) next to per-thread
// COMMIT time (single-digit to twenty-plus seconds, wildly uneven across
// the 4 threads). That imbalance is the tell -- almost the entire cost is
// in try_commit()'s apply phase, and per CLAUDE.md's invariant 7, apply is
// FULLY SERIALIZED behind commit_mu_: only one thread's apply runs at a
// time, no matter how many threads are calling try_commit() concurrently.
// Transaction *building* is the part that's genuinely parallel (invariant
// 10: building never takes commit_mu_ at all), but building 250k plain
// objects is cheap -- a string format plus a vector push -- so there is
// almost nothing to parallelize. Both scenarios pay for 1,000,000
// serialized apply-steps; the 4-way split just adds overhead the single
// transaction doesn't have (4 separate commit_mu_ acquisitions, 4 separate
// changelog/version-bump/publish sequences, and threads blocking on each
// other's apply) without adding any parallelism where the actual
// bottleneck is.
//
// This matches commit_bench.cpp's own documented finding (bench_throughput):
// "expect sub-linear scaling, not linear" as writer-thread count increases.
// Splitting one workload across threads doesn't just fail to scale here --
// it goes slightly negative, since the split adds coordination cost without
// touching the real bottleneck.
//
// For bulk creation where this actually matters, Model::begin_bulk()/
// commit_bulk_without_undo() is the documented escape hatch: no undo log,
// no per-object commit_mu_-protected bookkeeping. It requires exclusive
// access (no concurrent readers/writers) and wipes the whole model, so
// it's for initial load, not incremental writes -- see its own doc comment
// in model.h.
//
// Each scenario also runs again with set_max_undo_list_size(0). Measured
// result: it makes essentially no difference here --
//
//   single-txn, uncapped:   17726.4 ms
//   multi-txn,  uncapped:   20812.7 ms
//   single-txn, capped=0:   16832.5 ms
//   multi-txn,  capped=0:   20970.3 ms
//   ratio uncapped/capped=0, single-txn:  1.05x
//   ratio uncapped/capped=0, multi-txn:   0.99x
//
// -- which is exactly what set_max_undo_list_size()'s own doc comment
// predicts: n == 0 only skips APPENDING the finished UndoEntry to
// undo_list_; the per-attempt pending_undo_ capture during apply
// (collapse_undo_actions()) still happens exactly the same either way --
// that's a separate, earlier step gated by try_commit() vs.
// try_commit_without_undo(), not by this cap. With only 1 commit (scenario
// A) or 4 commits (scenario B) total, undo_list_ never grows large enough
// for its trim-from-the-front cost (set_max_undo_list_size()'s `while
// (undo_list_.size() >= max_undo_list_size_) erase(begin())` loop) to
// matter regardless of the cap. Contrast performance_tests.cpp's own
// four_threads_updating_private_objects_with_undo_list_capped_at_0_is_
// never_slower_than_uncapped, which stresses MANY repeated small commits
// instead of a few huge ones -- THAT's the shape where retention (and the
// repeated trim) actually costs something; a handful of huge transactions
// never accumulates enough undo history for the cap to matter.

#include "example/types.h"
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <memory>
#include <string>
#include <thread>
#include <vector>

using namespace model;
using namespace example;

using Clock = std::chrono::steady_clock;
namespace {
double ms_since(Clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

// Scenario A: one thread, one Transaction, n creates, one try_commit().
// `cap_undo_to_zero`: see the file header comment -- set_max_undo_list_size(0)
// skips RETAINING the commit's UndoEntry, not collecting it during apply.
double bench_single_txn(int n, bool cap_undo_to_zero) {
    Model m;
    if (cap_undo_to_zero) m.set_max_undo_list_size(0);
    auto t0 = Clock::now();
    Transaction txn = m.begin();
    for (int i = 0; i < n; ++i) {
        auto a = std::make_unique<Account>();
        a->name = "A" + std::to_string(i);
        a->balance = i;
        txn.create(std::move(a));
    }
    const double build_ms = ms_since(t0);
    auto t1 = Clock::now();
    const CommitResult res = m.try_commit(txn);
    const double commit_ms = ms_since(t1);
    std::printf("  [single] build=%.1f ms  commit=%.1f ms  total=%.1f ms  status=%s  size=%zu\n",
               build_ms, commit_ms, build_ms + commit_ms,
               res.status == CommitStatus::Committed ? "OK" : "FAIL", m.snapshot().size());
    return build_ms + commit_ms;
}

// Scenario B: n_threads threads, each builds + commits its OWN Transaction
// of items_per_thread creates. Names are prefixed per-thread so no two
// threads' Accounts collide on the define_keys() name field (a duplicate
// there is a rejected commit, not an overwrite -- see Account's own doc
// comment in example/types.h). `cap_undo_to_zero`: see bench_single_txn.
double bench_multi_txn(int n_threads, int items_per_thread, bool cap_undo_to_zero) {
    Model m;
    if (cap_undo_to_zero) m.set_max_undo_list_size(0);
    std::vector<double> build_ms(static_cast<std::size_t>(n_threads), 0.0);
    std::vector<double> commit_ms(static_cast<std::size_t>(n_threads), 0.0);
    std::vector<CommitStatus> status(static_cast<std::size_t>(n_threads));

    auto t0 = Clock::now();
    std::vector<std::thread> workers;
    for (int t = 0; t < n_threads; ++t) {
        workers.emplace_back([&, t] {
            auto tb0 = Clock::now();
            Transaction txn = m.begin();
            for (int i = 0; i < items_per_thread; ++i) {
                auto a = std::make_unique<Account>();
                a->name = "T" + std::to_string(t) + "_" + std::to_string(i);
                a->balance = i;
                txn.create(std::move(a));
            }
            build_ms[static_cast<std::size_t>(t)] = ms_since(tb0);
            auto tc0 = Clock::now();
            const CommitResult res = m.try_commit(txn);
            commit_ms[static_cast<std::size_t>(t)] = ms_since(tc0);
            status[static_cast<std::size_t>(t)] = res.status;
        });
    }
    for (auto& w : workers) w.join();
    const double wall_ms = ms_since(t0);

    for (int t = 0; t < n_threads; ++t) {
        std::printf("  [thread %d] build=%.1f ms  commit=%.1f ms  status=%s\n", t,
                   build_ms[static_cast<std::size_t>(t)], commit_ms[static_cast<std::size_t>(t)],
                   status[static_cast<std::size_t>(t)] == CommitStatus::Committed ? "OK" : "FAIL");
    }
    std::printf("  [multi]  wall=%.1f ms  size=%zu\n", wall_ms, m.snapshot().size());
    return wall_ms;
}
}  // namespace

int main() {
    constexpr int kTotal = 1'000'000;
    constexpr int kThreads = 4;
    constexpr int kPerThread = kTotal / kThreads;
    constexpr int kRepeats = 3;

    struct Result {
        const char* label;
        double best_ms;
    };
    std::vector<Result> results;

    for (bool capped : {false, true}) {
        const char* mode = capped ? "undo list capped at 0" : "undo list uncapped (default)";

        std::printf("=== Scenario A (%s): 1 thread, 1 transaction, %d creates ===\n", mode,
                   kTotal);
        std::vector<double> a_totals;
        for (int r = 0; r < kRepeats; ++r) {
            std::printf(" run %d:\n", r);
            a_totals.push_back(bench_single_txn(kTotal, capped));
        }
        results.push_back({capped ? "single-txn, capped=0" : "single-txn, uncapped",
                           *std::min_element(a_totals.begin(), a_totals.end())});

        std::printf("\n=== Scenario B (%s): %d threads, %d creates each (own transaction) ===\n",
                   mode, kThreads, kPerThread);
        std::vector<double> b_totals;
        for (int r = 0; r < kRepeats; ++r) {
            std::printf(" run %d:\n", r);
            b_totals.push_back(bench_multi_txn(kThreads, kPerThread, capped));
        }
        results.push_back({capped ? "multi-txn,  capped=0" : "multi-txn,  uncapped",
                           *std::min_element(b_totals.begin(), b_totals.end())});
        std::printf("\n");
    }

    std::printf("=== summary (best of %d) ===\n", kRepeats);
    for (const Result& r : results) std::printf("  %s:  %8.1f ms\n", r.label, r.best_ms);
    std::printf("ratio single/multi, uncapped:  %.2fx\n", results[0].best_ms / results[1].best_ms);
    std::printf("ratio single/multi, capped=0:  %.2fx\n", results[2].best_ms / results[3].best_ms);
    std::printf("ratio uncapped/capped=0, single-txn:  %.2fx\n",
               results[0].best_ms / results[2].best_ms);
    std::printf("ratio uncapped/capped=0, multi-txn:   %.2fx\n",
               results[1].best_ms / results[3].best_ms);
    return 0;
}
