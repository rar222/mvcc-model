// Whether Node::slots' vector (persistent_map.h) tracks actual trie
// occupancy at steady state, or gets padded by libstdc++'s generic
// growth-doubling -- backed by Model::slot_stats_diagnostics().
//
// Three scenarios, same N one-object Account-creating transactions:
//   A) single-threaded, one commit per object: Model::root_ always retains
//      the just-published Root, so the writer's own by_type_/by_key_ top
//      node is co-owned (use_count()>=2) at the START of every commit --
//      set_in's mutate-in-place fast path never applies here, regardless of
//      reader activity (verified: scenario B below is byte-for-byte
//      identical to A).
//   B) like A, plus a background thread hammering Model::snapshot() in a
//      tight loop throughout -- a control showing reader concurrency is NOT
//      what forces cloning in A; publish_now() retaining root_ already does.
//      Off by default (see run_b below); matches A exactly when enabled.
//   C) everything in ONE transaction/ONE commit: every intermediate Node the
//      apply phase touches is private to that one apply step, so the
//      mutate-in-place fast path applies on every revisit -- the tight-
//      packing control.
//
// At this project's target scale (100k-1M objects, sequential Id::index,
// constant Id::gen), by_type_'s trie shape saturates by roughly N=100,000
// and then just fills toward 32/32 per node as N climbs to 32^4=1,048,576 --
// so scenario A's vector, doubling on every clone, ends up LARGER than a
// hypothetical fixed 32-slot array would be (see the printed vector_buf vs.
// fixed32_buf columns). by_key_ (string-keyed, less saturated at this scale)
// stays a clear win for the vector in every scenario.

#include "example/types.h"
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <thread>

using namespace model;
using namespace example;

namespace {

void report(const char* label, const Model::Diagnostics::SlotStats& d) {
    auto line = [](const char* name, const model::pmap::SlotStats& s) {
        const double waste_bytes = static_cast<double>(s.total_capacity - s.total_size) * 16.0;
        const double fixed32_bytes = static_cast<double>(s.node_count) * 32.0 * 16.0;
        const double vector_bytes = static_cast<double>(s.total_capacity) * 16.0;
        std::printf(
            "  %-8s nodes=%8zu leaves=%8zu  size(sum)=%9zu  capacity(sum)=%9zu  "
            "avg_fill=%.1f%%  vector_buf=%.2fMB  fixed32_buf=%.2fMB  slack=%.2fMB\n",
            name, s.node_count, s.leaf_count, s.total_size, s.total_capacity,
            s.total_size ? 100.0 * static_cast<double>(s.total_size) / static_cast<double>(s.total_capacity)
                         : 0.0,
            vector_bytes / 1e6, fixed32_bytes / 1e6, waste_bytes / 1e6);
    };
    std::printf("%s\n", label);
    line("by_type", d.by_type);
    line("by_key", d.by_key);
}

void build_single_threaded(Model& m, int n) {
    m.set_max_undo_list_size(0);
    for (int i = 0; i < n; ++i) {
        Transaction txn = m.begin();
        auto a = std::make_unique<Account>();
        a->name = "Acct" + std::to_string(i);
        txn.create(std::move(a));
        m.try_commit(txn);
    }
}

// Everything in ONE transaction/ONE commit: every intermediate Node the
// apply phase touches is private to this one apply step (nothing has been
// published yet to co-own it), so set_in's mutate-in-place fast path should
// apply on every revisit of the same node -- the control for whether
// "always-clone" (A and B above) is really what's driving the observed
// fill ratio, or whether it's something else.
void build_single_transaction(Model& m, int n) {
    m.set_max_undo_list_size(0);
    Transaction txn = m.begin();
    for (int i = 0; i < n; ++i) {
        auto a = std::make_unique<Account>();
        a->name = "Acct" + std::to_string(i);
        txn.create(std::move(a));
    }
    m.try_commit(txn);
}

void build_with_concurrent_readers(Model& m, int n) {
    m.set_max_undo_list_size(0);
    std::atomic<bool> stop{false};
    std::thread reader([&] {
        while (!stop.load(std::memory_order_relaxed)) {
            Snapshot s = m.snapshot();  // held just long enough to pin the current Root
            (void)s;
        }
    });
    for (int i = 0; i < n; ++i) {
        Transaction txn = m.begin();
        auto a = std::make_unique<Account>();
        a->name = "Acct" + std::to_string(i);
        txn.create(std::move(a));
        m.try_commit(txn);
    }
    stop.store(true, std::memory_order_relaxed);
    reader.join();
}

}  // namespace

int main(int argc, char** argv) {
    const int n = argc > 1 ? std::atoi(argv[1]) : 200'000;
    // Scenario B is byte-for-byte identical to A at 200k (see the
    // conversation this probe is for): Model::root_ keeps the just-
    // published Root alive independent of any reader, so the writer's own
    // by_type_/by_key_ top node is co-owned (use_count()>=2) on every
    // single-object-per-commit transaction regardless of concurrent
    // snapshot() activity. Skipping it at large N to save wall time; pass a
    // second argv to force it back on.
    const bool run_b = argc > 2;

    std::printf("N = %d Accounts (by_type: Id-keyed set; by_key: Account::name, string-keyed map)\n\n",
                n);

    {
        Model m;
        build_single_threaded(m, n);
        report("Scenario A -- single-threaded, no concurrent readers:", m.slot_stats_diagnostics());
    }
    if (run_b) {
        std::printf("\n");
        Model m;
        build_with_concurrent_readers(m, n);
        report("Scenario B -- background thread hammering snapshot() throughout:",
               m.slot_stats_diagnostics());
    }
    std::printf("\n");
    {
        Model m;
        build_single_transaction(m, n);
        report("Scenario C -- one transaction, one commit (control: nothing published mid-build):",
               m.slot_stats_diagnostics());
    }
    return 0;
}
