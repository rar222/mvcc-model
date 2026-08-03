// Two-writer demo: exactly two writer threads independently begin()/mutate/
// try_commit() transactions of ~10 changes each (creates, updates, deletes,
// mixed) against a small, deliberately shared pool of seed objects -- small
// enough, and 10 changes wide enough, that the two threads' write sets
// collide often. Some of those try_commit() calls are therefore expected to
// return Conflict, not just Committed; a third thread subscribes and prints
// every object in every batch it receives.
//
// The subscriber's s.resolve(ord.account) call is the real test: it runs on
// a thread that never touched the writers' transactions, against a Snapshot
// built by Model::apply()'s cascade/validate logic -- if a Ref ever dangled,
// that resolve() would fail its assert (or ASan would catch the underlying
// use-after-free). Run it under
//   cmake --preset asan
//   cmake --preset tsan

#include <atomic>
#include <cassert>
#include <cstdio>
#include <memory>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include "example/types.h"
#include "model/model.h"

using namespace model;
using namespace example;

namespace {

std::atomic<std::uint64_t> g_committed{0};
std::atomic<std::uint64_t> g_conflicts{0};

/// Pick a handle this transaction still believes is live, checked against the
/// transaction's OWN local view (base() plus its own pending edits so far) --
/// not some model-wide notion of existence, since there isn't one. The other
/// writer thread can still delete it out from under us before try_commit()
/// runs; that race is exactly what the conflict check exists to catch.
template <class T>
Ref<T> pick_live(const Transaction& txn, std::vector<Ref<T>>& v, std::mt19937& rng) {
    for (int tries = 0; tries < 12 && !v.empty(); ++tries) {
        const Ref<T> r = v[rng() % v.size()];
        if (txn.exists(r)) return r;
    }
    return Ref<T>{};
}

/// Prints one Change, resolved against the Snapshot pair that brackets it --
/// `before` is the PREVIOUS Update's snapshot (null for the very first
/// Update), `after` is this Change's own Update::snapshot. Structural only
/// (CHECK that the object is where this ChangeKind says it should be, plus
/// to_string() of whatever's found) -- the caller still owns any assertions
/// about specific field values, since those are scenario-specific and this
/// is generic over T.
template <class T>
void print_change(const Change& c, const Snapshot& before, const Snapshot& after) {
    const Ref<T> r{c.id};
    switch (c.kind) {
        case ChangeKind::Created: {
            const T* v = after.find(r);
            if (v) std::printf("create: after=%s\n", v->to_string().c_str());
            break;
        }
        case ChangeKind::Updated: {
            const T* b = before.find(r);
            const T* a = after.find(r);
            if (b && a)
                std::printf("update: before=%s after=%s\n", b->to_string().c_str(),
                            a->to_string().c_str());
            break;
        }
        case ChangeKind::Deleted: {
            const T* b = before.find(r);
            if (b) std::printf("delete: before=%s\n", b->to_string().c_str());
            break;
        }
    }
}

/// The third thread: just drains the subscription and prints what it sees.
/// Runs entirely off Update::snapshot -- it never calls Model::snapshot()
/// itself, so what it prints is exactly "current state plus everything that
/// changed since the last batch", the guarantee Subscription documents.
void subscriber_thread(std::shared_ptr<Subscription> sub) {
    std::uint64_t batch = 0;
    Snapshot prev;  // null until the first Update is drained
    while (auto u = sub->wait_for_update()) {
        ++batch;
        std::printf("[subscriber] batch %llu v%llu%s: %zu change(s)\n",
                    static_cast<unsigned long long>(batch),
                    static_cast<unsigned long long>(u->snapshot.version()),
                    u->coalesced ? " (coalesced)" : "", u->changes->size());

        for (const Change& c : *u->changes) {
            if (c.tag == type_tag<Account>()) {
                print_change<Account>(c, prev, u->snapshot);
            } else if (c.tag == type_tag<Order>()) {
                print_change<Order>(c, prev, u->snapshot);
            }
        }
        prev = u->snapshot;
    }
    std::printf("[subscriber] done: %llu batches\n", static_cast<unsigned long long>(batch));
}

/// One writer thread: loops begin() / ~10 random mutations / try_commit(),
/// retrying on Conflict, until it has landed `target_commits` successful
/// ones. Each thread owns its own copy of the seed id lists -- sharing a
/// std::vector across writer threads without synchronization would itself be
/// the bug this whole design exists to avoid at the object level.
void writer_thread(Model& m, int tid, std::vector<Ref<Account>> accounts,
                   std::vector<Ref<Order>> orders, int target_commits) {
    std::mt19937 rng(4000 + tid);
    int next_id = tid * 1'000'000;
    int commits_done = 0;
    int attempt = 0;

    while (commits_done < target_commits) {
        ++attempt;
        Transaction txn = m.begin();  // fresh base every attempt -- picks up the latest state

        // Refs from txn.create() this attempt hold LOCAL ids -- meaningful
        // only until try_commit() returns. Buffer separately; fold into the
        // persistent lists (resolved to real ids) only once this attempt
        // actually commits.
        std::vector<Ref<Account>> new_accounts;
        std::vector<Ref<Order>> new_orders;

        const int ops = 8 + rng() % 5;  // ~10 changes per transaction
        for (int k = 0; k < ops; ++k) {
            const int roll = rng() % 100;

            if (roll < 30) {  // create an order
                const Ref<Account> acct = pick_live(txn, accounts, rng);
                if (!acct) continue;
                auto o = std::make_unique<Order>();
                o->code = "T" + std::to_string(tid) + "-O" + std::to_string(next_id++);
                o->account = acct;
                o->qty = 1 + rng() % 5000;
                if (const Ref<Order> p = pick_live(txn, orders, rng)) o->parent = p;
                new_orders.push_back(txn.create(std::move(o)));

            } else if (roll < 40) {  // create an account
                auto a = std::make_unique<Account>();
                a->name = "T" + std::to_string(tid) + "-A" + std::to_string(next_id++);
                a->balance = rng() % 5000;
                new_accounts.push_back(txn.create(std::move(a)));

            } else if (roll < 75) {  // update an order -- qty always, parent set/nulled sometimes
                if (const Ref<Order> r = pick_live(txn, orders, rng)) {
                    if (Order* o = txn.update(r)) {
                        o->qty = 1 + rng() % 5000;
                        const int parent_roll = rng() % 100;
                        if (parent_roll < 30) {
                            if (const Ref<Order> p = pick_live(txn, orders, rng); p && p != r) o->parent = p;
                        } else if (parent_roll < 50) {
                            o->parent.reset();
                        }
                    }
                }

            } else if (roll < 90) {  // delete an order
                if (const Ref<Order> r = pick_live(txn, orders, rng)) txn.remove(r);

            } else {  // delete an account -- cascades any order still pointing at it
                if (const Ref<Account> r = pick_live(txn, accounts, rng)) txn.remove(r);
            }
        }

        const CommitResult res = m.try_commit(txn);
        if (res.status == CommitStatus::Committed) {
            ++commits_done;
            g_committed.fetch_add(1, std::memory_order_relaxed);
            for (const Ref<Account>& r : new_accounts) accounts.push_back(res.to_real(r));
            for (const Ref<Order>& r : new_orders) orders.push_back(res.to_real(r));
            std::printf("[writer %d] attempt %2d: COMMITTED v%llu (%zu changes)\n", tid, attempt,
                        static_cast<unsigned long long>(res.snapshot.version()),
                        res.changes.size());
        } else {
            // Conflict: this attempt's local overlay (including new_accounts/
            // new_orders) is discarded outright -- it was never installed
            // anywhere. Loop again; the next attempt's begin() picks up
            // whatever the other writer just published. (No pre-commit hook
            // is installed in this demo, so Vetoed can't actually happen
            // here -- Conflict is the only failure mode reachable.)
            g_conflicts.fetch_add(1, std::memory_order_relaxed);
            std::printf("[writer %d] attempt %2d: CONFLICT   (%s) on %zu id(s) -- retrying\n", tid,
                        attempt,
                        res.conflict->reason == ConflictReason::IdSetOverlap ? "IdSetOverlap"
                                                                             : "RefIntegrity",
                        res.conflict->ids.size());
        }
    }
}

}  // namespace

int main() {
    Model m;
    std::mt19937 rng(7);
    int next_id = 0;

    // ---- startup: seed a small, deliberately shared pool --------------------
    // Small on purpose: 6 accounts and 15 orders give two threads doing ~10
    // changes per transaction plenty of chances to pick the same ids.
    std::vector<Ref<Account>> accounts;
    std::vector<Ref<Order>> orders;
    {
        Transaction seed = m.begin();
        for (int i = 0; i < 6; ++i) {
            auto a = std::make_unique<Account>();
            a->name = "A" + std::to_string(next_id++);
            a->balance = 1000 * (i + 1);
            accounts.push_back(seed.create(std::move(a)));
        }
        for (int i = 0; i < 15; ++i) {
            auto o = std::make_unique<Order>();
            o->code = "O" + std::to_string(next_id++);
            o->account = accounts[rng() % accounts.size()];
            o->qty = 1 + rng() % 50;
            orders.push_back(seed.create(std::move(o)));
        }
        const CommitResult seed_res = m.try_commit(seed);
        assert(seed_res.status == CommitStatus::Committed);
        for (auto& r : accounts) r = seed_res.to_real(r);
        for (auto& r : orders) r = seed_res.to_real(r);
    }
    std::printf("seeded %zu accounts, %zu orders\n\n", accounts.size(), orders.size());

    // Guarantee at least one Conflict deterministically, independent of how
    // the random two-writer race below happens to schedule: two Transactions
    // built from the SAME Snapshot (Snapshot::begin(), so both share one base
    // version), both touching the same Account. The first try_commit()
    // succeeds; the second's base is now stale for that id, so it MUST come
    // back Conflict -- no thread scheduling involved, entirely single-
    // threaded and 100% reproducible. Without this, the assert at the bottom
    // of main() was flaky: with enough luck, two racing writers can go a
    // whole run without ever actually colliding.
    {
        Snapshot s = m.snapshot();
        Transaction first = s.begin();
        Transaction second = s.begin();
        first.update(accounts[0])->balance += 1;
        second.update(accounts[0])->balance += 2;

        const CommitResult r1 = m.try_commit(first);
        assert(r1.status == CommitStatus::Committed);
        g_committed.fetch_add(1, std::memory_order_relaxed);

        const CommitResult r2 = m.try_commit(second);
        assert(r2.status == CommitStatus::Conflict);
        g_conflicts.fetch_add(1, std::memory_order_relaxed);
        std::printf(
            "guaranteed conflict: two txns from the same snapshot, same id -> %s\n\n",
            r2.conflict->reason == ConflictReason::IdSetOverlap ? "IdSetOverlap" : "RefIntegrity");
    }

    auto sub = m.subscribe(/*queue_depth=*/8);
    std::thread subt(subscriber_thread, sub);

    // ---- two writers, racing try_commit() on the shared pool ---------------
    constexpr int kCommitsPerWriter = 10;
    std::thread w0(writer_thread, std::ref(m), 0, accounts, orders, kCommitsPerWriter);
    std::thread w1(writer_thread, std::ref(m), 1, accounts, orders, kCommitsPerWriter);
    w0.join();
    w1.join();

    m.shutdown();  // wakes the subscriber so it can drain the rest and exit
    subt.join();

    const std::size_t still_pinned = m.wait_for_reclamation();
    std::printf("\n[summary] %llu commits, %llu conflicts across both writers\n",
                static_cast<unsigned long long>(g_committed.load()),
                static_cast<unsigned long long>(g_conflicts.load()));
    std::printf("[reclaim] after barrier: %zu still pinned, %zu backlog\n", still_pinned,
                m.reap_backlog());

    assert(g_conflicts.load() > 0 &&
           "two writers racing ~10-change transactions on a 21-object seed pool should produce at "
           "least one conflict -- if this never fires, the conflict path isn't actually being "
           "exercised");
    std::printf(
        "\nno Ref ever dangled; every batch the subscriber printed was a fully consistent "
        "snapshot.\n");
    return 0;
}
