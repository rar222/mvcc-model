// Concurrent demo: several writer threads independently begin()/mutate/
// try_commit() transactions -- retrying on Conflict -- while two readers
// snapshot and resolve every reference, and one deliberately slow subscriber
// forces the event queue to coalesce.
//
// The reader's resolve() assert is the real test: if cascade resolution or
// conflict detection is wrong, a Ref dangles and this program dies. Run it
// under
//   cmake --preset asan  (use-after-free -- did the reaper free too early?)
//   cmake --preset tsan  (races between concurrent writers/readers)
//
// The writer log is expected to show BOTH successful commits AND observed
// conflicts: a demo that never exercises the conflict path hasn't proven
// anything about a multi-writer design.

#include <algorithm>
#include <atomic>
#include <cassert>
#include <chrono>
#include <cinttypes>
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
using namespace std::chrono_literals;

namespace {

std::atomic<bool> g_stop{false};
std::atomic<std::uint64_t> g_snapshots{0};
std::atomic<std::uint64_t> g_refs_resolved{0};
std::atomic<std::uint64_t> g_null_parents{0};

std::atomic<std::uint64_t> g_committed{0};
std::atomic<std::uint64_t> g_conflicts{0};
std::atomic<std::uint64_t> g_created{0};
std::atomic<std::uint64_t> g_updated{0};
std::atomic<std::uint64_t> g_deleted{0};
std::atomic<std::uint64_t> g_cascades{0};

void reader_thread(Model& m, int tid) {
    std::mt19937 rng(1000 + tid);
    while (!g_stop.load(std::memory_order_relaxed)) {
        Snapshot s = m.snapshot();
        ++g_snapshots;

        std::size_t orders = 0, resolved = 0, nulls = 0;

        // Every order, as a view: the object paired with the snapshot it came
        // from. Traversal uses the view's own snapshot, so mixing versions is
        // not expressible.
        s.for_each_view<Order>([&](View<Order> ord) {
            ++orders;

            // Ref<Account> -> View<Account>. Guaranteed to resolve. Never null.
            View<Account> acct = ord[&Order::account];
            assert(!acct->name.empty());
            ++resolved;

            // Opt<Order> -> OptView<Order>. Empty if a cascade took the
            // parent and nulled the field rather than killing this order.
            if (auto parent = ord[&Order::parent]) {
                assert(!parent->code.empty());
                ++resolved;
            } else {
                ++nulls;
            }

            // An empty hop propagates instead of resolving, so the whole
            // chain needs one check rather than one per Opt<> in it.
            if (auto owner = ord[&Order::parent][&Order::account]) {
                assert(!owner->name.empty());
            }

            assert(s.find_by_key<&Order::computed_key>(ord->computed_key()) == &*ord);
        });

        g_refs_resolved += resolved;
        g_null_parents += nulls;

        if (rng() % 40 == 0)
            std::printf("[reader %d] v%" PRIu64
                        ": %zu orders, %zu refs resolved, %zu null parents\n",
                        tid, s.version(), orders, resolved, nulls);

        std::this_thread::sleep_for(std::chrono::microseconds(200 + rng() % 800));
    }
}

void subscriber_thread(std::shared_ptr<Subscription> sub) {
    std::uint64_t created = 0, updated = 0, deleted = 0, coalesced = 0, batches = 0;
    while (auto u = sub->wait_for_update()) {
        ++batches;
        if (u->coalesced) ++coalesced;
        for (const Change& c : *u->changes) {
            switch (c.kind) {
                case ChangeKind::Created: ++created; break;
                case ChangeKind::Updated: ++updated; break;
                case ChangeKind::Deleted: ++deleted; break;
            }
        }
        std::this_thread::sleep_for(3ms);  // deliberately slow: forces coalescing
    }
    std::printf("\n[subscriber] %" PRIu64 " batches (%" PRIu64 " coalesced): %" PRIu64
                " created, %" PRIu64 " updated, %" PRIu64 " deleted\n",
                batches, coalesced, created, updated, deleted);
}

/// Pick a handle this transaction still believes is live -- checked against
/// the transaction's OWN local view (its base plus its own pending edits),
/// not some model-wide notion of existence, since there isn't one anymore:
/// existence is only ever meaningful relative to a Snapshot or a Transaction.
/// Another thread's concurrent delete can still make this stale by the time
/// try_commit() runs; that's fine -- it's exactly what the conflict check is
/// for.
template <class T>
Ref<T> pick_live(const Transaction& txn, std::vector<Ref<T>>& v, std::mt19937& rng) {
    for (int tries = 0; tries < 12 && !v.empty(); ++tries) {
        const Ref<T> r = v[rng() % v.size()];
        if (txn.exists(r)) return r;
    }
    return Ref<T>{};
}

/// One writer thread: loops begin() / random mutations / try_commit(),
/// retrying on Conflict. Each thread owns its own copy of the seed id lists
/// -- sharing a std::vector across writer threads without synchronization
/// would itself be the bug this whole design exists to avoid at the object
/// level.
void writer_thread(Model& m, int tid, std::vector<Ref<Account>> accounts,
                   std::vector<Ref<Order>> orders, int target_commits) {
    std::mt19937 rng(2000 + tid);
    int next_id = tid * 1'000'000;
    int commits_done = 0;
    int attempts = 0;

    while (commits_done < target_commits) {
        ++attempts;
        Transaction txn = m.begin();  // fresh base every attempt -- picks up the latest state

        // Refs returned by txn.create() this attempt hold LOCAL ids -- only
        // meaningful until try_commit() returns (see CommitResult::to_real's
        // doc comment). Buffer them separately and only fold them into the
        // persistent accounts/orders lists, resolved to their real ids, once
        // this attempt actually commits; a local id must never survive into
        // a later attempt's Transaction.
        std::vector<Ref<Account>> new_accounts;
        std::vector<Ref<Order>> new_orders;

        const int ops = 1 + rng() % 4;
        for (int k = 0; k < ops; ++k) {
            const int roll = rng() % 100;

            if (roll < 40) {  // create an order
                const Ref<Account> acct = pick_live(txn, accounts, rng);
                if (!acct) continue;
                auto o = std::make_unique<Order>();
                o->code = "T" + std::to_string(tid) + "-O" + std::to_string(next_id++);
                o->account = acct;
                if (rng() % 3 == 0) {
                    if (const Ref<Order> p = pick_live(txn, orders, rng)) o->parent = p;
                }
                o->qty = 1 + rng() % 100;
                new_orders.push_back(txn.create(std::move(o)));

            } else if (roll < 50) {  // create an account
                auto a = std::make_unique<Account>();
                a->name = "T" + std::to_string(tid) + "-A" + std::to_string(next_id++);
                a->balance = rng() % 5000;
                new_accounts.push_back(txn.create(std::move(a)));

            } else if (roll < 85) {  // update an order
                const Ref<Order> r = pick_live(txn, orders, rng);
                if (Order* o = txn.update(r)) o->qty = 1 + rng() % 100;

            } else if (roll < 95) {  // delete an order
                const Ref<Order> r = pick_live(txn, orders, rng);
                if (r) txn.remove(r);

            } else if (accounts.size() > 2) {  // delete an account -- the interesting one
                const Ref<Account> r = pick_live(txn, accounts, rng);
                if (r) txn.remove(r);
            }
        }

        const CommitResult res = m.try_commit(txn);
        if (res.status == CommitStatus::Committed) {
            ++commits_done;
            g_committed.fetch_add(1);
            for (const Ref<Account>& r : new_accounts) accounts.push_back(res.to_real(r));
            for (const Ref<Order>& r : new_orders) orders.push_back(res.to_real(r));
            std::size_t killed = 0, made = 0, changed = 0;
            for (const Change& c : res.changes) {
                if (c.kind == ChangeKind::Created) ++made;
                else if (c.kind == ChangeKind::Updated) ++changed;
                else ++killed;
            }
            g_created += made;
            g_updated += changed;
            g_deleted += killed;
            if (killed > 1) g_cascades.fetch_add(1);
        } else if (res.status == CommitStatus::Conflict) {
            g_conflicts.fetch_add(1);
            // new_accounts/new_orders are simply discarded: they were never
            // installed anywhere. Loop again: the next attempt's begin()
            // picks up whatever won.
        }
        std::this_thread::sleep_for(500us);
    }

    if (attempts > commits_done)
        std::printf("[writer %d] %d commits in %d attempts (%d conflicts absorbed)\n", tid,
                    commits_done, attempts, attempts - commits_done);
}

}  // namespace

int main() {
    Model m;
    std::mt19937 rng(42);
    int next_id = 0;

    // ---- startup: seed accounts and orders in one transaction --------------
    // Scoped: both `seed` (a Transaction) and the CommitResult below pin a
    // Snapshot for as long as they're alive -- a Transaction via base(), a
    // CommitResult via its published `snapshot` field. Left unscoped at
    // main()'s top level, they would pin the model's very first two versions
    // for the ENTIRE rest of the run, and the reaper could never reclaim
    // anything retired since. accounts/orders below are just Ref<T> (8 bytes,
    // no snapshot attached), so copying them back out of this block is free.
    std::vector<Ref<Account>> accounts;
    std::vector<Ref<Order>> orders;
    {
        Transaction seed = m.begin();
        for (int i = 0; i < 5; ++i) {
            auto a = std::make_unique<Account>();
            a->name = "A" + std::to_string(next_id++);
            a->balance = 1000 * (i + 1);
            accounts.push_back(seed.create(std::move(a)));
        }
        for (int i = 0; i < 20; ++i) {
            auto o = std::make_unique<Order>();
            o->code = "O" + std::to_string(next_id++);
            o->account = accounts[rng() % accounts.size()];
            if (!orders.empty() && rng() % 2) o->parent = orders[rng() % orders.size()];
            o->qty = 1 + rng() % 100;
            orders.push_back(seed.create(std::move(o)));
        }
        const CommitResult seed_res = m.try_commit(seed);
        assert(seed_res.status == CommitStatus::Committed);
        // accounts/orders currently hold LOCAL ids (see CommitResult::to_real's
        // doc comment) -- translate to real, post-commit ids before anything
        // outside this transaction uses them.
        for (auto& r : accounts) r = seed_res.to_real(r);
        for (auto& r : orders) r = seed_res.to_real(r);
    }
    std::printf("seeded %zu accounts, %zu orders\n", accounts.size(), orders.size());

    // Demonstrate an integrity failure before the concurrent phase. Nothing
    // is validated until try_commit()'s apply phase -- building a transaction
    // never touches shared state, so there is nothing to roll back afterward
    // on the caller's side. try_commit() rejects the whole transaction as
    // Invalid (no exceptions in this project -- see CLAUDE.md) and unwinds
    // everything it had applied, DOOMED included.
    {
        const std::size_t before = m.snapshot().size();
        Transaction bad_txn = m.begin();
        auto doomed = std::make_unique<Order>();
        doomed->code = "DOOMED";
        doomed->account = accounts[0];
        bad_txn.create(std::move(doomed));  // fine so far -- not checked yet
        auto bad = std::make_unique<Order>();
        bad->code = "BAD";
        bad->account = Ref<Account>(Id{999999, 1});  // dangling Ref
        bad_txn.create(std::move(bad));              // still fine locally
        const CommitResult res = m.try_commit(bad_txn);  // rejected mid-apply, fully unwound
        assert(res.status == CommitStatus::Invalid);
        std::printf("rejected invalid txn: %s -> discarded, nothing was ever published\n",
                    res.error->message.c_str());
        std::printf("after rejection: %zu objects (was %zu) -- DOOMED never happened\n\n",
                    m.snapshot().size(), before);
    }

    // Guarantee at least one Conflict deterministically, independent of how
    // the random concurrent phase below happens to schedule: two
    // Transactions built from the SAME Snapshot (Snapshot::begin(), so both
    // share one base version), both touching the same Account. The first
    // try_commit() succeeds; the second's base is now stale for that id, so
    // it MUST come back Conflict -- no thread scheduling involved, entirely
    // single-threaded and 100% reproducible. Without this, the assert at the
    // bottom of main() was flaky: with enough luck, three racing writer
    // threads can go a whole run without ever actually colliding.
    {
        Snapshot s = m.snapshot();
        Transaction first = s.begin();
        Transaction second = s.begin();
        first.update(accounts[0])->balance += 1;
        second.update(accounts[0])->balance += 2;

        const CommitResult r1 = m.try_commit(first);
        assert(r1.status == CommitStatus::Committed);
        g_committed.fetch_add(1);
        g_updated.fetch_add(1);

        const CommitResult r2 = m.try_commit(second);
        assert(r2.status == CommitStatus::Conflict);
        g_conflicts.fetch_add(1);
        std::printf("guaranteed conflict: two txns from the same snapshot, same id -> %s\n\n",
                    r2.conflict->reason == ConflictReason::IdSetOverlap ? "IdSetOverlap" : "RefIntegrity");
    }

    auto sub = m.subscribe(/*queue_depth=*/4);
    std::thread subt(subscriber_thread, sub);
    std::thread r1(reader_thread, std::ref(m), 1);
    std::thread r2(reader_thread, std::ref(m), 2);

    // ---- writers: three independent threads, each racing try_commit() -----
    constexpr int kWriters = 3;
    constexpr int kCommitsPerWriter = 120;
    std::vector<std::thread> writers;
    for (int t = 0; t < kWriters; ++t)
        writers.emplace_back(writer_thread, std::ref(m), t, accounts, orders, kCommitsPerWriter);
    for (auto& w : writers) w.join();

    g_stop = true;
    r1.join();
    r2.join();
    m.shutdown();
    subt.join();

    const std::size_t still_pinned = m.wait_for_reclamation();

    std::printf("\n[writers] %" PRIu64 " commits, %" PRIu64 " conflicts: %" PRIu64
                " created, %" PRIu64 " updated, %" PRIu64 " deleted (%" PRIu64 " cascades)\n",
                g_committed.load(), g_conflicts.load(), g_created.load(), g_updated.load(),
                g_deleted.load(), g_cascades.load());
    std::printf("[readers] %" PRIu64 " snapshots, %" PRIu64 " refs resolved, %" PRIu64
                " null OptRefs\n",
                g_snapshots.load(), g_refs_resolved.load(), g_null_parents.load());
    std::printf("[reclaim] after barrier: %zu still pinned, %zu backlog\n", still_pinned,
                m.reap_backlog());

    assert(g_conflicts.load() > 0 &&
          "three writer threads racing on a shared object graph should produce at least one "
          "conflict -- if this never fires, the conflict path isn't actually being exercised");
    std::printf("\nno Ref ever dangled; multi-writer conflicts were detected and absorbed.\n");
    return 0;
}
