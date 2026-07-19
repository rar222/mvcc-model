#include <algorithm>
#include <atomic>
#include <cctype>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <functional>
#include <map>
#include <memory>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include "demo/types.h"
#include "model/model.h"
#include "test_harness.h"
#include "test_helpers.h"

using namespace model;
using namespace demo;


// The project's primary concurrency invariant test (see CLAUDE.md):
// many writer threads racing try_commit() on overlapping state, plus
// readers resolving Refs throughout, must never corrupt referrers_ or
// publish a dangling Ref -- extend this one rather than adding a new
// writer-only stress test where the scenario is genuinely the same.
TEST(
    concurrent_stress_many_writer_threads_hammering_try_commit_never_corrupts_referrers_or_leaks_a_dangling_ref) {
    Model m;
    std::vector<Ref<Account>> accounts;
    for (int i = 0; i < 4; ++i) accounts.push_back(make_account(m, "A" + std::to_string(i)));
    std::vector<Ref<Order>> seed_orders;
    for (int i = 0; i < 10; ++i)
        seed_orders.push_back(make_order(m, "O" + std::to_string(i), accounts[i % 4]));

    std::atomic<bool> stop{false};
    std::atomic<std::uint64_t> resolved{0};
    std::atomic<std::uint64_t> snaps{0};
    std::atomic<int> readers_ready{0};
    std::atomic<int> next_id{100};
    std::atomic<std::uint64_t> accessor_calls{0};

    // A live Subscription for the whole run: proves push()/collapse() (the
    // REAL per-commit path every writer's publish_now() drives, not the
    // synchronous manual calls the dedicated Subscription tests use) survives
    // sustained concurrent commit pressure, with nobody draining it -- forcing
    // repeated overflow/coalescing for the duration of the run.
    auto sub = m.subscribe(/*queue_depth=*/4);
    (void)sub;  // never drained on purpose -- see the comment above

    auto reader = [&] {
        bool signaled = false;
        while (!stop.load(std::memory_order_relaxed)) {
            Snapshot s = m.snapshot();
            ++snaps;
            if (!signaled) {
                readers_ready.fetch_add(1, std::memory_order_relaxed);
                signaled = true;
            }
            s.for_each<Order>([&](const Order& o) {
                const Account& acct = s.resolve(o.account);  // must never dangle
                if (acct.name.empty()) ++g_failures;
                ++resolved;
                (void)s.resolve(o.parent);
            });
        }
    };

    auto writer = [&](unsigned seed) {
        std::mt19937 rng(seed);
        while (!stop.load(std::memory_order_relaxed)) {
            Transaction txn = m.begin();
            const int roll = rng() % 100;
            if (roll < 50) {
                auto o = std::make_unique<Order>();
                o->code = "W" + std::to_string(next_id.fetch_add(1));
                o->account = accounts[rng() % accounts.size()];
                txn.create(std::move(o));
            } else if (roll < 80) {
                if (Order* o = txn.update(seed_orders[rng() % seed_orders.size()]))
                    o->qty = static_cast<std::int64_t>(rng() % 50);
            } else {
                txn.remove(seed_orders[rng() % seed_orders.size()]);
            }
            m.try_commit(txn);  // conflicts are expected and fine; just don't corrupt anything
        }
    };

    // Exercises every commit_mu_/ver_mu_/reap_mu_-touching PUBLIC accessor
    // from a thread that is neither a reader nor a writer, concurrently with
    // both -- every other test in this file calls these only before or
    // after a concurrent section, never during one. This is exactly the
    // kind of multi-entry-point pressure that would expose a lock-order
    // violation (commit_mu_ -> ver_mu_ -> reap_mu_, never reversed -- see
    // CLAUDE.md invariant 10) that a single-caller test cannot.
    auto accessor_hammer = [&] {
        while (!stop.load(std::memory_order_relaxed)) {
            (void)m.retired_pending();
            (void)m.current_version();
            (void)m.exhausted_slots();
            (void)m.wait_for_reclamation();
            accessor_calls.fetch_add(1, std::memory_order_relaxed);
        }
    };

    std::thread r1(reader), r2(reader);
    while (readers_ready.load(std::memory_order_relaxed) < 2) std::this_thread::yield();

    std::thread w1([&] { writer(11); }), w2([&] { writer(22); }), w3([&] { writer(33); });
    std::thread acc(accessor_hammer);

    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    stop = true;
    w1.join();
    w2.join();
    w3.join();
    r1.join();
    r2.join();
    acc.join();

    m.shutdown();  // closes `sub` -- safe now that every writer/reader thread has stopped

    CHECK(snaps.load() > 0);
    CHECK(resolved.load() > 0);
    CHECK(accessor_calls.load() > 0);

    Snapshot final_s = m.snapshot();
    final_s.for_each<Order>(
        [&](const Order& o) { CHECK(final_s.resolve(o.account).id == o.account.raw()); });
}

// The intersection the individual hook tests above never cover: every hook
// added this session (PreTransactionsFn, PreCommitFn, PostCommitFn)
// installed globally, doing REAL work, at the same time as the flagship
// stress test's actual cascade-producing multi-writer + reader load. Each
// hook has its own dedicated correctness test elsewhere, using simple,
// isolated creates -- this is the only place a hook's own side effects run
// while a cascade from a DIFFERENT thread is actually in flight. Extend
// this one (not the flagship test above, which stays hook-free so it keeps
// testing the base cascade/referrers_ invariant in isolation) if you add a
// fourth hook.
TEST(hooks_survive_heavy_concurrent_cascade_churn_without_corruption) {
    Model m;
    std::vector<Ref<Account>> accounts;
    for (int i = 0; i < 4; ++i) accounts.push_back(make_account(m, "A" + std::to_string(i)));
    std::vector<Ref<Order>> seed_orders;
    for (int i = 0; i < 10; ++i)
        seed_orders.push_back(make_order(m, "O" + std::to_string(i), accounts[i % 4]));

    std::atomic<std::uint64_t> pre_transactions_ran{0};
    std::atomic<std::uint64_t> pre_commit_ran{0};
    std::atomic<std::uint64_t> post_commit_ran{0};
    std::atomic<std::uint64_t> post_commit_committed{0};
    std::atomic<int> next_pretxn_id{0};

    // Every hook does REAL work, not a no-op -- a no-op hook can't expose an
    // interaction bug with a cascade in flight on another thread.
    m.set_pre_transactions([&](Model& model, const Transaction&) {
        pre_transactions_ran.fetch_add(1, std::memory_order_relaxed);
        Transaction pre = model.begin();
        auto a = std::make_unique<Account>();
        a->name = "PRE" + std::to_string(next_pretxn_id.fetch_add(1, std::memory_order_relaxed));
        pre.create(std::move(a));
        (void)model.run_pre_transaction(pre);  // a fresh create can't conflict; always Committed
    });
    m.set_pre_commit([&](Model&, const Transaction&, const std::vector<Change>&) {
        pre_commit_ran.fetch_add(1, std::memory_order_relaxed);
        return true;  // this test is about survival under load, not veto logic
    });
    m.set_post_commit([&](Model&, const Transaction&, const CommitResult& r) {
        post_commit_ran.fetch_add(1, std::memory_order_relaxed);
        if (r.status == CommitStatus::Committed)
            post_commit_committed.fetch_add(1, std::memory_order_relaxed);
    });

    std::atomic<bool> stop{false};
    std::atomic<std::uint64_t> resolved{0};
    std::atomic<int> readers_ready{0};
    std::atomic<int> next_id{100};

    auto reader = [&] {
        bool signaled = false;
        while (!stop.load(std::memory_order_relaxed)) {
            Snapshot s = m.snapshot();
            if (!signaled) {
                readers_ready.fetch_add(1, std::memory_order_relaxed);
                signaled = true;
            }
            s.for_each<Order>([&](const Order& o) {
                const Account& acct = s.resolve(o.account);  // must never dangle
                if (acct.name.empty()) ++g_failures;
                resolved.fetch_add(1, std::memory_order_relaxed);
            });
        }
    };

    auto writer = [&](unsigned seed) {
        std::mt19937 rng(seed);
        while (!stop.load(std::memory_order_relaxed)) {
            Transaction txn = m.begin();
            const int roll = rng() % 100;
            if (roll < 50) {
                auto o = std::make_unique<Order>();
                o->code = "W" + std::to_string(next_id.fetch_add(1, std::memory_order_relaxed));
                o->account = accounts[rng() % accounts.size()];
                txn.create(std::move(o));
            } else if (roll < 80) {
                if (Order* o = txn.update(seed_orders[rng() % seed_orders.size()]))
                    o->qty = static_cast<std::int64_t>(rng() % 50);
            } else {
                txn.remove(seed_orders[rng() % seed_orders.size()]);
            }
            m.try_commit(txn);  // any status is fine; just don't corrupt anything
        }
    };

    std::thread r1(reader), r2(reader);
    while (readers_ready.load(std::memory_order_relaxed) < 2) std::this_thread::yield();

    std::thread w1([&] { writer(11); }), w2([&] { writer(22); }), w3([&] { writer(33); });

    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    stop = true;
    w1.join();
    w2.join();
    w3.join();
    r1.join();
    r2.join();

    m.set_pre_transactions({});
    m.set_pre_commit({});
    m.set_post_commit({});

    CHECK(resolved.load() > 0);
    CHECK(pre_transactions_ran.load() > 0);
    CHECK(pre_commit_ran.load() > 0);
    CHECK(post_commit_ran.load() > 0);
    // Every attempt that ran pre_commit_ (i.e. reached apply) also ran
    // post_commit_ afterward -- the two must stay in lockstep even under
    // heavy contention, since post_commit_ fires for every non-empty
    // attempt regardless of its outcome.
    CHECK(post_commit_ran.load() >= pre_commit_ran.load());
    CHECK(post_commit_ran.load() >= post_commit_committed.load());

    Snapshot final_s = m.snapshot();
    final_s.for_each<Order>(
        [&](const Order& o) { CHECK(final_s.resolve(o.account).id == o.account.raw()); });
}

// Up to N threads mixed reader/writer — kThreads = 8, split 4 readers / 4 writers, all launched and
// alive concurrently for the run window (same pattern as the existing stress test, just with a
// configurable thread count and roughly even role split instead of a fixed 2+3). 10,000–100,000
// total objects — seeded 100 Accounts + 25,000 Records (25,100 total), with an explicit CHECK
// asserting the count falls in [10000, 100000] both right after seeding and again at the end (after
// the concurrent churn of creates/removes). 5 fields, 2 references — new type Record: label
// (string), value (int64), flag (bool), owner (Ref<Account>, non-nullable — cascade-delete edge),
// related (Opt<Record>, nullable, self-referential — cascade-null edge). Writers touch all five on
// every create, and reassign both references plus a plain field on every update.
TEST(concurrent_stress_mixed_readers_and_writers_at_scale_across_five_fields_two_of_them_refs) {
    Model m;

    // ---- seed: a corpus in [10000, 100000] objects, well before any thread
    // starts -- large enough that the concurrent phase below is genuinely
    // contending over substantial state, not a handful of hot ids. ---------
    constexpr int kAccounts = 100;
    constexpr int kRecords = 25000;
    std::vector<Ref<Account>> accounts;
    {
        Transaction seed = m.begin();
        std::vector<Ref<Account>> local;
        for (int i = 0; i < kAccounts; ++i) {
            auto a = std::make_unique<Account>();
            a->name = "A" + std::to_string(i);
            local.push_back(seed.create(std::move(a)));
        }
        const CommitResult res = commit_ok(m, seed);
        for (auto& r : local) accounts.push_back(res.to_real(r));
    }

    std::vector<Ref<Record>> records;
    for (int i = 0; i < kRecords; i += 1000) {
        Transaction txn = m.begin();
        std::vector<Ref<Record>> local;
        for (int j = 0; j < 1000 && i + j < kRecords; ++j) {
            auto r = std::make_unique<Record>();
            r->label = "R" + std::to_string(i + j);
            r->value = i + j;
            r->flag = (i + j) % 2 == 0;
            r->owner = accounts[static_cast<std::size_t>((i + j) % accounts.size())];
            local.push_back(txn.create(std::move(r)));
        }
        const CommitResult res = commit_ok(m, txn);
        for (auto& r : local) records.push_back(res.to_real(r));
    }
    const std::size_t seeded = accounts.size() + records.size();
    CHECK(seeded >= std::size_t{10000} && seeded <= std::size_t{100000});

    // ---- concurrent phase: up to kThreads alive at once, mixed roles -----
    constexpr int kThreads = 8;
    constexpr int kReaders = kThreads / 2;
    constexpr int kWriters = kThreads - kReaders;

    std::atomic<bool> stop{false};
    std::atomic<std::uint64_t> snaps{0};
    std::atomic<std::uint64_t> resolved{0};
    std::atomic<int> readers_ready{0};
    std::atomic<int> next_id{1'000'000};

    // Readers scan every Record (both ref fields must always resolve/never
    // crash) AND cross-check the two "who points at this Account?" answers
    // against a fixed target -- the O(#Records) scan (find_referrers) and
    // the O(log n + matches) index (find_cached_referrers) -- against the
    // SAME frozen Snapshot, so they must agree exactly even while other
    // threads race concurrent commits underneath.
    auto reader = [&] {
        bool signaled = false;
        while (!stop.load(std::memory_order_relaxed)) {
            Snapshot s = m.snapshot();
            ++snaps;
            if (!signaled) {
                readers_ready.fetch_add(1, std::memory_order_relaxed);
                signaled = true;
            }
            s.for_each<Record>([&](const Record& r) {
                const Account& owner = s.resolve(r.owner);  // non-nullable: must never dangle
                if (owner.name.empty()) ++g_failures;
                ++resolved;
                (void)s.resolve(r.related);  // nullable: null or a live Record, never UB
            });

            auto scan = s.find_referrers<&Record::owner>(accounts[0]);
            auto idx = s.find_cached_referrers<&Record::owner>(accounts[0]);
            std::sort(scan.begin(), scan.end());
            std::sort(idx.begin(), idx.end());
            CHECK(scan == idx);
        }
    };

    // Writers touch all 5 fields: create sets every field at once (including
    // both refs); update reassigns owner (cascade-edge move) and related
    // (nullable edge move or clear) independently of the plain fields;
    // remove exercises both the cascade-delete (an owner Account dying takes
    // its Records with it -- not attempted here directly, but a removed
    // Record can itself have been a cascade victim) and cascade-null paths
    // (a removed Record that other Records' `related` pointed at).
    auto writer = [&](unsigned seed) {
        std::mt19937 rng(seed);
        while (!stop.load(std::memory_order_relaxed)) {
            Transaction txn = m.begin();
            const int roll = static_cast<int>(rng() % 100);
            if (roll < 35) {
                auto r = std::make_unique<Record>();
                r->label = "W" + std::to_string(next_id.fetch_add(1));
                r->value = static_cast<std::int64_t>(rng() % 1000);
                r->flag = (rng() % 2) == 0;
                r->owner = accounts[rng() % accounts.size()];
                if (rng() % 3 == 0) {
                    if (const Ref<Record> rel = pick_live(txn, records, rng)) r->related = rel;
                }
                txn.create(std::move(r));
            } else if (roll < 80) {
                if (Record* r = txn.update(pick_live(txn, records, rng))) {
                    r->value = static_cast<std::int64_t>(rng() % 1000);
                    r->flag = !r->flag;
                    r->owner = accounts[rng() % accounts.size()];
                    if (rng() % 4 == 0)
                        r->related.reset();
                    else if (const Ref<Record> rel = pick_live(txn, records, rng))
                        r->related = rel;
                }
            } else {
                if (const Ref<Record> r = pick_live(txn, records, rng)) txn.remove(r);
            }
            m.try_commit(txn);  // conflicts are expected and fine; just don't corrupt anything
        }
    };

    std::vector<std::thread> pool;
    for (int i = 0; i < kReaders; ++i) pool.emplace_back(reader);
    while (readers_ready.load(std::memory_order_relaxed) < kReaders) std::this_thread::yield();
    for (int i = 0; i < kWriters; ++i)
        pool.emplace_back([&, i] { writer(static_cast<unsigned>(700 + i)); });

    std::this_thread::sleep_for(std::chrono::milliseconds(300));
    stop = true;
    for (auto& t : pool) t.join();

    CHECK(snaps.load() > 0);
    CHECK(resolved.load() > 0);

    // ---- final consistency: every live Record's non-nullable ref resolves,
    // and the two "who points at this Account?" answers still agree for
    // EVERY account, not just the one readers happened to poll. ------------
    Snapshot final_s = m.snapshot();
    const std::size_t final_size = final_s.size();
    CHECK(final_size >= std::size_t{10000} && final_size <= std::size_t{100000});
    final_s.for_each<Record>(
        [&](const Record& r) { CHECK(final_s.resolve(r.owner).id == r.owner.raw()); });
    for (const Ref<Account>& a : accounts) {
        auto scan = final_s.find_referrers<&Record::owner>(a);
        auto idx = final_s.find_cached_referrers<&Record::owner>(a);
        std::sort(scan.begin(), scan.end());
        std::sort(idx.begin(), idx.end());
        CHECK(scan == idx);
    }
}

