#include <algorithm>
#include <atomic>
#include <cctype>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include "model/model.h"
#include "test_harness.h"
#include "test_helpers.h"

using namespace model;

namespace {
// Dedicated type for stress-testing KeyLookupType::Coarse's first-touch
// seed_index_entry() path under real concurrent try_commit() contention --
// several writer threads racing to construct the SAME field's PersistentMap
// for the first time (the exact moment max_shift_ gets baked in) is the one
// scenario a single-threaded test can't exercise. Kept separate from Order
// (which test_types.h keeps deliberately field-for-field identical to
// include/example/types.h's copy) rather than adding a field there.
class CoarseThing final : public model::Object<CoarseThing> {
public:
    std::string code;

    // Second, high-cardinality field so the same concurrent create/
    // reassign/find traffic that stresses by_key_'s Coarse first-touch race
    // also stresses by_cached_field_merged_'s (LookupType::Coarse) --
    // several writer threads racing to construct/merge the SAME field's
    // merged bucket for the first time, and readers concurrently filtering
    // it via find_by_field, is exactly the scenario the single-threaded
    // coarse_field_bucket_merge_* tests (test_lookup.cpp) can't exercise.
    std::string tag;

    // Narrow-int key/field, populated alongside code/tag in the same
    // create/reassign branches below -- stresses by_key_narrow_'s/
    // by_cached_field_narrow_'s own first-touch seed_index_entry() race and
    // reconcile path under real concurrent try_commit() contention, the
    // narrow-int counterpart of what code/tag already stress for Coarse.
    std::int64_t narrow_id = 0;
    std::int64_t narrow_tag = 0;

    template <class Self>
    static void define_keys(Self& s, const model::FieldKeyReader& v) {
        v.key<&CoarseThing::code>(s.code, model::KeyLookupType::Coarse, "code");
        v.key<&CoarseThing::narrow_id>(s.narrow_id, model::KeyLookupType::Exact, "narrow_id");
    }

    template <class Self>
    static void define_fields(Self& s, const model::LookupFieldReader& v) {
        v.field<&CoarseThing::tag>(s.tag, model::LookupType::Coarse, "tag");
        v.field<&CoarseThing::narrow_tag>(s.narrow_tag, model::LookupType::Exact, "narrow_tag");
    }
};
}  // namespace

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

    // 100 orders on accounts[0] put that account's Order::account referrers in hub form from the start.
    // Writers remove and repoint orders from this pool and from the W orders they create, so hub
    // removal, repointing and demotion run under real concurrent commits.
    std::mutex live_orders_mu;
    std::vector<Ref<Order>> live_orders;
    std::atomic<std::uint64_t> hub_ops{0};
    // Creates the 100 seed orders in one commit.
    {
        Transaction seed = m.begin();
        std::vector<Ref<Order>> locals;
        // Builds each order on accounts[0].
        for (int i = 0; i < 100; ++i) {
            auto o = std::make_unique<Order>();
            o->code = "H" + std::to_string(i);
            o->account = accounts[0];
            locals.push_back(seed.create(std::move(o)));
        }
        const CommitResult res = commit_ok(m, seed);
        for (const Ref<Order>& l : locals) live_orders.push_back(res.to_real(l));
    }

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
            // by_key_/by_cached_field_ leaves are dedup'd (pmap::DedupMap):
            // every lookup here resolves a stored Id back through THIS
            // snapshot's own (immutable) spine to recover a comparison key
            // (find_by_key_raw/cached_field_short_circuit_raw, model.cpp),
            // concurrently with writer threads publishing new Roots and
            // reconciling the SAME indexes (the `roll < 75` qty update
            // below, and CoarseThing's create/update branch) -- proves that
            // resolve path is race-free under real concurrent commit
            // pressure, not just against a single writer.
            if (const Order* found = s.find_by_key<&Order::computed_key>("ord:W0")) {
                (void)found->qty;
            }
            for (const Order* o : s.find_by_field<&Order::qty>(0)) (void)o->code;
            (void)s.find_by_key<&CoarseThing::code>("C0");
            // by_cached_field_merged_'s filtered read path (a merged bucket
            // may hold candidates from several different real `tag` values
            // sharing a coarse prefix -- see cached_field_short_circuit_raw),
            // racing the writer's CoarseThing create/reassign branches above.
            for (const CoarseThing* c : s.find_by_field<&CoarseThing::tag>("T0")) (void)c->code;
            // Narrow-int read paths (find_by_key_narrow_raw, cached_field_
            // short_circuit_narrow_raw), racing the same writer traffic.
            (void)s.find_by_key<&CoarseThing::narrow_id>(0);
            for (const CoarseThing* c : s.find_by_field<&CoarseThing::narrow_tag>(0)) (void)c->code;
        }
    };

    std::atomic<int> next_coarse_id{0};
    std::atomic<std::uint64_t> coarse_created{0};
    // Guards `live_coarse` below (a plain std::vector, not lock-free) -- the
    // reassignment branch needs SOME already-committed CoarseThing to update,
    // which create-only writing can never provide.
    std::mutex live_coarse_mu;
    std::vector<Ref<CoarseThing>> live_coarse;

    auto writer = [&](unsigned seed) {
        std::mt19937 rng(seed);
        while (!stop.load(std::memory_order_relaxed)) {
            Transaction txn = m.begin();
            const int roll = rng() % 100;
            if (roll < 45) {
                auto o = std::make_unique<Order>();
                o->code = "W" + std::to_string(next_id.fetch_add(1));
                o->account = accounts[rng() % accounts.size()];
                const Ref<Order> local = txn.create(std::move(o));
                const CommitResult res = m.try_commit(txn);
                // Publishes the committed order so later rolls can remove or repoint it.
                if (res.status == CommitStatus::Committed) {
                    std::lock_guard<std::mutex> lk(live_orders_mu);
                    live_orders.push_back(res.to_real(local));
                }
                continue;
            } else if (roll < 75) {
                if (Order* o = txn.update(seed_orders[rng() % seed_orders.size()]))
                    o->qty = static_cast<std::int64_t>(rng() % 50);
            } else if (roll < 83) {
                txn.remove(seed_orders[rng() % seed_orders.size()]);
            } else if (roll < 90) {
                Ref<Order> victim;
                // Picks a random order from the shared pool.
                {
                    std::lock_guard<std::mutex> lk(live_orders_mu);
                    victim = live_orders[rng() % live_orders.size()];
                }
                // Removes the order half the time, else repoints it to a random account.
                hub_ops.fetch_add(1);
                if (rng() % 2 == 0) {
                    txn.remove(victim);
                } else if (Order* o = txn.update(victim)) {
                    o->account = accounts[rng() % accounts.size()];
                }
            } else if (roll < 96) {
                // All 3 writer threads racing to touch CoarseThing's Coarse-
                // tagged by_key_ entry for the first time -- exactly the
                // seed_index_entry() first-touch race this field exists to
                // stress. try_commit()'s conflict check already serializes
                // the ACTUAL index mutation (commit_mu_), so this is really
                // testing that racing to be first through that gate is safe,
                // not that concurrent mutation of the index itself is.
                auto c = std::make_unique<CoarseThing>();
                const int cid = next_coarse_id.fetch_add(1);
                c->code = "C" + std::to_string(cid);
                c->tag = "T" + std::to_string(cid);
                c->narrow_id = cid;
                c->narrow_tag = cid;
                const Ref<CoarseThing> local = txn.create(std::move(c));
                const CommitResult res = m.try_commit(txn);
                if (res.status == CommitStatus::Committed) {
                    coarse_created.fetch_add(1);
                    std::lock_guard<std::mutex> lk(live_coarse_mu);
                    live_coarse.push_back(res.to_real(local));
                }
                continue;
            } else {
                // Reassign an existing CoarseThing's Coarse-tagged code --
                // reconcile_field_keys's old-vs-new resolve path (see
                // resolve_field_key_string's callers, model.cpp), under real
                // concurrent commit_mu_ contention rather than the single-
                // threaded coverage in test_lookup.cpp's
                // coarse_routing_reconciles_reassigned_keys_correctly_under_forced_collisions.
                Ref<CoarseThing> target;
                bool have_target = false;
                {
                    std::lock_guard<std::mutex> lk(live_coarse_mu);
                    if (!live_coarse.empty()) {
                        target = live_coarse[rng() % live_coarse.size()];
                        have_target = true;
                    }
                }
                if (have_target) {
                    if (CoarseThing* c = txn.update(target)) {
                        const int cid = next_coarse_id.fetch_add(1);
                        c->code = "C" + std::to_string(cid);
                        // by_cached_field_merged_'s reconcile branch, under
                        // the same concurrent contention as code's by_key_
                        // reassignment above.
                        c->tag = "T" + std::to_string(cid);
                        // by_key_narrow_'s/by_cached_field_narrow_'s own
                        // reconcile branch, same contention.
                        c->narrow_id = cid;
                        c->narrow_tag = cid;
                    }
                } else {
                    continue;  // nothing committed yet to reassign: retry
                }
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
            (void)m.reap_backlog();
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
    // Extends the run, up to 10 s, until a CoarseThing was created and a hub-pool order was touched; a
    // fixed window is too short under asan.
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
    while ((coarse_created.load() == 0 || hub_ops.load() == 0) && std::chrono::steady_clock::now() < deadline)
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
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
    CHECK(coarse_created.load() > 0);
    CHECK(hub_ops.load() > 0);

    Snapshot final_s = m.snapshot();
    final_s.for_each<Order>(
        [&](const Order& o) { CHECK(final_s.resolve(o.account).id == o.account.id()); });

    // Every CoarseThing this run committed must still resolve correctly --
    // no corruption of by_key_[CoarseThing::code] survived the concurrent
    // first-touch race.
    std::uint64_t coarse_seen = 0;
    final_s.for_each<CoarseThing>([&](const CoarseThing& c) {
        CHECK(final_s.find_by_key<&CoarseThing::code>(c.code) == &c);
        CHECK(final_s.find_by_key<&CoarseThing::narrow_id>(c.narrow_id) == &c);
        ++coarse_seen;
    });
    CHECK_EQ(coarse_seen, coarse_created.load());

    // The reverse index, hub form included, must still name every referrer the run left:
    // deleting an account has to kill exactly the orders that reference it.
    std::vector<std::size_t> per_account(accounts.size(), 0);
    // Counts the surviving orders per account.
    final_s.for_each<Order>([&](const Order& o) {
        for (std::size_t i = 0; i < accounts.size(); ++i)
            if (o.account.id() == accounts[i].id()) ++per_account[i];
    });
    for (std::size_t i = 0; i < accounts.size(); ++i)
        CHECK_EQ(remove_and_commit(m, accounts[i]), 1 + per_account[i]);
    std::size_t orders_left = 0;
    m.snapshot().for_each<Order>([&](const Order&) { ++orders_left; });
    CHECK_EQ(orders_left, std::size_t{0});
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
        (void)model.run_pre_transaction_without_undo(pre);  // a fresh create can't conflict; always Committed
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
        [&](const Order& o) { CHECK(final_s.resolve(o.account).id == o.account.id()); });
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
            r->owner = accounts[(i + j) % accounts.size()];
            r->owner_scan = r->owner;
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
    // against a fixed target -- find_referrers's O(log n + matches) cache-hit
    // branch on owner, and its O(#Records) scan-fallback branch on owner's
    // scan-only twin owner_scan (kept equal to owner by every writer below)
    // -- against the SAME frozen Snapshot, so they must agree exactly even
    // while other threads race concurrent commits underneath.
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

            auto scan = s.find_referrers<&Record::owner_scan>(accounts[0]);
            auto idx = s.find_referrers<&Record::owner>(accounts[0]);
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
                r->owner_scan = r->owner;
                if (rng() % 3 == 0) {
                    if (const Ref<Record> rel = pick_live(txn, records, rng)) r->related = rel;
                }
                txn.create(std::move(r));
            } else if (roll < 80) {
                if (Record* r = txn.update(pick_live(txn, records, rng))) {
                    r->value = static_cast<std::int64_t>(rng() % 1000);
                    r->flag = !r->flag;
                    r->owner = accounts[rng() % accounts.size()];
                    r->owner_scan = r->owner;
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
        [&](const Record& r) { CHECK(final_s.resolve(r.owner).id == r.owner.id()); });
    for (const Ref<Account>& a : accounts) {
        auto scan = final_s.find_referrers<&Record::owner_scan>(a);
        auto idx = final_s.find_referrers<&Record::owner>(a);
        std::sort(scan.begin(), scan.end());
        std::sort(idx.begin(), idx.end());
        CHECK(scan == idx);
    }
}

// Liveness/fairness, not correctness: every writer thread races
// try_commit() against the SAME single id every single attempt -- the
// most adversarial contention shape, since every in-flight attempt from
// every thread conflicts with every other thread's. commit_mu_ serializes
// apply, but nothing about that serialization is meant to let one thread
// keep winning while another is starved out indefinitely. Distinct from
// concurrent_stress_many_writer_threads_hammering_try_commit_..., which
// checks that concurrent commits never corrupt anything but never asks
// whether any one thread's writes actually got through -- a starved
// thread there would be invisible to that test. Bounded retries (not a
// blocking wait) so a genuine starvation bug fails this test loudly
// instead of hanging the suite.
TEST(every_writer_thread_eventually_lands_a_commit_under_sustained_single_id_contention) {
    Model m;
    const Ref<Account> hot = make_account(m, "HOT", 0);

    constexpr int kThreads = 6;
    constexpr int kMaxAttemptsPerThread = 20000;  // generous: this FAILS, not hangs, if exceeded
    std::vector<std::thread> workers;
    std::vector<char> succeeded(kThreads, 0);   // each index written only by its own thread
    std::vector<int> attempts_used(kThreads, 0);

    for (int i = 0; i < kThreads; ++i) {
        workers.emplace_back([&, i] {
            int attempts = 0;
            for (; attempts < kMaxAttemptsPerThread; ++attempts) {
                Transaction txn = m.begin();
                txn.update(hot)->balance += 1;  // every thread, every attempt: the same id
                if (m.try_commit(txn).status == CommitStatus::Committed) {
                    succeeded[i] = 1;
                    ++attempts;  // count the winning attempt too
                    break;
                }
            }
            attempts_used[i] = attempts;
        });
    }
    for (auto& t : workers) t.join();

    for (int i = 0; i < kThreads; ++i) {
        CHECK(succeeded[i] != 0);
        CHECK(attempts_used[i] < kMaxAttemptsPerThread);  // same fact, phrased for a readable failure
    }
}

