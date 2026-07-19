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


// The baseline success path: a straightforward create commits, and the
// returned CommitResult's snapshot/changes reflect that publish.
TEST(try_commit_of_a_non_conflicting_transaction_succeeds_and_publishes) {
    Model m;
    Transaction txn = m.begin();
    auto a = std::make_unique<Account>();
    a->name = "A1";
    txn.create(std::move(a));
    CommitResult res = m.try_commit(txn);
    CHECK(res.status == CommitStatus::Committed);
    CHECK(res.snapshot.version() == m.current_version());
    CHECK_EQ(res.changes.size(), std::size_t{1});
}

// Two threads committing updates to DIFFERENT ids never conflict with
// each other -- object-write-set OCC only serializes overlapping writes,
// not all concurrent writers.
TEST(two_concurrent_transactions_touching_disjoint_ids_both_succeed) {
    Model m;
    const Ref<Account> a1 = make_account(m, "A1", 1);
    const Ref<Account> a2 = make_account(m, "A2", 2);

    std::atomic<int> committed{0};
    auto worker = [&](Ref<Account> target, std::int64_t new_balance) {
        Transaction txn = m.begin();
        txn.update(target)->balance = new_balance;
        CommitResult res = m.try_commit(txn);
        if (res.status == CommitStatus::Committed) committed.fetch_add(1);
    };

    std::thread t1([&] { worker(a1, 100); });
    std::thread t2([&] { worker(a2, 200); });
    t1.join();
    t2.join();

    CHECK_EQ(committed.load(), 2);
    Snapshot s = m.snapshot();
    CHECK_EQ(s.find(a1)->balance, 100);
    CHECK_EQ(s.find(a2)->balance, 200);
}

// Two transactions racing an update to the SAME id: the first commit
// wins, the second is rejected as Conflict/IdSetOverlap, and a fresh
// retry against the new base succeeds cleanly.
TEST(two_concurrent_transactions_updating_the_same_id_one_conflicts_with_id_set_overlap) {
    Model m;
    const Ref<Account> a = make_account(m, "A1", 0);

    Transaction t1 = m.begin(m.snapshot());
    Transaction t2 = m.begin(m.snapshot());
    t1.update(a)->balance = 111;
    t2.update(a)->balance = 222;

    CommitResult r1 = m.try_commit(t1);
    CHECK(r1.status == CommitStatus::Committed);

    CommitResult r2 = m.try_commit(t2);
    CHECK(r2.status == CommitStatus::Conflict);
    CHECK(r2.conflict.has_value());
    CHECK(r2.conflict->reason == ConflictReason::IdSetOverlap);
    CHECK(std::find(r2.conflict->ids.begin(), r2.conflict->ids.end(), a.raw()) !=
          r2.conflict->ids.end());

    // The loser can retry cleanly against the new base.
    Transaction retry = m.begin();
    retry.update(a)->balance = 222;
    CommitResult rr = m.try_commit(retry);
    CHECK(rr.status == CommitStatus::Committed);
    CHECK_EQ(m.snapshot().find(a)->balance, 222);
}

// A concurrent remove() and update() targeting the same id conflict via
// IdSetOverlap exactly like two updates would -- removal isn't a special
// case of the id-overlap check.
TEST(update_conflicts_with_a_concurrent_remove_of_the_same_id) {
    Model m;
    const Ref<Account> a = make_account(m, "A1");

    Transaction t1 = m.begin(m.snapshot());
    Transaction t2 = m.begin(m.snapshot());
    t1.remove(a);
    t2.update(a)->balance = 5;

    CommitResult r1 = m.try_commit(t1);
    CHECK(r1.status == CommitStatus::Committed);

    CommitResult r2 = m.try_commit(t2);
    CHECK(r2.status == CommitStatus::Conflict);
    CHECK(r2.conflict->reason == ConflictReason::IdSetOverlap);
}

// The headline multi-writer guarantee: see the in-body comment. A
// transaction that never touched a since-deleted id, but references it
// via a brand-new object, still fails -- against the LATEST state, not base.
TEST(ref_integrity_is_revalidated_against_latest_not_just_base) {
    // txn A's base has an Account alive. txn B deletes it and commits first.
    // A never touches that id directly, but references it via a brand-new
    // object -- A's commit must still fail (not id-set overlap: A never
    // wrote that id -- but because the Ref<> it just created would dangle
    // against the LATEST state).
    Model m;
    const Ref<Account> a = make_account(m, "A1");

    Snapshot base = m.snapshot();
    Transaction a_txn = m.begin(base);
    Transaction b_txn = m.begin(base);

    b_txn.remove(a);
    CommitResult rb = m.try_commit(b_txn);
    CHECK(rb.status == CommitStatus::Committed);

    auto o = std::make_unique<Order>();
    o->code = "O1";
    o->account = a;  // new Order -> now-dead Account
    a_txn.create(std::move(o));

    // The target was alive at a_txn's base and died to a CONCURRENT commit:
    // classified as a Conflict (retryable contention), not Invalid.
    const CommitResult ra = m.try_commit(a_txn);
    CHECK(ra.status == CommitStatus::Conflict);
    CHECK(ra.conflict->reason == ConflictReason::RefIntegrity);
}

// A hub-and-spoke cascade (1 account, 20 dependent orders) kills exactly
// the expected count in one commit, confirming try_commit()'s deferred
// cascade resolution matches what an eager, synchronous BFS would do.
TEST(cascade_delete_resolved_at_commit_time_matches_a_synchronous_bfs) {
    Model m;
    const Ref<Account> hub = make_account(m, "hub");
    Transaction seed = m.begin();
    for (int i = 0; i < 20; ++i) {
        auto o = std::make_unique<Order>();
        o->code = "O" + std::to_string(i);
        o->account = hub;
        seed.create(std::move(o));
    }
    commit_ok(m, seed);

    const std::size_t killed = remove_and_commit(m, hub);
    CHECK_EQ(killed, std::size_t{21});  // hub + all 20 spokes, exactly

    Snapshot s = m.snapshot();
    CHECK_EQ(s.size(), std::size_t{0});
}


// prune_changelog()'s retention policy: entries accumulate while a
// Snapshot pins an old watermark, then collapse back down once that
// Snapshot drops and later commits move the watermark past them.
TEST(changelog_entries_are_pruned_once_no_open_transaction_or_snapshot_needs_them) {
    Model m;
    make_account(m, "A1");

    {
        Snapshot pin = m.snapshot();  // pins the watermark at this version
        for (int i = 0; i < 5; ++i) make_account(m, "X" + std::to_string(i));
        CHECK(m.debug_changelog_size() >= std::size_t{5});
    }
    // `pin` dropped -- each subsequent commit's prune sees a watermark that no
    // longer includes it, so the backlog collapses over the next couple of
    // commits. A commit's own changelog entry always survives ITS OWN prune
    // call (that same commit's Transaction::base() is still registered in
    // live_ at the moment prune runs, inside try_commit() itself) -- it is
    // only pruned once a LATER commit's base has moved past it. So the
    // backlog settles at "the most recent commit's own entry", not zero.
    make_account(m, "after_pin_drop");
    make_account(m, "flush");
    CHECK(m.debug_changelog_size() <= std::size_t{1});
}


TEST(diagnostics_reports_object_population_storage_and_indexes) {
    Model m;
    const Ref<Account> a = make_account(m, "A1");
    make_order(m, "O1", a, Opt<Order>{}, 5);

    const Model::Diagnostics d = m.diagnostics();
    CHECK_EQ(d.version, m.current_version());
    CHECK_EQ(d.live_object_count, std::size_t{2});
    CHECK_EQ(d.live_by_type.size(), std::size_t{2});
    for (const auto& tc : d.live_by_type) {
        CHECK_EQ(tc.live_count, std::size_t{1});  // one Account, one Order
        CHECK(tc.type_name.find("Account") != std::string::npos ||
              tc.type_name.find("Order") != std::string::npos);
    }
    CHECK(d.chunk_count >= std::size_t{1});
    CHECK(d.slots_allocated >= std::size_t{2});

    // Account::name (define_keys), Order::computed_key (define_keys).
    CHECK_EQ(d.key_indexed_fields, std::size_t{2});
    // Order::qty, Order::computed_key (define_cached_fields).
    CHECK_EQ(d.cached_value_indexed_fields, std::size_t{2});
    // Order::account (define_cached_references) -- parent is deliberately uncached.
    CHECK_EQ(d.cached_reference_indexed_fields, std::size_t{1});

    // referrers_ tracks every define_references() field regardless of caching:
    // one Order.account edge into the Account it was created with.
    CHECK_EQ(d.reverse_index_targets, std::size_t{1});
    CHECK_EQ(d.reverse_index_edges, std::size_t{1});
}

TEST(diagnostics_reports_live_snapshot_pins_and_the_reclamation_watermark) {
    Model m;
    make_account(m, "A1");

    const Model::Diagnostics before_pin = m.diagnostics();
    CHECK_EQ(before_pin.live_snapshot_versions, std::size_t{0});

    {
        Snapshot pin = m.snapshot();
        const Model::Diagnostics pinned = m.diagnostics();
        CHECK_EQ(pinned.live_snapshot_versions, std::size_t{1});
        CHECK(pinned.live_snapshot_refs >= std::size_t{1});
        CHECK_EQ(pinned.reclamation_watermark, pin.version());
    }
}

TEST(diagnostics_reports_installed_hooks_and_subscriber_count) {
    Model m;
    CHECK(!m.diagnostics().pre_commit_hook_installed);
    CHECK(!m.diagnostics().pre_transactions_hook_installed);
    CHECK_EQ(m.diagnostics().subscriber_count, std::size_t{0});

    m.set_pre_commit([](Model&, const Transaction&, const std::vector<Change>&) { return true; });
    m.set_pre_transactions([](Model&, const Transaction&) {});
    auto sub = m.subscribe();

    const Model::Diagnostics d = m.diagnostics();
    CHECK(d.pre_commit_hook_installed);
    CHECK(d.pre_transactions_hook_installed);
    CHECK_EQ(d.subscriber_count, std::size_t{1});
}

// Mirrors changelog_entries_are_pruned_once_no_open_transaction_or_snapshot_needs_them
// above: retained_commit_history is a lightweight (version, change count) summary
// of the SAME changelog_ that test exercises, so it should track debug_changelog_size()
// exactly, entry for entry.
TEST(diagnostics_retained_commit_history_tracks_the_changelog) {
    Model m;
    make_account(m, "A1");
    make_account(m, "A2");
    make_account(m, "A3");

    const Model::Diagnostics d = m.diagnostics();
    CHECK_EQ(d.retained_commit_history.size(), m.debug_changelog_size());
    for (const auto& [version, change_count] : d.retained_commit_history) {
        CHECK(version <= d.version);
        CHECK(change_count >= std::size_t{1});
    }
}

