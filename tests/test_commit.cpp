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

#include "model/model.h"
#include "test_harness.h"
#include "test_helpers.h"

using namespace model;


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

// DESIGN.md is explicit that this is object-write-set OCC, not full
// serializable OCC: "two transactions that only read overlapping state,
// without either writing or referencing it, can both commit even if the
// result is a write-skew anomaly." That's prose describing intended
// behavior, not something any test pins down -- this is that pin, so a
// future change can't silently strengthen OR weaken the guarantee without
// a test noticing (see CLAUDE.md's "Known scope boundaries").
//
// Two accounts share a combined-overdraft invariant enforced only by each
// transaction reading BOTH balances before deciding to debit ONE of them.
// T1 and T2 both read the same pre-debit snapshot (sum=200, limit=100),
// each independently concludes its own 100 debit is safe, and each writes
// only its own account -- disjoint write sets, no Ref<> between the two
// Accounts, so nothing in the object-write-set model has any reason to
// conflict them. Both commit; the combined invariant ends up violated
// anyway. That is write skew, and it is expected, not a bug.
TEST(two_transactions_reading_the_same_invariant_and_writing_disjoint_ids_both_commit_as_write_skew) {
    Model m;
    const Ref<Account> a1 = make_account(m, "A1", 100);
    const Ref<Account> a2 = make_account(m, "A2", 100);
    constexpr std::int64_t kCombinedOverdraftLimit = 100;

    const Snapshot base = m.snapshot();
    CHECK_EQ(base.find(a1)->balance + base.find(a2)->balance, std::int64_t{200});

    Transaction t1 = m.begin(base);
    Transaction t2 = m.begin(base);

    // Each transaction reads BOTH balances off its own (shared) base before
    // deciding to debit just one -- the read that establishes the invariant
    // is never turned into a write, so it never enters either write set.
    const std::int64_t t1_combined = t1.base().find(a1)->balance + t1.base().find(a2)->balance;
    CHECK(t1_combined - 100 >= kCombinedOverdraftLimit);
    t1.update(a1)->balance -= 100;

    const std::int64_t t2_combined = t2.base().find(a1)->balance + t2.base().find(a2)->balance;
    CHECK(t2_combined - 100 >= kCombinedOverdraftLimit);
    t2.update(a2)->balance -= 100;

    CHECK(m.try_commit(t1).status == CommitStatus::Committed);
    CHECK(m.try_commit(t2).status == CommitStatus::Committed);  // NOT a Conflict: disjoint write sets

    Snapshot s = m.snapshot();
    const std::int64_t final_combined = s.find(a1)->balance + s.find(a2)->balance;
    CHECK_EQ(final_combined, std::int64_t{0});                    // invariant actually violated
    CHECK(final_combined < kCombinedOverdraftLimit);              // < 100, below the enforced floor
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
    CHECK(std::find(r2.conflict->ids.begin(), r2.conflict->ids.end(), a.id()) !=
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

// Same setup as ref_integrity_is_revalidated_against_latest_not_just_base
// above, but checking WHAT classify_apply_failure() reports, not just the
// reason enum: ConflictInfo::ids for a RefIntegrity conflict names the
// specific dangling target (err.bad_target), not the id(s) this transaction
// itself wrote (there are none -- the create is local, and the Account was
// never touched by a_txn at all).
TEST(conflict_info_ids_names_the_dangling_target_for_ref_integrity) {
    Model m;
    const Ref<Account> a = make_account(m, "A1");

    Snapshot base = m.snapshot();
    Transaction a_txn = m.begin(base);
    Transaction b_txn = m.begin(base);

    b_txn.remove(a);
    CHECK(m.try_commit(b_txn).status == CommitStatus::Committed);

    auto o = std::make_unique<Order>();
    o->code = "O1";
    o->account = a;  // new Order -> now-dead Account
    a_txn.create(std::move(o));

    const CommitResult ra = m.try_commit(a_txn);
    CHECK(ra.status == CommitStatus::Conflict);
    CHECK(ra.conflict.has_value());
    CHECK(ra.conflict->reason == ConflictReason::RefIntegrity);
    CHECK_EQ(ra.conflict->ids.size(), std::size_t{1});
    CHECK(std::find(ra.conflict->ids.begin(), ra.conflict->ids.end(), a.id()) !=
          ra.conflict->ids.end());
}

// check_id_overlap() collects EVERY colliding id from the changelog, not
// just the first one it sees -- a transaction that touches several ids,
// racing a concurrent commit that touched several of the SAME ids, must
// see all of them in conflict->ids.
TEST(id_set_overlap_conflict_reports_every_overlapping_id_not_just_one) {
    Model m;
    const Ref<Account> a1 = make_account(m, "A1", 0);
    const Ref<Account> a2 = make_account(m, "A2", 0);
    const Ref<Account> a3 = make_account(m, "A3", 0);

    Snapshot base = m.snapshot();
    Transaction winner = m.begin(base);
    Transaction loser = m.begin(base);

    // The winner touches a1 and a2 only.
    winner.update(a1)->balance = 111;
    winner.update(a2)->balance = 222;
    CHECK(m.try_commit(winner).status == CommitStatus::Committed);

    // The loser touches a1, a2 (both overlapping) AND a3 (no overlap).
    loser.update(a1)->balance = 999;
    loser.update(a2)->balance = 999;
    loser.update(a3)->balance = 999;

    const CommitResult res = m.try_commit(loser);
    CHECK(res.status == CommitStatus::Conflict);
    CHECK(res.conflict.has_value());
    CHECK(res.conflict->reason == ConflictReason::IdSetOverlap);
    CHECK(std::find(res.conflict->ids.begin(), res.conflict->ids.end(), a1.id()) !=
          res.conflict->ids.end());
    CHECK(std::find(res.conflict->ids.begin(), res.conflict->ids.end(), a2.id()) !=
          res.conflict->ids.end());
    // a3 never collided -- it must NOT be reported as an overlap id.
    CHECK(std::find(res.conflict->ids.begin(), res.conflict->ids.end(), a3.id()) ==
          res.conflict->ids.end());
}

// The documented wrinkle on CommitResult::snapshot: an empty Transaction
// (no creates/updates/removes) takes try_commit()'s fast path and returns
// Committed with the TRANSACTION'S OWN base() as its snapshot -- even if a
// concurrent commit has since moved the model to a later version. "Nothing
// changed, so any version is after this commit" is the doc comment's own
// reasoning; this test proves the returned snapshot is the STALE base, not
// whatever is latest at the moment try_commit() actually runs.
TEST(empty_transaction_fast_path_returns_the_transactions_own_possibly_stale_base_as_its_snapshot) {
    Model m;
    make_account(m, "A1");

    Transaction empty_txn = m.begin();  // pins base() at version V; never touched
    const std::uint64_t v = empty_txn.base_version();

    // A DIFFERENT transaction commits something unrelated, advancing the
    // model past V.
    make_account(m, "A2");
    CHECK(m.current_version() > v);

    const CommitResult res = m.try_commit(empty_txn);
    CHECK(res.status == CommitStatus::Committed);
    CHECK_EQ(res.snapshot.version(), v);
    CHECK(res.snapshot.version() != m.current_version());
}

// Diagnostics::reap_backlog is documented as == reap_backlog() -- both read
// the SAME underlying commit_mu_-protected state, just through two
// different public accessors (a single-value one, and the full readout).
// Churn some removes while a Snapshot holds an older version pinned (so
// there's a genuine non-zero backlog), and confirm the two calls agree.
TEST(diagnostics_reap_backlog_matches_model_reap_backlog) {
    Model m;
    const Ref<Account> a = make_account(m, "A1");
    std::vector<Ref<Order>> orders;
    for (int i = 0; i < 5; ++i) orders.push_back(make_order(m, "O" + std::to_string(i), a));

    Snapshot pin = m.snapshot();  // holds the pre-removal version alive
    for (const Ref<Order>& o : orders) remove_and_commit(m, o);

    CHECK(m.reap_backlog() > 0);
    CHECK_EQ(m.diagnostics().reap_backlog, m.reap_backlog());
}

// Diagnostics::slots_free/slots_exhausted track exactly what their doc
// comments say: (a) a removed, reclaimed object's slot becomes available
// again (free_slots_ grows); (b) debug_set_generation()-forced exhaustion
// (same testing seam the single-threaded exhaustion tests in test_basics.cpp
// use) is reflected in slots_exhausted, in lockstep with exhausted_slots().
TEST(diagnostics_slots_free_and_slots_exhausted_track_recycling) {
    Model m;
    const Ref<Account> a = make_account(m, "A1");
    const Ref<Order> o1 = make_order(m, "O1", a);
    const std::uint32_t slot = o1.id().index;

    const std::size_t free_before = m.diagnostics().slots_free;
    remove_and_commit(m, o1);
    CHECK_EQ(m.wait_for_reclamation(), std::size_t{0});  // nothing pins the old version here
    CHECK_EQ(m.diagnostics().slots_free, free_before + 1);

    // Force that freed slot toward exhaustion (mirrors
    // an_exhausted_slot_is_retired_not_reused in test_basics.cpp), then
    // trigger a create so alloc_slot() actually pops and retires it.
    m.debug_set_generation(slot, kGenMax);
    make_order(m, "O2", a);

    CHECK_EQ(m.exhausted_slots(), std::size_t{1});
    CHECK_EQ(m.diagnostics().slots_exhausted, m.exhausted_slots());
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
    std::size_t saw_account = 0, saw_order = 0;
    for (const auto& tc : d.live_by_type) {
        CHECK_EQ(tc.live_count, std::size_t{1});  // one Account, one Order
        // Tight, not a loose ||: exactly one entry must name each type --
        // a build that (bug) reported the SAME name for both types would
        // still pass a plain "Account" || "Order" check on both entries.
        if (tc.type_name.find("Account") != std::string::npos) ++saw_account;
        if (tc.type_name.find("Order") != std::string::npos) ++saw_order;
    }
    CHECK_EQ(saw_account, std::size_t{1});
    CHECK_EQ(saw_order, std::size_t{1});
    CHECK(d.chunk_count >= std::size_t{1});
    CHECK(d.slots_allocated >= std::size_t{2});

    // Account::name (define_keys), Order::computed_key (define_keys).
    CHECK_EQ(d.key_indexed_fields, std::size_t{2});
    // Order::qty, Order::computed_key (define_fields, LookupType::Cache).
    CHECK_EQ(d.cached_value_indexed_fields, std::size_t{2});
    // Order::account (define_references, LookupType::Cache) -- parent is deliberately Scan-tagged.
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

// transactions_begun counts Transactions MINTED via begin(), committed or
// not -- deliberately not a commit count (that's what the commits_* fields
// above are for). Every begin() overload feeds the same next_txn_id_
// counter, including Snapshot::begin() and a Transaction dropped without
// ever being committed.
TEST(diagnostics_transactions_begun_counts_every_begin_committed_or_not) {
    Model m;
    CHECK_EQ(m.diagnostics().transactions_begun, std::uint64_t{0});

    Transaction t1 = m.begin();
    Transaction t2 = m.begin();
    { Transaction t3 = m.begin(); }  // dropped, never committed -- still counted
    CHECK_EQ(m.diagnostics().transactions_begun, std::uint64_t{3});

    CHECK(m.try_commit(t1).status == CommitStatus::Committed);  // empty commit -- still just a mint
    CHECK_EQ(m.diagnostics().transactions_begun, std::uint64_t{3});  // committing mints nothing new

    Snapshot s = m.snapshot();
    Transaction t4 = s.begin();  // routes through the same counter
    CHECK_EQ(m.diagnostics().transactions_begun, std::uint64_t{4});
    CHECK(t2.id() < m.diagnostics().transactions_begun);
    CHECK_EQ(t4.id(), m.diagnostics().transactions_begun);  // ids are minted sequentially from 1
}

// ObjectBase::type() -- reached in every test only transitively, via
// Diagnostics::TypeCount::type_name. Checked directly here: it demangles to
// something containing the type's own name, differs across types (as an
// address, not just as a string), and is stable (the SAME address) across
// two different instances of the same type -- the function-local static
// Object<Derived>::type() computes once per Derived, not once per object.
TEST(object_base_type_reports_a_readable_per_type_name_and_is_stable_across_instances) {
    Model m;
    const Ref<Account> a1 = make_account(m, "A1");
    const Ref<Account> a2 = make_account(m, "A2");
    const Ref<Order> o = make_order(m, "O1", a1);
    Snapshot s = m.snapshot();

    const char* account_type = s.find(a1)->type();
    CHECK(std::string(account_type).find("Account") != std::string::npos);
    const char* order_type = s.find(o)->type();
    CHECK(std::string(order_type).find("Order") != std::string::npos);
    CHECK(account_type != order_type);  // different types: different addresses

    CHECK_EQ(s.find(a1)->type(), s.find(a2)->type());  // same type, two instances: SAME address
    CHECK_EQ(account_type, s.find(a1)->type());
}

// Drives every CommitStatus at least once and checks each lands in its own
// counter, cumulatively -- one bucket per Diagnostics::commits_* field, no
// cross-contamination between them.
TEST(diagnostics_commit_outcome_counters_tally_every_status_independently) {
    Model m;
    const Model::Diagnostics before = m.diagnostics();

    // Committed.
    const Ref<Account> a = make_account(m, "A1", 0);

    // Conflict (IdSetOverlap): two transactions racing the same id.
    {
        Transaction t1 = m.begin(m.snapshot());
        Transaction t2 = m.begin(m.snapshot());
        t1.update(a)->balance = 111;
        t2.update(a)->balance = 222;
        CHECK(m.try_commit(t1).status == CommitStatus::Committed);
        CHECK(m.try_commit(t2).status == CommitStatus::Conflict);
    }

    // Invalid: a create referencing an already-dead target at the txn's own base.
    {
        const Ref<Account> victim = make_account(m, "VICTIM");
        remove_and_commit(m, victim);
        Transaction txn = m.begin();
        auto o = std::make_unique<Order>();
        o->code = "BAD";
        o->account = victim;
        txn.create(std::move(o));
        CHECK(m.try_commit(txn).status == CommitStatus::Invalid);
    }

    // Vetoed: pre-commit hook says no.
    {
        m.set_pre_commit([](Model&, const Transaction&, const std::vector<Change>&) { return false; });
        Transaction txn = m.begin();
        auto o = std::make_unique<Order>();
        o->code = "VETOED";
        o->account = a;
        txn.create(std::move(o));
        CHECK(m.try_commit(txn).status == CommitStatus::Vetoed);
        m.set_pre_commit({});
    }

    // PrecommitConflict: a pre-transaction fails, so the main transaction never runs.
    {
        m.set_pre_transactions([](Model& model, const Transaction&) {
            Transaction pre = model.begin();
            auto o = std::make_unique<Order>();
            o->code = "PRE_BAD";  // account left default -> null non-nullable Ref
            pre.create(std::move(o));
            (void)model.run_pre_transaction_without_undo(pre);
        });
        Transaction txn = m.begin();
        auto o = std::make_unique<Order>();
        o->code = "NEVER_APPLIED";
        o->account = a;
        txn.create(std::move(o));
        CHECK(m.try_commit(txn).status == CommitStatus::PrecommitConflict);
        m.set_pre_transactions({});
    }

    const Model::Diagnostics after = m.diagnostics();
    // Several Committed calls happen along the way as setup (seeding `a`,
    // t1's winning update, VICTIM's create and remove) -- assert only that
    // it moved, not an exact count, so this stays independent of exactly
    // how much setup each case needs. The failing pre-transaction's own
    // Invalid result (run_pre_transaction_without_undo, inside the
    // PrecommitConflict case) is never counted here at all -- only the
    // outer try_commit() call that wraps it is, and that one comes back
    // PrecommitConflict, not Invalid -- see record_commit_outcome().
    CHECK(after.commits_succeeded > before.commits_succeeded);
    CHECK_EQ(after.commits_conflicted, before.commits_conflicted + 1);
    CHECK_EQ(after.commits_invalid, before.commits_invalid + 1);
    CHECK_EQ(after.commits_vetoed, before.commits_vetoed + 1);
    CHECK_EQ(after.commits_precommit_conflicted, before.commits_precommit_conflicted + 1);
}

