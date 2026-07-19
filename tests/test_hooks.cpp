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


// A pre-commit hook returning true sees the resolved changeset and lets
// the commit through unmodified.
TEST(pre_commit_hook_approving_commits_normally) {
    Model m;
    int seen = 0;
    m.set_pre_commit([&](Model&, const Transaction&, const std::vector<Change>& changes) {
        seen = static_cast<int>(changes.size());
        return true;
    });

    make_account(m, "A1");
    CHECK_EQ(seen, 1);
    CHECK_EQ(m.snapshot().size(), std::size_t{1});
}

// try_commit() on a Transaction with nothing pending is a no-op fast
// path that never calls the pre-commit hook at all.
TEST(pre_commit_hook_is_not_invoked_for_an_empty_commit) {
    Model m;
    bool called = false;
    m.set_pre_commit([&](Model&, const Transaction&, const std::vector<Change>&) {
        called = true;
        return true;
    });

    Transaction txn = m.begin();  // nothing pending
    CommitResult res = m.try_commit(txn);
    CHECK(res.status == CommitStatus::Committed);
    CHECK(!called);
}

// The hook runs AFTER cascade resolution, so it observes every
// cascade-deleted object too -- not just the ones the Transaction's own
// remove() intent explicitly named.
TEST(pre_commit_hook_sees_the_fully_resolved_changeset_including_cascade_deletes) {
    Model m;
    const Ref<Account> a = make_account(m, "A1");
    const Ref<Order> o = make_order(m, "O1", a);

    std::size_t seen_deletes = 0;
    m.set_pre_commit([&](Model&, const Transaction&, const std::vector<Change>& changes) {
        for (const Change& c : changes)
            if (c.kind == ChangeKind::Deleted) ++seen_deletes;
        return true;
    });

    const std::size_t killed = remove_and_commit(m, a);  // cascades to o
    CHECK_EQ(killed, std::size_t{2});
    CHECK_EQ(seen_deletes, std::size_t{2});
    (void)o;
}

// A hook returning false (Vetoed) leaves the model in EXACTLY its
// pre-attempt state (checked via state_of()), and a later attempt still
// commits normally once the hook is cleared.
TEST(pre_commit_hook_veto_unwinds_everything_as_if_try_commit_were_never_called) {
    Model m;
    const Ref<Account> a = make_account(m, "A1");
    const std::string before = state_of(m);

    m.set_pre_commit([](Model&, const Transaction&, const std::vector<Change>&) { return false; });

    Transaction txn = m.begin();
    auto o = std::make_unique<Order>();
    o->code = "O1";
    o->account = a;
    txn.create(std::move(o));
    CommitResult res = m.try_commit(txn);
    CHECK(res.status == CommitStatus::Vetoed);
    CHECK_EQ(state_of(m), before);
    CHECK(m.snapshot().find_by_key<&Order::computed_key>("ord:O1") == nullptr);

    m.set_pre_commit({});
    const Ref<Order> o2 = make_order(m, "O2", a);
    CHECK(m.snapshot().find(o2) != nullptr);
}

// Regression test for the reconcile_out_refs undo-log bug -- see
// the in-body comment. A vetoed reassignment must restore the old edge
// AND remove the phantom new one, or a later cascade sees a lie.
TEST(veto_rollback_restores_the_reverse_index_so_later_cascades_stay_correct) {
    Model m;
    const Ref<Account> x = make_account(m, "X");
    const Ref<Account> y = make_account(m, "Y");
    const Ref<Order> o = make_order(m, "O1", x);

    // Repoint O's Ref<Account> from X to Y, then get vetoed. The rollback
    // must also undo the reconcile step's referrers_ edits (X->O edge erased,
    // Y->O edge added), or the committed state (O still points at X) and the
    // reverse index disagree forever.
    m.set_pre_commit([](Model&, const Transaction&, const std::vector<Change>&) { return false; });
    Transaction txn = m.begin();
    txn.update(o)->account = y;
    CHECK(m.try_commit(txn).status == CommitStatus::Vetoed);
    m.set_pre_commit({});
    CHECK(m.snapshot().find(o)->account == x);

    // No phantom edge: removing Y must not drag O along.
    CHECK_EQ(remove_and_commit(m, y), std::size_t{1});
    CHECK(m.snapshot().find(o) != nullptr);

    // No lost edge: removing X must cascade O, not strand it dangling.
    CHECK_EQ(remove_and_commit(m, x), std::size_t{2});
    CHECK(m.snapshot().find(o) == nullptr);
}

// Same undo-log discipline for by_field_: a vetoed rename must leave the
// OLD value still findable and the NEW value NOT findable, once a later
// commit actually publishes.
TEST(veto_rollback_restores_the_field_key_index) {
    Model m;
    const Ref<Account> a = make_account(m, "OLD");

    m.set_pre_commit([](Model&, const Transaction&, const std::vector<Change>&) { return false; });
    Transaction txn = m.begin();
    txn.update(a)->name = "NEW";
    CHECK(m.try_commit(txn).status == CommitStatus::Vetoed);
    m.set_pre_commit({});

    make_account(m, "UNRELATED");  // publish a fresh root carrying by_field_
    Snapshot s = m.snapshot();
    CHECK_EQ(s.find(a)->name, std::string("OLD"));
    CHECK(s.find_by_key<&Account::name>("OLD") == s.find(a));
    CHECK(s.find_by_key<&Account::name>("NEW") == nullptr);
}

// A structurally invalid Transaction (a null non-nullable Ref alongside
// otherwise-valid creates) is rejected as CommitStatus::Invalid, fully
// unwound, with the specific violation reported via CommitResult::error.
TEST(an_invalid_transaction_unwinds_like_a_veto_and_reports_why) {
    Model m;
    const std::string before = state_of(m);

    Transaction txn = m.begin();
    auto g = std::make_unique<Account>();
    g->name = "GHOST";
    txn.create(std::move(g));
    auto bad = std::make_unique<Order>();
    bad->code = "BAD";  // account left default -> null non-nullable Ref
    txn.create(std::move(bad));

    const CommitResult res = m.try_commit(txn);
    CHECK(res.status == CommitStatus::Invalid);
    CHECK(res.error.has_value());
    CHECK(!res.error->bad_target);  // null Ref: nothing to classify against base
    CHECK_EQ(state_of(m), before);

    // The next commit must publish ONLY its own changes -- nothing left over
    // from the rejected attempt (its slot writes, index entries, changes_,
    // and undo log must all have been unwound before try_commit returned).
    const Ref<Account> real = make_account(m, "REAL");
    Snapshot s = m.snapshot();
    CHECK(s.find(real) != nullptr);
    CHECK(s.find_by_key<&Account::name>("GHOST") == nullptr);
    CHECK_EQ(s.size(), std::size_t{1});
}

// TSan-targeted: see the in-body comment. Installing/clearing the hook
// concurrently with a hammering committer thread must never race --
// set_pre_commit() takes commit_mu_ the same way try_commit() does.
TEST(set_pre_commit_can_race_try_commit_without_a_data_race) {
    // The real assertion here is TSan's (ctest --preset tsan): installing a
    // hook must be serialized against the commits that invoke it.
    Model m;
    std::atomic<bool> stop{false};
    std::thread committer([&] {
        int n = 0;
        // do-while: at least one commit lands even if the main thread finishes
        // all its set_pre_commit() calls before this thread is scheduled.
        do {
            Transaction txn = m.begin();
            auto a = std::make_unique<Account>();
            a->name = "A" + std::to_string(n++);
            txn.create(std::move(a));
            (void)m.try_commit(txn);
        } while (!stop.load(std::memory_order_relaxed));
    });
    for (int i = 0; i < 500; ++i)
        m.set_pre_commit(
            [](Model&, const Transaction&, const std::vector<Change>&) { return true; });
    m.set_pre_commit({});
    stop = true;
    committer.join();
    CHECK(m.snapshot().size() > 0);
}

// Try to break invariant 10 (snapshot()/Transaction-building never take
// commit_mu_) WITHOUT a wall-clock guess: a pre-commit hook parks a
// committer thread INSIDE the commit_mu_-held critical section (pre_commit_
// runs from commit_main_locked(), under the same lock_guard try_commit()
// takes -- see model.cpp) until released, and release only happens AFTER a
// concurrent snapshot() call -- issued from the main thread only once the
// hook is PROVABLY parked -- has already returned. If snapshot() took
// commit_mu_ even briefly, that snapshot() call could never return (nothing
// else ever releases the hook), and this test hangs until ctest's own
// TIMEOUT kills it: a deadlock is a hang here, not a flaky timing check.
TEST(snapshot_never_blocks_behind_a_commit_holding_commit_mu) {
    Model m;
    make_account(m, "SEED");

    std::atomic<bool> hook_entered{false};
    std::atomic<bool> release_hook{false};
    m.set_pre_commit([&](Model&, const Transaction&, const std::vector<Change>&) {
        hook_entered.store(true, std::memory_order_release);
        while (!release_hook.load(std::memory_order_acquire)) std::this_thread::yield();
        return true;
    });

    std::thread committer([&] {
        Transaction txn = m.begin();
        auto a = std::make_unique<Account>();
        a->name = "HELD";
        txn.create(std::move(a));
        (void)m.try_commit(txn);  // parks inside the hook, holding commit_mu_ the whole time
    });

    while (!hook_entered.load(std::memory_order_acquire)) std::this_thread::yield();

    // The committer is now provably parked inside the critical section --
    // nothing releases it until AFTER this call returns.
    const Snapshot s = m.snapshot();
    CHECK_EQ(s.size(), std::size_t{1});  // pre-commit state only; the pending create isn't published yet

    release_hook.store(true, std::memory_order_release);
    committer.join();
    m.set_pre_commit({});

    CHECK_EQ(m.snapshot().size(), std::size_t{2});
}

// A create() whose Ref<> target was already dead even at the
// Transaction's OWN base is classified Invalid (a build-time bug), not
// Conflict, with bad_target naming the dead id.
TEST(create_with_a_dead_ref_is_rejected_as_invalid) {
    Model m;
    const Ref<Account> a = make_account(m, "A1");
    const Ref<Order> good = make_order(m, "GOOD", a);
    remove_and_commit(m, a);  // a and good are now dead

    Transaction txn = m.begin();
    auto o = std::make_unique<Order>();
    o->code = "BAD";
    o->account = a;  // Ref<Account> to an already-dead account (as of txn's own base)
    txn.create(std::move(o));

    // Dead even at the txn's own base: a transaction-building bug (Invalid,
    // with the offending target reported), not a concurrency Conflict.
    const CommitResult res = m.try_commit(txn);
    CHECK(res.status == CommitStatus::Invalid);
    CHECK(res.error.has_value());
    CHECK(res.error->bad_target == a.raw());
    CHECK(m.snapshot().find(good) == nullptr);  // txn's base already reflects a/good gone
}

// Leaving a non-nullable Ref<> field at its default (null) value is
// rejected as Invalid at apply time, with no target to report.
TEST(create_with_a_null_nonnullable_ref_is_rejected_as_invalid) {
    Model m;
    Transaction txn = m.begin();
    auto o = std::make_unique<Order>();
    o->code = "X";  // account left default -> null Ref<Account>
    txn.create(std::move(o));

    const CommitResult res = m.try_commit(txn);
    CHECK(res.status == CommitStatus::Invalid);
    CHECK(res.error.has_value());
    CHECK(!res.error->bad_target);  // a null Ref has no target to report
    CHECK_EQ(m.snapshot().size(), std::size_t{0});
}

// Try to break invariant 2 (RefNuller has no overload that can clear a
// Ref<T>) from the OTHER direction: not "leave it null," but "explicitly
// write a null Ref<> onto an already-live, non-nullable field via update()."
// apply_update() runs the SAME validate() as apply_create() (see
// apply_update's own call site in model.cpp), so this must be rejected
// exactly like the create-path test above -- and, being an Invalid classify
// rather than a Conflict, must unwind completely: the object's account
// field is untouched afterward, not left half-written.
TEST(update_setting_a_non_nullable_ref_to_null_is_rejected_as_invalid) {
    Model m;
    const Ref<Account> a = make_account(m, "A1");
    const Ref<Order> o = make_order(m, "O1", a);

    Transaction txn = m.begin();
    txn.update(o)->account = Ref<Account>{};  // explicitly null out a non-nullable field

    const CommitResult res = m.try_commit(txn);
    CHECK(res.status == CommitStatus::Invalid);
    CHECK(res.error.has_value());
    CHECK(!res.error->bad_target);  // a null Ref has no target to report

    // Unwound completely: o still points at a, exactly as before the attempt.
    Snapshot s = m.snapshot();
    CHECK(s.find(o)->account == a);
    CHECK(s.find(a) != nullptr);
}


// A pre-transaction run from inside the hook publishes for real, BEFORE the
// main transaction is even conflict-checked -- so the main transaction, once
// it runs, already sees the pre-transaction's object.
TEST(pre_transactions_hook_runs_and_publishes_before_the_main_transaction) {
    Model m;
    m.set_pre_transactions([](Model& model, const Transaction&) {
        Transaction pre = model.begin();
        auto a = std::make_unique<Account>();
        a->name = "PRE";
        pre.create(std::move(a));
        CommitResult r = model.run_pre_transaction(pre);
        CHECK(r.status == CommitStatus::Committed);
    });

    Transaction txn = m.begin();
    auto a = std::make_unique<Account>();
    a->name = "MAIN";
    const Ref<Account> local = txn.create(std::move(a));
    const CommitResult res = m.try_commit(txn);
    CHECK(res.status == CommitStatus::Committed);

    Snapshot s = m.snapshot();
    CHECK_EQ(s.size(), std::size_t{2});
    CHECK(s.find_by_key<&Account::name>("PRE") != nullptr);
    CHECK(s.find(res.to_real(local)) != nullptr);
    m.set_pre_transactions({});
}

// try_commit()'s existing fast path for an empty Transaction never touches
// commit_mu_ at all, so the pre-transactions hook -- like the pre-commit
// hook -- is never invoked for it either.
TEST(pre_transactions_hook_is_not_invoked_for_an_empty_main_transaction) {
    Model m;
    bool called = false;
    m.set_pre_transactions([&](Model&, const Transaction&) { called = true; });

    Transaction txn = m.begin();  // nothing pending
    CommitResult res = m.try_commit(txn);
    CHECK(res.status == CommitStatus::Committed);
    CHECK(!called);
    m.set_pre_transactions({});
}

// If a pre-transaction fails (here: Invalid, a null non-nullable Ref), the
// main transaction is never conflict-checked or applied at all -- try_commit
// reports PrecommitConflict, forwarding the failing pre-transaction's own
// `error`. Because the main transaction was never touched, `txn` is still
// fresh and can be retried as-is once the hook stops failing.
TEST(a_failing_pre_transaction_reports_precommit_conflict_and_skips_the_main_transaction) {
    Model m;
    m.set_pre_transactions([](Model& model, const Transaction&) {
        Transaction pre = model.begin();
        auto o = std::make_unique<Order>();
        o->code = "BAD";  // account left default -> null non-nullable Ref
        pre.create(std::move(o));
        CommitResult r = model.run_pre_transaction(pre);
        CHECK(r.status == CommitStatus::Invalid);
    });

    Transaction txn = m.begin();
    auto a = std::make_unique<Account>();
    a->name = "NEVER_APPLIED";
    const Ref<Account> local = txn.create(std::move(a));

    CommitResult res = m.try_commit(txn);
    CHECK(res.status == CommitStatus::PrecommitConflict);
    CHECK(res.error.has_value());                   // forwarded from the failing pre-transaction
    CHECK_EQ(m.snapshot().size(), std::size_t{0});  // main txn never applied

    // txn was never consumed -- retry it as-is once the hook is well-behaved.
    m.set_pre_transactions({});
    res = m.try_commit(txn);
    CHECK(res.status == CommitStatus::Committed);
    CHECK(m.snapshot().find(res.to_real(local)) != nullptr);
}

// A pre-transaction that succeeds stays published even if the main
// transaction is then skipped because a LATER pre-transaction in the same
// phase fails -- published state is immutable (CLAUDE.md invariant 3).
// PrecommitConflict means only "the main transaction didn't run."
TEST(an_earlier_successful_pre_transaction_stays_committed_even_if_a_later_one_fails) {
    Model m;
    m.set_pre_transactions([](Model& model, const Transaction&) {
        Transaction pre1 = model.begin();
        auto a = std::make_unique<Account>();
        a->name = "PRE_OK";
        pre1.create(std::move(a));
        CHECK(model.run_pre_transaction(pre1).status == CommitStatus::Committed);

        Transaction pre2 = model.begin();
        auto o = std::make_unique<Order>();
        o->code = "PRE_BAD";  // null non-nullable Ref -> Invalid
        pre2.create(std::move(o));
        CHECK(model.run_pre_transaction(pre2).status == CommitStatus::Invalid);
        // A well-behaved hook stops calling run_pre_transaction() here.
    });

    Transaction txn = m.begin();
    auto a = std::make_unique<Account>();
    a->name = "MAIN_NEVER_APPLIED";
    txn.create(std::move(a));

    const CommitResult res = m.try_commit(txn);
    CHECK(res.status == CommitStatus::PrecommitConflict);

    m.set_pre_transactions({});
    Snapshot s = m.snapshot();
    CHECK_EQ(s.size(), std::size_t{1});  // PRE_OK published; PRE_BAD and MAIN did not
    CHECK(s.find_by_key<&Account::name>("PRE_OK") != nullptr);
    CHECK(s.find_by_key<&Account::name>("MAIN_NEVER_APPLIED") == nullptr);
}

// The main transaction gets its conflict-checking against a pre-transaction
// for free: both target the same Account's slot, so the main transaction
// (built from a base that predates the pre-transaction) reports an ordinary
// Conflict(IdSetOverlap) -- exactly as if the pre-transaction had been made
// by a different writer thread racing it.
TEST(main_transaction_conflicts_with_a_pre_transaction_touching_the_same_object) {
    Model m;
    const Ref<Account> a = make_account(m, "SHARED");

    Transaction txn = m.begin();  // base predates the pre-transaction below
    txn.update(a)->name = "FROM_MAIN";

    m.set_pre_transactions([a](Model& model, const Transaction&) {
        Transaction pre = model.begin();
        pre.update(a)->name = "FROM_PRE";
        CHECK(model.run_pre_transaction(pre).status == CommitStatus::Committed);
    });

    const CommitResult res = m.try_commit(txn);
    CHECK(res.status == CommitStatus::Conflict);
    CHECK(res.conflict.has_value());
    CHECK(res.conflict->reason == ConflictReason::IdSetOverlap);

    m.set_pre_transactions({});
    CHECK_EQ(m.snapshot().find(a)->name, std::string("FROM_PRE"));
}

// PreCommitFn (the existing veto hook) is unaffected by, and unaware of, a
// pre-transactions phase: it still runs exactly once, after the MAIN
// transaction's own apply, seeing only the main transaction's own resolved
// changeset -- not the pre-transaction's.
TEST(pre_commit_veto_hook_still_only_sees_the_main_transactions_own_changeset) {
    Model m;
    m.set_pre_transactions([](Model& model, const Transaction&) {
        Transaction pre = model.begin();
        auto a = std::make_unique<Account>();
        a->name = "PRE";
        pre.create(std::move(a));
        CHECK(model.run_pre_transaction(pre).status == CommitStatus::Committed);
    });

    std::size_t seen = 0;
    m.set_pre_commit([&](Model&, const Transaction&, const std::vector<Change>& changes) {
        seen = changes.size();
        return true;
    });

    Transaction txn = m.begin();
    auto a = std::make_unique<Account>();
    a->name = "MAIN";
    txn.create(std::move(a));
    CHECK(m.try_commit(txn).status == CommitStatus::Committed);
    CHECK_EQ(seen, std::size_t{1});  // only MAIN's own create, not PRE's

    m.set_pre_transactions({});
    m.set_pre_commit({});
}

// Empirical proof, not just a locking argument: under heavy CONCURRENT
// contention from many other threads racing try_commit(), a pre-transaction
// and its main transaction always publish ADJACENT versions. version_ is a
// monotonic counter, incremented exactly once per publish_now() call, only
// ever while commit_mu_ is held (CLAUDE.md invariant 7). If any other
// thread's commit could land between "this thread's pre-transaction
// published" and "this thread's main transaction published", the two
// versions this SAME try_commit() call produced would not be consecutive --
// some other version would be wedged in between. Extend this test (not just
// the locking) if you ever touch the pre-transactions phase.
TEST(pre_transactions_and_the_main_transaction_publish_with_no_other_commit_landing_between_them) {
    Model m;
    // Thread-local, not shared: each racing thread runs its own
    // pre-transaction-then-main-transaction pair, and only ever needs to
    // compare its OWN pair's versions against each other.
    thread_local std::uint64_t t_pre_version = 0;

    m.set_pre_transactions([](Model& model, const Transaction&) {
        Transaction pre = model.begin();
        auto a = std::make_unique<Account>();
        a->name = "PRE";
        pre.create(std::move(a));
        CommitResult r = model.run_pre_transaction(pre);
        if (r.status == CommitStatus::Committed) t_pre_version = r.snapshot.version();
    });

    constexpr int kThreads = 8;
    constexpr int kItersPerThread = 300;
    std::atomic<int> gap_violations{0};
    std::atomic<int> committed{0};

    std::vector<std::thread> threads;
    for (int t = 0; t < kThreads; ++t) {
        threads.emplace_back([&, t] {
            for (int i = 0; i < kItersPerThread; ++i) {
                Transaction txn = m.begin();
                auto a = std::make_unique<Account>();
                a->name = "T" + std::to_string(t) + "_" + std::to_string(i);
                txn.create(std::move(a));
                CommitResult main = m.try_commit(txn);
                if (main.status != CommitStatus::Committed) continue;
                if (main.snapshot.version() != t_pre_version + 1) gap_violations.fetch_add(1);
                ++committed;
            }
        });
    }
    for (auto& th : threads) th.join();
    m.set_pre_transactions({});

    CHECK(committed.load() > 0);
    CHECK_EQ(gap_violations.load(), 0);
}


// A post-commit hook receives the SAME CommitResult the caller of
// try_commit() got back -- same status, same snapshot, same changes.
TEST(post_commit_hook_receives_the_same_commit_result_the_caller_got) {
    Model m;
    std::optional<CommitStatus> seen_status;
    std::size_t seen_changes = 0;
    Snapshot seen_snapshot;
    m.set_post_commit([&](Model&, const Transaction&, const CommitResult& r) {
        seen_status = r.status;
        seen_changes = r.changes.size();
        seen_snapshot = r.snapshot;
    });

    Transaction txn = m.begin();
    auto a = std::make_unique<Account>();
    a->name = "A1";
    txn.create(std::move(a));
    const CommitResult res = m.try_commit(txn);

    CHECK(seen_status.has_value());
    CHECK(*seen_status == CommitStatus::Committed);
    CHECK_EQ(seen_changes, std::size_t{1});
    CHECK_EQ(seen_snapshot.version(), res.snapshot.version());
    m.set_post_commit({});
}

// try_commit()'s existing fast path for an empty Transaction never takes
// commit_mu_ at all, so -- like PreCommitFn and PreTransactionsFn -- the
// post-commit hook is never invoked for it either.
TEST(post_commit_hook_is_not_invoked_for_an_empty_transaction) {
    Model m;
    bool called = false;
    m.set_post_commit([&](Model&, const Transaction&, const CommitResult&) { called = true; });

    Transaction txn = m.begin();  // nothing pending
    CommitResult res = m.try_commit(txn);
    CHECK(res.status == CommitStatus::Committed);
    CHECK(!called);
    m.set_post_commit({});
}

// The post-commit hook sees a NON-Committed result too -- here, a veto --
// with the exact same status/error info the caller of try_commit() got.
// (Contrast with PreCommitFn, which only ever runs for an attempt that is
// ABOUT to publish; PostCommitFn runs for every non-empty attempt regardless
// of how it ended.)
TEST(post_commit_hook_sees_a_vetoed_result_too) {
    Model m;
    std::optional<CommitStatus> seen_status;
    m.set_post_commit(
        [&](Model&, const Transaction&, const CommitResult& r) { seen_status = r.status; });
    m.set_pre_commit([](Model&, const Transaction&, const std::vector<Change>&) { return false; });

    Transaction txn = m.begin();
    auto a = std::make_unique<Account>();
    a->name = "VETOED";
    txn.create(std::move(a));
    const CommitResult res = m.try_commit(txn);

    CHECK(res.status == CommitStatus::Vetoed);
    CHECK(seen_status.has_value());
    CHECK(*seen_status == CommitStatus::Vetoed);
    m.set_pre_commit({});
    m.set_post_commit({});
}

// The defining difference from PreCommitFn/PreTransactionsFn: because
// commit_mu_ has ALREADY been released by the time the post-commit hook
// runs, it is safe to call try_commit() (or begin(), or snapshot()) again
// from inside it -- there is nothing left to self-deadlock against. This
// chains a second, independent transaction from inside the hook and
// confirms it actually commits.
TEST(post_commit_hook_can_safely_chain_another_try_commit_call) {
    Model m;
    // chained_once guards against exactly the risk PostCommitFn's own doc
    // comment calls out: the hook is still installed, so the CHAINED commit
    // below would itself re-trigger this same hook, chaining forever, if
    // this test didn't stop it after the first hop.
    bool chained_once = false;
    m.set_post_commit([&](Model& model, const Transaction&, const CommitResult& r) {
        if (r.status != CommitStatus::Committed || chained_once) return;
        chained_once = true;
        Transaction chained = model.begin();
        auto a = std::make_unique<Account>();
        a->name = "CHAINED";
        chained.create(std::move(a));
        CommitResult chained_res = model.try_commit(chained);
        CHECK(chained_res.status == CommitStatus::Committed);
    });

    Transaction txn = m.begin();
    auto a = std::make_unique<Account>();
    a->name = "ORIGINAL";
    txn.create(std::move(a));
    CHECK(m.try_commit(txn).status == CommitStatus::Committed);
    m.set_post_commit({});

    Snapshot s = m.snapshot();
    CHECK_EQ(s.size(), std::size_t{2});
    CHECK(s.find_by_key<&Account::name>("ORIGINAL") != nullptr);
    CHECK(s.find_by_key<&Account::name>("CHAINED") != nullptr);
}

// TSan-targeted: installing/clearing the post-commit hook concurrently with
// a hammering committer thread must never race -- set_post_commit() takes
// commit_mu_ the same way try_commit() does to copy it out.
TEST(set_post_commit_can_race_try_commit_without_a_data_race) {
    Model m;
    std::atomic<bool> stop{false};
    std::thread committer([&] {
        int n = 0;
        do {
            Transaction txn = m.begin();
            auto a = std::make_unique<Account>();
            a->name = "A" + std::to_string(n++);
            txn.create(std::move(a));
            (void)m.try_commit(txn);
        } while (!stop.load(std::memory_order_relaxed));
    });
    for (int i = 0; i < 500; ++i)
        m.set_post_commit([](Model&, const Transaction&, const CommitResult&) {});
    m.set_post_commit({});
    stop = true;
    committer.join();
    CHECK(m.snapshot().size() > 0);
}

