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
#include <unordered_set>
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

// For every "true" new or updated object in the transaction (Created or
// Updated in the resolved changeset, but not ALSO Deleted -- a referenced
// local create can install and immediately cascade-remove within one
// commit per invariant 8, and that is not a surviving object worth
// walking), finds every outgoing Ref<>/Opt<> field that points OUTSIDE the
// transaction -- at an id the transaction itself did not create, update,
// or remove.
//
// This is a GENUINE pre-commit hook: it uses Model::peek_as<T>(), the
// public wrapper this project's Model gained specifically to close the gap
// PreCommitFn's own doc comment already promised ("peek_as<T> ... via the
// Model reference") but, until now, never delivered --
// Transaction::peek_as<T> can't help here, since apply has already moved
// the contents out of local_created_/local_updated_ by the time this hook
// runs (see PreCommitFn's doc comment). Because this runs BEFORE publish, a
// real caller could use exactly this computation to veto a commit (e.g.
// "reject any order referencing an account outside a whitelist") -- a
// post-commit hook structurally cannot do that, since by then it's too
// late.
TEST(pre_commit_hook_finds_outgoing_refs_pointing_outside_the_transaction) {
    Model m;
    const Ref<Account> a1 = make_account(m, "A1");  // to_update's OLD target -- must NOT appear
    const Ref<Account> a2 = make_account(m, "A2");  // referenced by a brand-new object
    const Ref<Account> a3 = make_account(m, "A3");  // referenced by a create that survives cascade
    const Ref<Account> a4 = make_account(m, "A4");  // to_update's NEW target
    const Ref<Order> existing = make_order(m, "EXISTING", a1);  // referenced by a new object
    const Ref<Order> to_update = make_order(m, "UPD", a1);      // updated this transaction

    std::vector<Id> outside;
    m.set_pre_commit([&](Model& model, const Transaction&, const std::vector<Change>& changes) {
        // `alive`: insert on Created/Updated, ERASE on Deleted -- reflects
        // the FINAL state after replaying every change in order, unlike a
        // separately-accumulated "ever seen as Deleted" set, which would
        // wrongly treat an id as permanently excluded even if a later
        // Created/Updated entry for that same id showed up afterward.
        std::unordered_set<Id, IdHash> alive;
        for (const Change& c : changes) {
            if (c.kind == ChangeKind::Deleted)
                alive.erase(c.id);
            else
                alive.insert(c.id);
        }

        std::unordered_set<Id, IdHash> referenced;
        for (const Change& c : changes) {
            if (!alive.count(c.id) || c.tag != type_tag<Order>()) continue;
            const Order* o = model.peek_as<Order>(c.id);
            if (!o) continue;
            referenced.insert(o->account.raw());
            if (o->parent) referenced.insert(o->parent.raw());
        }
        for (Id id : referenced)
            if (!alive.count(id)) outside.push_back(id);
        return true;
    });

    Transaction txn = m.begin();

    // A referenced local create that gets cascade-removed within the same
    // transaction -- its outgoing ref (to a3) must NOT count, since it
    // never survives to be a "true" new object.
    auto victim = std::make_unique<Order>();
    victim->code = "VICTIM";
    victim->account = a3;
    const Ref<Order> victim_local = txn.create(std::move(victim));

    auto survivor = std::make_unique<Order>();
    survivor->code = "SURVIVOR";
    survivor->account = a3;  // this one DOES count -- survivor survives
    survivor->parent = victim_local;
    txn.create(std::move(survivor));

    txn.remove(victim_local);  // referenced by pending survivor -> deferred, not cancelled outright

    // A clean, brand-new object referencing two pre-existing, otherwise
    // untouched objects.
    auto fresh = std::make_unique<Order>();
    fresh->code = "FRESH";
    fresh->account = a2;
    fresh->parent = existing;
    txn.create(std::move(fresh));

    txn.update(to_update)->account = a4;  // repoint away from a1, toward a4

    const CommitResult res = m.try_commit(txn);
    CHECK(res.status == CommitStatus::Committed);
    m.set_pre_commit({});

    CHECK_EQ(outside.size(), std::size_t{4});
    for (Id id : {a2.raw(), a3.raw(), a4.raw(), existing.raw()})
        CHECK(std::find(outside.begin(), outside.end(), id) != outside.end());
    // a1 was to_update's OLD target -- no longer referenced after the
    // update, so reading it would mean the hook saw a stale value instead
    // of the final one (invariant 9).
    CHECK(std::find(outside.begin(), outside.end(), a1.raw()) == outside.end());
}

// The type-agnostic version of the test above: the hook above hardcodes
// "if tag == type_tag<Order>()" because it only ever has to deal with one
// ref-bearing type. A model with many object types can't do that -- but it
// doesn't need to, because ObjectBase::each_ref() already type-erases
// define_references() for exactly this purpose (see Object<Derived>::
// each_ref, the same virtual dispatch cascade/nulling themselves use).
// `peek_raw(id)->each_ref(fn)` walks Account, Order, Node, Record, and Link
// here with NOT ONE type-specific branch in the hook -- it doesn't even
// need to know these five types exist.
TEST(pre_commit_hook_finds_outgoing_refs_across_many_object_types_with_no_per_type_dispatch) {
    Model m;
    const Ref<Account> a1 = make_account(m, "A1");
    const Ref<Account> a2 = make_account(m, "A2");
    const Ref<Order> o_old = make_order(m, "OLD", a1);
    const Ref<Node> n_old = make_node(m, "N_OLD");

    Transaction rt = m.begin();
    auto r = std::make_unique<Record>();
    r->label = "R_OLD";
    r->owner = a1;
    const Ref<Record> r_old_local = rt.create(std::move(r));
    const Ref<Record> r_old = commit_ok(m, rt).to_real(r_old_local);

    Transaction lt = m.begin();
    const Ref<Link> l_a_local = lt.create(std::make_unique<Link>());
    lt.update(l_a_local)->next = l_a_local;  // self-loop, just to have a valid non-nullable Ref<>
    const Ref<Link> l_b_local = lt.create(std::make_unique<Link>());
    lt.update(l_b_local)->next = l_b_local;
    const CommitResult lres = commit_ok(m, lt);
    const Ref<Link> l_a = lres.to_real(l_a_local);
    const Ref<Link> l_b = lres.to_real(l_b_local);

    std::vector<std::pair<TypeTag, Id>> outside;
    m.set_pre_commit([&](Model& model, const Transaction&, const std::vector<Change>& changes) {
        std::unordered_set<Id, IdHash> alive;
        for (const Change& c : changes) {
            if (c.kind == ChangeKind::Deleted)
                alive.erase(c.id);
            else
                alive.insert(c.id);
        }

        std::unordered_set<Id, IdHash> referenced;
        for (const Change& c : changes) {
            if (!alive.count(c.id)) continue;
            const ObjectBase* o = model.peek_raw(c.id);
            if (!o) continue;
            o->each_ref([&](const void*, const char*, Id target, bool) { referenced.insert(target); });
        }
        for (Id id : referenced) {
            if (alive.count(id)) continue;  // inside the transaction -- not "outside"
            if (const ObjectBase* o = model.peek_raw(id)) outside.emplace_back(o->tag(), id);
        }
        return true;
    });

    Transaction txn = m.begin();

    // Order: a non-nullable Ref<Account> to an existing account, and an
    // Opt<Order> to an existing order.
    auto o1 = std::make_unique<Order>();
    o1->code = "O1";
    o1->account = a2;
    o1->parent = o_old;
    const Ref<Order> o1_local = txn.create(std::move(o1));

    // Order again, but its Opt<Order> now points at o1 -- INSIDE this same
    // transaction. Must NOT show up in `outside`.
    auto o2 = std::make_unique<Order>();
    o2->code = "O2";
    o2->account = a1;
    o2->parent = o1_local;
    txn.create(std::move(o2));

    // Node: an Opt<Node> to an existing node.
    auto n1 = std::make_unique<Node>();
    n1->label = "N1";
    n1->parent = n_old;
    txn.create(std::move(n1));

    // Record: a non-nullable Ref<Account> AND an Opt<Record>, both to
    // existing objects.
    auto rec1 = std::make_unique<Record>();
    rec1->label = "REC1";
    rec1->owner = a1;
    rec1->related = r_old;
    txn.create(std::move(rec1));

    // Link: UPDATE an existing one, repointing its non-nullable Ref<Link>
    // away from its own self-loop toward a different existing Link.
    txn.update(l_a)->next = l_b;

    const CommitResult res = m.try_commit(txn);
    CHECK(res.status == CommitStatus::Committed);
    m.set_pre_commit({});

    const auto contains = [&](TypeTag tag, Id id) {
        return std::find(outside.begin(), outside.end(), std::pair<TypeTag, Id>{tag, id}) !=
               outside.end();
    };
    CHECK_EQ(outside.size(), std::size_t{6});
    CHECK(contains(type_tag<Account>(), a1.raw()));  // referenced by o2.account AND rec1.owner
    CHECK(contains(type_tag<Account>(), a2.raw()));
    CHECK(contains(type_tag<Order>(), o_old.raw()));
    CHECK(contains(type_tag<Node>(), n_old.raw()));
    CHECK(contains(type_tag<Record>(), r_old.raw()));
    CHECK(contains(type_tag<Link>(), l_b.raw()));
    CHECK(!contains(type_tag<Order>(), o1_local.raw()));       // stale local id, never real
    CHECK(!contains(type_tag<Order>(), res.to_real(o1_local).raw()));  // real, but inside the txn
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

// A vetoed reassignment must restore the old edge AND remove the phantom
// new one, or a later cascade sees a lie -- see the in-body comment.
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

// Same undo-log discipline for by_key_: a vetoed rename must leave the
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

    make_account(m, "UNRELATED");  // publish a fresh root carrying by_key_
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
        CommitResult r = model.run_pre_transaction_without_undo(pre);
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
        CommitResult r = model.run_pre_transaction_without_undo(pre);
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
        CHECK(model.run_pre_transaction_without_undo(pre1).status == CommitStatus::Committed);

        Transaction pre2 = model.begin();
        auto o = std::make_unique<Order>();
        o->code = "PRE_BAD";  // null non-nullable Ref -> Invalid
        pre2.create(std::move(o));
        CHECK(model.run_pre_transaction_without_undo(pre2).status == CommitStatus::Invalid);
        // A well-behaved hook stops calling run_pre_transaction_without_undo() here.
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

// The variant of a_failing_pre_transaction_reports_precommit_conflict_and_
// skips_the_main_transaction where the pre-transaction itself loses a real
// Conflict (not Invalid): its Transaction is built BEFORE an unrelated,
// ordinary commit touches the same object (Transaction-building never takes
// commit_mu_ -- invariant 10 -- so this is legal to do ahead of installing
// any hook at all), making it genuinely stale by the time the hook finally
// runs it via run_pre_transaction_without_undo(). try_commit() must still
// report PrecommitConflict for the main attempt, but this time res.conflict
// -- not just res.error -- is populated, carrying the underlying Conflict's
// own reason/ids through unchanged (see CommitResult::conflict's own doc
// comment: "also set if status == PrecommitConflict and the failing
// pre-transaction's own outcome was itself a Conflict").
TEST(precommit_conflict_reports_conflict_reason_when_the_failing_pretransaction_was_itself_a_conflict) {
    Model m;
    const Ref<Account> a = make_account(m, "SHARED", 0);

    // Built now, from a base that is about to go stale -- this is the
    // Transaction the hook will (much later) hand to
    // run_pre_transaction_without_undo().
    Transaction pre = m.begin();
    pre.update(a)->name = "FROM_PRE";

    // An ordinary, unrelated commit that touches the SAME object -- by the
    // time `pre` actually runs, its base predates this, so it is doomed to
    // an IdSetOverlap Conflict, exactly like two racing writer threads.
    update_field(m, a, [](Account* p) { p->balance = 111; });

    m.set_pre_transactions([&](Model& model, const Transaction&) {
        CommitResult r = model.run_pre_transaction_without_undo(pre);
        CHECK(r.status == CommitStatus::Conflict);
    });

    Transaction txn = m.begin();
    auto o = std::make_unique<Account>();
    o->name = "NEVER_APPLIED";
    txn.create(std::move(o));

    const CommitResult res = m.try_commit(txn);
    CHECK(res.status == CommitStatus::PrecommitConflict);
    CHECK(res.conflict.has_value());  // forwarded from the failing pre-transaction's OWN Conflict
    CHECK(res.conflict->reason == ConflictReason::IdSetOverlap);
    CHECK(std::find(res.conflict->ids.begin(), res.conflict->ids.end(), a.raw()) !=
          res.conflict->ids.end());
    CHECK_EQ(m.snapshot().size(), std::size_t{1});  // main txn never applied

    m.set_pre_transactions({});
    CHECK(m.snapshot().find_by_key<&Account::name>("NEVER_APPLIED") == nullptr);
    // pre's stale rename never landed; the unrelated commit's balance did.
    CHECK_EQ(m.snapshot().find(a)->name, std::string("SHARED"));
    CHECK_EQ(m.snapshot().find(a)->balance, std::int64_t{111});
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
        CHECK(model.run_pre_transaction_without_undo(pre).status == CommitStatus::Committed);
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
        CHECK(model.run_pre_transaction_without_undo(pre).status == CommitStatus::Committed);
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
        CommitResult r = model.run_pre_transaction_without_undo(pre);
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

// Computes two derived views of a commit's resolved changeset:
//   1. "True" new objects: Created, but not ALSO Deleted in the same
//      changeset -- i.e. NOT a same-transaction create that only ever got
//      installed to be immediately cascade-removed (a referenced local
//      create per invariant 8; see removing_a_referenced_local_create_
//      cascades_at_commit_like_a_committed_remove in test_cascade.cpp).
//   2. Objects that are themselves UNCHANGED by this transaction, but are
//      referenced (via any Ref<>/Opt<> field) by something the transaction
//      DID create or update.
//
// This is deliberately a POST-commit hook, even though it's answering the
// question a PRE-commit hook was asked to answer: PreCommitFn runs AFTER
// apply but BEFORE publish, and by that point txn's local_created_/
// local_updated_ have already been moved from during apply (see
// PreCommitFn's own doc comment) -- there is no public way to read a
// created/updated object's FIELD VALUES from inside a pre-commit hook, only
// its id/kind/tag via the Change list itself (Model::peek(Id) exists, but
// is private; Transaction::peek_as<T> reads the transaction's OWN overlay,
// which is exactly what's been moved from by the time the hook runs).
// PostCommitFn runs after commit_mu_ is released, with the real,
// newly-published CommitResult::snapshot to read from instead -- everything
// needed to walk references. The a1/a4 assertions below exist specifically
// to prove the computation reads the FINAL value (invariant 9), not a
// stale one: to_update is repointed from a1 to a4 mid-transaction, and only
// a4 -- never a1 -- may show up in the "unchanged, but referenced" list.
TEST(post_commit_hook_computes_new_objects_and_unchanged_referenced_objects) {
    Model m;
    const Ref<Account> a1 = make_account(m, "A1");  // to_update's OLD target -- must NOT appear
    const Ref<Account> a2 = make_account(m, "A2");  // referenced by a brand-new object
    const Ref<Account> a3 = make_account(m, "A3");  // referenced by a create that survives cascade
    const Ref<Account> a4 = make_account(m, "A4");  // to_update's NEW target
    const Ref<Order> existing = make_order(m, "EXISTING", a1);  // referenced by a new object
    const Ref<Order> to_update = make_order(m, "UPD", a1);      // updated this transaction

    std::vector<Id> new_objects;
    std::vector<Id> unchanged_referenced;
    m.set_post_commit([&](Model&, const Transaction&, const CommitResult& r) {
        if (r.status != CommitStatus::Committed) return;

        // `created`: ever appeared as a Created change -- a plain historical
        // fact, unaffected by anything that happens to the id afterward.
        // `alive`: insert on Created/Updated, ERASE on Deleted -- reflects
        // the FINAL state after replaying every change in order, unlike a
        // separately-accumulated "ever seen as Deleted" set, which would
        // wrongly treat an id as permanently excluded even if a later
        // Created/Updated entry for that same id showed up afterward.
        std::unordered_set<Id, IdHash> created, alive;
        for (const Change& c : r.changes) {
            if (c.kind == ChangeKind::Created) created.insert(c.id);
            if (c.kind == ChangeKind::Deleted)
                alive.erase(c.id);
            else
                alive.insert(c.id);
        }

        // 1. Truly new: created AND still alive (not cascade-removed within
        // this same commit).
        for (Id id : created)
            if (alive.count(id)) new_objects.push_back(id);

        // 2. Referenced-but-unchanged: walk every surviving Created/Updated
        // object's ref fields against the final, just-published snapshot.
        // Account has no reference fields, so only Order contributes.
        std::unordered_set<Id, IdHash> referenced;
        for (const Change& c : r.changes) {
            if (!alive.count(c.id) || c.tag != type_tag<Order>()) continue;
            const Order* o = r.snapshot.find<Order>(Ref<Order>(c.id));
            if (!o) continue;
            referenced.insert(o->account.raw());
            if (o->parent) referenced.insert(o->parent.raw());
        }
        for (Id id : referenced)
            if (!alive.count(id)) unchanged_referenced.push_back(id);
    });

    Transaction txn = m.begin();

    // A referenced local create that gets cascade-removed within the same
    // transaction -- must NOT show up in new_objects.
    auto victim = std::make_unique<Order>();
    victim->code = "VICTIM";
    victim->account = a3;
    const Ref<Order> victim_local = txn.create(std::move(victim));

    auto survivor = std::make_unique<Order>();
    survivor->code = "SURVIVOR";
    survivor->account = a3;
    survivor->parent = victim_local;  // references the about-to-be-removed victim
    const Ref<Order> survivor_local = txn.create(std::move(survivor));

    txn.remove(victim_local);  // referenced by pending `survivor` -> deferred, not cancelled outright

    // A clean, brand-new object referencing two pre-existing, otherwise
    // untouched objects.
    auto fresh = std::make_unique<Order>();
    fresh->code = "FRESH";
    fresh->account = a2;
    fresh->parent = existing;
    const Ref<Order> fresh_local = txn.create(std::move(fresh));

    txn.update(to_update)->account = a4;  // repoint away from a1, toward a4

    const CommitResult res = m.try_commit(txn);
    CHECK(res.status == CommitStatus::Committed);
    m.set_post_commit({});

    // --- 1: truly new objects --- (Id has no operator<, only ==, so this
    // checks set membership directly rather than sorting)
    CHECK_EQ(new_objects.size(), std::size_t{2});
    CHECK(std::find(new_objects.begin(), new_objects.end(), res.to_real(survivor_local).raw()) !=
          new_objects.end());
    CHECK(std::find(new_objects.begin(), new_objects.end(), res.to_real(fresh_local).raw()) !=
          new_objects.end());
    CHECK(std::find(new_objects.begin(), new_objects.end(), res.to_real(victim_local).raw()) ==
          new_objects.end());  // cancelled-via-cascade -- filtered out

    // --- 2: unchanged objects referenced by created/updated ones ---
    CHECK_EQ(unchanged_referenced.size(), std::size_t{4});
    for (Id id : {a2.raw(), a3.raw(), a4.raw(), existing.raw()})
        CHECK(std::find(unchanged_referenced.begin(), unchanged_referenced.end(), id) !=
              unchanged_referenced.end());
    CHECK(std::find(unchanged_referenced.begin(), unchanged_referenced.end(), a1.raw()) ==
          unchanged_referenced.end());  // stale target -- reading it would be a reconciliation bug
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

