#include <algorithm>
#include <any>
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


// Model::begin() pins a Transaction's base() to the current committed
// version, matching a Snapshot taken at the same moment.
TEST(begin_returns_a_transaction_bound_to_the_current_snapshot) {
    Model m;
    make_account(m, "A1");

    Snapshot s = m.snapshot();
    Transaction txn = m.begin();
    CHECK_EQ(txn.base_version(), s.version());
    CHECK_EQ(txn.base().version(), m.current_version());
}

// name()/data() default to empty/nothing when begin() is called with no
// arguments, and are read back exactly as passed otherwise -- purely
// descriptive labels, never compared or dispatched on by the model itself.
TEST(transaction_name_and_data_default_to_empty_and_round_trip_when_set) {
    Model m;

    Transaction plain = m.begin();
    CHECK(plain.name().empty());
    CHECK(!plain.data().has_value());

    Transaction labeled = m.begin("import batch 7", std::any(std::int64_t{42}));
    CHECK_EQ(labeled.name(), std::string("import batch 7"));
    CHECK(labeled.data().has_value());
    CHECK_EQ(std::any_cast<std::int64_t>(labeled.data()), std::int64_t{42});
}

// A same-transaction create can reference another same-transaction
// create via its local placeholder id, and that reference resolves
// correctly to the REAL id once try_commit() remaps and publishes.
TEST(transaction_local_creates_can_reference_each_other_via_placeholder_ids) {
    Model m;
    Transaction txn = m.begin();
    auto a = std::make_unique<Account>();
    a->name = "A1";
    const Ref<Account> a_local = txn.create(std::move(a));
    // Order references an Account created earlier in the SAME, uncommitted
    // transaction -- `a_local` is still a local placeholder id here.
    auto o = std::make_unique<Order>();
    o->code = "O1";
    o->account = a_local;
    const Ref<Order> o_local = txn.create(std::move(o));
    CHECK(txn.exists(o_local));
    CHECK(txn.peek(o_local)->account == a_local);

    CommitResult res = m.try_commit(txn);
    CHECK(res.status == CommitStatus::Committed);
    const Ref<Order> o_real = res.to_real(o_local);
    Snapshot s = m.snapshot();
    CHECK(s.find(o_real) != nullptr);
    CHECK_EQ(s.resolve(s.find(o_real)->account).name, std::string("A1"));
}

// A local id is meaningless outside the Transaction that minted it: a
// second Transaction can't peek/exists/update it, and embedding it in a
// commit is rejected as Invalid -- PROVIDED its raw index doesn't collide
// with one of the committing transaction's own local indexes (the pre-mint
// pass maps every own-create index, so a colliding stray id silently
// aliases instead; that sharper hazard is the NEXT test).
TEST(a_local_id_from_one_transaction_does_not_resolve_in_a_different_transaction) {
    Model m;
    Transaction txn1 = m.begin();
    auto a0 = std::make_unique<Account>();
    a0->name = "A0";
    txn1.create(std::move(a0));  // txn1 local index 0
    auto a1 = std::make_unique<Account>();
    a1->name = "A1";
    const Ref<Account> a_local =
        txn1.create(std::move(a1));  // txn1 local index 1 -- only meaningful inside txn1

    // A second, independent transaction never saw txn1's creates -- to txn2,
    // a_local is just some id it never issued and never cloned from base().
    Transaction txn2 = m.begin();
    CHECK(txn2.peek(a_local) == nullptr);
    CHECK(!txn2.exists(a_local));
    CHECK(txn2.update(a_local) == nullptr);

    // Embedding the stray local id in something txn2 commits fails loudly:
    // try_commit()'s remap table (built fresh per attempt) only knows about
    // local ids THIS transaction's own local_created_ produced -- and txn2
    // has only ONE create (its own index 0), so txn1's index 1 stays
    // unmapped.
    auto o = std::make_unique<Order>();
    o->code = "O1";
    o->account = a_local;
    txn2.create(std::move(o));

    const CommitResult res = m.try_commit(txn2);
    CHECK(res.status == CommitStatus::Invalid);
    CHECK(res.error.has_value());
    CHECK(!res.error->bad_target);                  // a stray local id has no real target to report
    CHECK_EQ(m.snapshot().size(), std::size_t{0});  // neither txn1 nor txn2 ever published anything
}

// The sharper, documented hazard behind the previous test: when two
// transactions' local counters land on the SAME raw index, the stray id
// silently aliases the wrong object instead of failing at all.
TEST(a_local_id_that_collides_with_another_transactions_own_local_index_silently_aliases) {
    Model m;

    Transaction txn1 = m.begin();
    auto a = std::make_unique<Account>();
    a->name = "FROM-TXN1";
    const Ref<Account> a_local = txn1.create(std::move(a));  // txn1's local counter: index 0

    Transaction txn2 = m.begin();
    auto own = std::make_unique<Account>();
    own->name = "TXN2-OWN";
    const Ref<Account> own_local =
        txn2.create(std::move(own));  // txn2's local counter ALSO starts at 0

    // Local ids are only unique WITHIN a transaction (kLocalIdBit | a per-txn
    // counter) -- across transactions the same raw Id can name two completely
    // different objects. Nothing here can tell them apart: no exception, no
    // null, just the wrong object.
    CHECK(a_local.raw() == own_local.raw());
    CHECK(txn2.peek(a_local) == txn2.peek(own_local));
    CHECK_EQ(txn2.peek(a_local)->name, std::string("TXN2-OWN"));  // NOT txn1's "FROM-TXN1"

    // update() is just as fooled -- this mutates txn2's own object, not txn1's.
    txn2.update(a_local)->balance = 42;
    CHECK_EQ(txn2.peek(own_local)->balance, std::int64_t{42});

    // And a commit built on this stray ref succeeds outright, because as far
    // as txn2's remap table is concerned, index 0 was always its own create.
    CommitResult res = m.try_commit(txn2);
    CHECK(res.status == CommitStatus::Committed);
    CHECK_EQ(m.snapshot().find(res.to_real(own_local))->name, std::string("TXN2-OWN"));
    CHECK_EQ(m.snapshot().size(), std::size_t{1});  // only txn2's object was ever real
}

// A pending update() is visible via txn.peek() immediately, but the
// committed Snapshot (and state_of()) stays untouched until try_commit()
// actually publishes.
TEST(transaction_update_is_visible_locally_without_touching_shared_state) {
    Model m;
    const Ref<Account> a = make_account(m, "A1", 10);
    const std::string before = state_of(m);

    Transaction txn = m.begin();
    txn.update(a)->balance = 999;
    CHECK_EQ(txn.peek(a)->balance, 999);          // visible locally
    CHECK_EQ(m.snapshot().find(a)->balance, 10);  // shared state untouched
    CHECK_EQ(state_of(m), before);

    commit_ok(m, txn);
    CHECK_EQ(m.snapshot().find(a)->balance, 999);
}

// Letting a Transaction go out of scope without committing is a
// complete, silent rollback: every pending create/update it held simply
// vanishes, and the model is byte-for-byte as if it never existed.
TEST(dropping_a_transaction_without_committing_touches_no_shared_state) {
    Model m;
    const Ref<Account> a = make_account(m, "A1", 5);
    const std::string before = state_of(m);

    {
        Transaction txn = m.begin();
        auto o1 = std::make_unique<Order>();
        o1->code = "O1";
        o1->account = a;
        txn.create(std::move(o1));
        auto a2 = std::make_unique<Account>();
        a2->name = "A2";
        txn.create(std::move(a2));
        txn.update(a)->balance = 12345;
        // txn destructs here, uncommitted.
    }

    CHECK_EQ(state_of(m), before);
    CHECK(m.snapshot().find_by_key<&Order::computed_key>("ord:O1") == nullptr);
    CHECK_EQ(m.snapshot().size(), std::size_t{1});
    CHECK_EQ(m.snapshot().find(a)->balance, 5);
}

// remove() masks its OWN target from the transaction's local view
// immediately, but the intent doesn't appear in pending_changes() (that
// list only carries creates/updates) until try_commit() resolves it.
TEST(remove_intent_is_not_yet_visible_as_deleted_in_pending_changes) {
    Model m;
    const Ref<Account> a = make_account(m, "A1");
    const Ref<Order> o = make_order(m, "O1", a);

    Transaction txn = m.begin();
    txn.remove(o);
    CHECK_EQ(txn.remove_intents().size(), std::size_t{1});
    CHECK(txn.remove_intents().count(o.raw()) == 1);
    for (const Change& c : txn.pending_changes()) CHECK(c.id != o.raw());
    // `o` itself is masked locally the instant remove() is called on it --
    // that's immediate, unlike cascade fan-out (a DIFFERENT object that
    // would die as a side effect of this remove, which stays visible
    // locally until try_commit() resolves the cascade).
    CHECK(!txn.exists(o));

    CommitResult res = m.try_commit(txn);
    CHECK(res.status == CommitStatus::Committed);
    CHECK(m.snapshot().find(o) == nullptr);
}

// With no pending remove() intents, estimate_changes_with_cascades() has nothing to
// estimate -- it's exactly pending_changes(), verbatim.
TEST(estimate_changes_with_cascades_with_no_pending_removes_matches_pending_changes_exactly) {
    Model m;
    const Ref<Account> a = make_account(m, "A1");
    Transaction txn = m.begin();
    auto o = std::make_unique<Order>();
    o->code = "O1";
    o->account = a;
    txn.create(std::move(o));
    txn.update(a)->balance = 99;

    const std::vector<Change> estimate = txn.estimate_changes_with_cascades();
    const std::vector<Change>& pending = txn.pending_changes();
    CHECK_EQ(estimate.size(), pending.size());
    for (std::size_t i = 0; i < estimate.size(); ++i) {
        CHECK(estimate[i].id == pending[i].id);
        CHECK(estimate[i].kind == pending[i].kind);
        CHECK(estimate[i].tag == pending[i].tag);
    }
}

// A pending remove() of an Account estimates the cascade into every Order
// whose (non-nullable) account field points at it -- Deleted for both,
// computed read-only against base(), matching what the real try_commit()
// (checked afterward) actually produces.
TEST(estimate_changes_with_cascades_estimates_a_non_nullable_cascade_delete) {
    Model m;
    const Ref<Account> a = make_account(m, "A1");
    const Ref<Order> o = make_order(m, "O1", a);

    Transaction txn = m.begin();
    txn.remove(a);
    const std::vector<Change> estimate = txn.estimate_changes_with_cascades();

    std::size_t deletes = 0;
    bool saw_account = false, saw_order = false;
    for (const Change& c : estimate) {
        if (c.kind != ChangeKind::Deleted) continue;
        ++deletes;
        if (c.id == a.raw()) saw_account = true;
        if (c.id == o.raw()) saw_order = true;
    }
    CHECK_EQ(deletes, std::size_t{2});
    CHECK(saw_account);
    CHECK(saw_order);

    // The estimate took no lock and mutated nothing -- the transaction
    // commits normally afterward, with a real changeset matching the
    // estimate's shape.
    const CommitResult res = m.try_commit(txn);
    CHECK(res.status == CommitStatus::Committed);
    CHECK_EQ(res.changes.size(), std::size_t{2});
}

// A pending remove() of an Order estimates an Updated (not a Deleted) for
// any OTHER Order whose Opt<Order> parent field points at it -- the
// nullable-cascade-null case, distinct from the non-nullable case above.
TEST(estimate_changes_with_cascades_estimates_a_nullable_cascade_null_as_updated_not_deleted) {
    Model m;
    const Ref<Account> a = make_account(m, "A1");
    const Ref<Order> parent = make_order(m, "P1", a);
    const Ref<Order> child = make_order(m, "C1", a, parent);

    Transaction txn = m.begin();
    txn.remove(parent);
    const std::vector<Change> estimate = txn.estimate_changes_with_cascades();

    bool saw_parent_deleted = false, saw_child_updated = false, saw_child_deleted = false;
    for (const Change& c : estimate) {
        if (c.id == parent.raw() && c.kind == ChangeKind::Deleted) saw_parent_deleted = true;
        if (c.id == child.raw() && c.kind == ChangeKind::Updated) saw_child_updated = true;
        if (c.id == child.raw() && c.kind == ChangeKind::Deleted) saw_child_deleted = true;
    }
    CHECK(saw_parent_deleted);
    CHECK(saw_child_updated);  // nullable ref -> estimated null-out, not a delete
    CHECK(!saw_child_deleted);
}

// TSan-targeted, same pattern as set_pre_commit_can_race_try_commit_
// without_a_data_race: estimate_changes_with_cascades() takes no lock at all (commit_mu_
// is unreachable from Transaction-building code -- invariant 10), so
// calling it repeatedly from one thread must never race with another
// thread hammering try_commit() concurrently. The real assertion here is
// TSan's (ctest --preset tsan).
TEST(estimate_changes_with_cascades_never_races_a_concurrently_hammering_try_commit) {
    Model m;
    const Ref<Account> a = make_account(m, "A1");
    for (int i = 0; i < 20; ++i) make_order(m, "O" + std::to_string(i), a);

    std::atomic<bool> stop{false};
    std::thread committer([&] {
        int n = 0;
        do {
            Transaction t = m.begin();
            auto o = std::make_unique<Order>();
            o->code = "W" + std::to_string(n++);
            o->account = a;
            t.create(std::move(o));
            (void)m.try_commit(t);
        } while (!stop.load(std::memory_order_relaxed));
    });

    Transaction txn = m.begin();
    txn.remove(a);
    for (int i = 0; i < 200; ++i) CHECK(!txn.estimate_changes_with_cascades().empty());

    stop = true;
    committer.join();
}

// When nothing else touches the model between building the estimate and
// actually committing, estimate_changes_with_cascades() and the changeset PreCommitFn
// sees are the SAME set of (id, kind, tag) triples. A pure remove() (no
// creates) keeps every id real on both sides, so this is an exact
// comparison, not just a size check.
TEST(
    estimate_changes_with_cascades_matches_the_pre_commit_hooks_changeset_when_nothing_else_happened) {
    Model m;
    const Ref<Account> a = make_account(m, "A1");
    const Ref<Order> o = make_order(m, "O1", a);
    (void)o;  // exists only so removing `a` has something to cascade into

    Transaction txn = m.begin();
    txn.remove(a);  // cascades to o
    const std::vector<Change> estimate = txn.estimate_changes_with_cascades();

    std::vector<Change> seen;
    m.set_pre_commit([&](Model&, const Transaction&, const std::vector<Change>& changes) {
        seen = changes;
        return true;
    });
    CHECK(m.try_commit(txn).status == CommitStatus::Committed);
    m.set_pre_commit({});

    auto by_id = [](const Change& x, const Change& y) { return x.id.index < y.id.index; };
    std::vector<Change> est_sorted = estimate;
    std::vector<Change> seen_sorted = seen;
    std::sort(est_sorted.begin(), est_sorted.end(), by_id);
    std::sort(seen_sorted.begin(), seen_sorted.end(), by_id);

    CHECK_EQ(est_sorted.size(), seen_sorted.size());
    for (std::size_t i = 0; i < est_sorted.size(); ++i) {
        CHECK(est_sorted[i].id == seen_sorted[i].id);
        CHECK(est_sorted[i].kind == seen_sorted[i].kind);
        CHECK(est_sorted[i].tag == seen_sorted[i].tag);
    }
}

// If a DIFFERENT transaction repoints a reference the estimate depended on
// before the original transaction actually commits, the estimate (computed
// against a now-stale base()) and the real changeset PreCommitFn sees
// diverge -- the estimate is a strict superset of what actually happened.
TEST(
    estimate_changes_with_cascades_and_the_pre_commit_hooks_changeset_diverge_after_a_reference_changes) {
    Model m;
    const Ref<Account> a = make_account(m, "A1");
    const Ref<Account> b = make_account(m, "B1");
    const Ref<Order> o = make_order(m, "O1", a);  // o.account == a, for now

    Transaction txn = m.begin();  // base() still sees o.account == a
    txn.remove(a);
    const std::vector<Change> estimate = txn.estimate_changes_with_cascades();

    // Estimated cascade: a itself, plus o (its non-nullable account field
    // still points at a, as far as txn's base() can see).
    std::size_t estimated_deletes = 0;
    for (const Change& c : estimate)
        if (c.kind == ChangeKind::Deleted) ++estimated_deletes;
    CHECK_EQ(estimated_deletes, std::size_t{2});

    // A different transaction repoints o away from a, in between -- txn's
    // own base() is pinned and never sees this happen.
    {
        Transaction other = m.begin();
        other.update(o)->account = b;
        CHECK(m.try_commit(other).status == CommitStatus::Committed);
    }

    // txn itself never conflicts (it only ever touches a's slot, never
    // o's), so it applies normally -- but the REAL cascade BFS runs against
    // the LATEST referrers_, where o no longer points at a. Only a dies.
    std::vector<Change> seen;
    m.set_pre_commit([&](Model&, const Transaction&, const std::vector<Change>& changes) {
        seen = changes;
        return true;
    });
    CHECK(m.try_commit(txn).status == CommitStatus::Committed);
    m.set_pre_commit({});

    CHECK_EQ(seen.size(), std::size_t{1});
    CHECK(seen[0].id == a.raw());
    CHECK(seen[0].kind == ChangeKind::Deleted);

    // The estimate (2 deletes, computed before the repoint) and reality (1
    // delete, after it) disagree -- exactly the staleness estimate_changes_with_cascades()'s
    // own doc comment warns about.
    CHECK(estimate.size() != seen.size());
}

// A same-transaction "create X referencing Y, then remove Y" correctly
// cascades the brand-new X too -- the cascade BFS sees the local create
// as a real referrer once apply installs it, not just pre-existing ones.
TEST(
    same_transaction_create_referencing_an_existing_object_then_remove_of_that_object_cascades_the_new_object_too) {
    Model m;
    const Ref<Account> a = make_account(m, "A1");

    Transaction txn = m.begin();
    auto o = std::make_unique<Order>();
    o->code = "O1";
    o->account = a;  // new Order -> existing Account
    const Ref<Order> o_local = txn.create(std::move(o));
    txn.remove(a);  // and, same txn, delete that Account
    CommitResult res = m.try_commit(txn);

    CHECK(res.status == CommitStatus::Committed);
    std::size_t killed = 0;
    for (const Change& c : res.changes)
        if (c.kind == ChangeKind::Deleted) ++killed;
    CHECK_EQ(killed, std::size_t{2});  // a AND the brand-new order
    CHECK(m.snapshot().find(a) == nullptr);
    CHECK(m.snapshot().find(res.to_real(o_local)) == nullptr);
}

// remove()'ing an id that was create()'d earlier in the SAME transaction
// cancels the create outright -- there is nothing to publish, and the
// commit is a real no-op, not a create-then-immediate-delete pair.
TEST(removing_a_locally_created_object_cancels_the_create_outright) {
    Model m;
    Transaction txn = m.begin();
    auto a = std::make_unique<Account>();
    a->name = "TEMP";
    const Ref<Account> a_local = txn.create(std::move(a));
    txn.remove(a_local);  // never committed -- nothing for anything else to reference
    CHECK(!txn.exists(a_local));

    CommitResult res = m.try_commit(txn);
    CHECK(res.status == CommitStatus::Committed);
    CHECK(res.changes.empty());
    CHECK_EQ(m.snapshot().size(), std::size_t{0});
}


// Two Transactions from the same Model never collide, and id() survives the
// move that returns a Transaction out of begin() (Transaction is move-only;
// see its own comment on why).
TEST(transaction_id_is_unique_per_transaction_and_stable_across_a_move) {
    Model m;
    Transaction a = m.begin();
    Transaction b = m.begin();
    CHECK(a.id() != b.id());

    const std::uint64_t id_before_move = a.id();
    Transaction c = std::move(a);
    CHECK_EQ(c.id(), id_before_move);
}

// The defining proof of the correlation mechanism: PreTransactionsFn,
// PreCommitFn, and PostCommitFn all receive a `const Transaction&` for one
// try_commit() attempt, and it is the SAME Transaction throughout -- all
// three see the exact id() the caller's own `txn.id()` has, before
// try_commit() is even called. A pre-transaction built INSIDE
// PreTransactionsFn is a genuinely different Transaction and gets its own,
// different id -- proving the mechanism distinguishes "the main attempt"
// from "a pre-transaction that ran as part of it" without any extra API.
TEST(pre_transactions_pre_commit_and_post_commit_all_see_the_same_transaction_id) {
    Model m;
    std::uint64_t seen_in_pre_transactions = 0;
    std::uint64_t seen_in_pre_commit = 0;
    std::uint64_t seen_in_post_commit = 0;
    std::uint64_t pre_transaction_own_id = 0;

    m.set_pre_transactions([&](Model& model, const Transaction& main_txn) {
        seen_in_pre_transactions = main_txn.id();

        Transaction pre = model.begin();
        pre_transaction_own_id = pre.id();  // a DIFFERENT id from main_txn's
        auto a = std::make_unique<Account>();
        a->name = "PRE";
        pre.create(std::move(a));
        CHECK(model.run_pre_transaction(pre).status == CommitStatus::Committed);
    });
    m.set_pre_commit([&](Model&, const Transaction& main_txn, const std::vector<Change>&) {
        seen_in_pre_commit = main_txn.id();
        return true;
    });
    m.set_post_commit([&](Model&, const Transaction& main_txn, const CommitResult&) {
        seen_in_post_commit = main_txn.id();
    });

    Transaction txn = m.begin();
    auto a = std::make_unique<Account>();
    a->name = "MAIN";
    txn.create(std::move(a));
    const std::uint64_t txn_id = txn.id();  // read BEFORE try_commit() -- proves the caller can
                                            // already know the correlation key in advance
    CHECK(m.try_commit(txn).status == CommitStatus::Committed);

    CHECK_EQ(seen_in_pre_transactions, txn_id);
    CHECK_EQ(seen_in_pre_commit, txn_id);
    CHECK_EQ(seen_in_post_commit, txn_id);
    CHECK(pre_transaction_own_id != txn_id);  // the pre-transaction is a DIFFERENT attempt

    m.set_pre_transactions({});
    m.set_pre_commit({});
    m.set_post_commit({});
}

