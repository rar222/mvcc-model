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


// Deleting a target cascades to kill every non-nullable (Ref<>) referrer
// too, and removes them from every index (by_type, define_keys) as well
// -- not just the object store.
TEST(cascade_kills_non_nullable_referrers) {
    Model m;
    const Ref<Account> a = make_account(m, "A1");
    const Ref<Order> o1 = make_order(m, "O1", a);
    const Ref<Order> o2 = make_order(m, "O2", a);

    const std::size_t killed = remove_and_commit(m, a);  // both orders hold a Ref<Account>

    CHECK_EQ(killed, std::size_t{3});
    Snapshot s = m.snapshot();
    CHECK(s.find(a) == nullptr);
    CHECK(s.find(o1) == nullptr);
    CHECK(s.find(o2) == nullptr);
    CHECK(s.find_by_key<&Order::computed_key>("ord:O1") == nullptr);
    CHECK_EQ(s.size(), std::size_t{0});
}

// A referrer reachable only via a nullable Opt<> field survives its
// target's deletion with that field nulled, while its OTHER (Ref<>)
// edges stay completely intact.
TEST(cascade_nulls_nullable_referrers_instead_of_killing_them) {
    Model m;
    const Ref<Account> a = make_account(m, "A1");
    const Ref<Order> p = make_order(m, "P", a);
    const Ref<Order> c = make_order(m, "C", a, p);  // c->parent is an Opt<Order> to p

    const std::size_t killed = remove_and_commit(m, p);

    CHECK_EQ(killed, std::size_t{1});  // only p
    Snapshot s = m.snapshot();
    CHECK(s.find(p) == nullptr);
    CHECK(s.find(c) != nullptr);  // c survives
    CHECK(!s.find(c)->parent);    // with its Opt<> nulled
    CHECK(s.resolve(s.find(c)->parent) == nullptr);
    CHECK_EQ(s.resolve(s.find(c)->account).id, a.raw());  // its Ref<> is intact
}

// A chain of non-nullable Ref<> edges (o3 -> o2 -> o1 -> a) all die
// together when the chain's root is removed -- cascade isn't limited to
// one hop.
TEST(cascade_is_transitive_and_mixed) {
    Model m;
    const Ref<Account> a = make_account(m, "A1");
    const Ref<Order> o1 = make_order(m, "O1", a);
    const Ref<Order> o2 = make_order(m, "O2", a, o1);
    const Ref<Order> o3 = make_order(m, "O3", a, o2);

    const std::size_t killed = remove_and_commit(m, a);

    CHECK_EQ(killed, std::size_t{4});
    Snapshot s = m.snapshot();
    CHECK_EQ(s.size(), std::size_t{0});
    (void)o3;
}

// The cascade BFS's visited-set guards against a Ref<>/Opt<> cycle
// looping forever: a cycle resolves by nulling the Opt<> edge in it,
// not by hanging try_commit().
TEST(cascade_terminates_on_cycles) {
    Model m;
    const Ref<Account> a = make_account(m, "A1");
    const Ref<Order> o1 = make_order(m, "O1", a);
    const Ref<Order> o2 = make_order(m, "O2", a, o1);

    update_field(m, o1, [&](Order* p) { p->parent = o2; });  // o1 <-> o2 cycle via Opt<>

    const std::size_t killed = remove_and_commit(m, o1);  // must not spin forever

    CHECK_EQ(killed, std::size_t{1});
    Snapshot s = m.snapshot();
    CHECK(s.find(o1) == nullptr);
    CHECK(s.find(o2) != nullptr);
    CHECK(!s.find(o2)->parent);  // cycle edge nulled
}

// After Transaction::update() repoints a non-nullable Ref<> to a new
// target, the reverse index tracks the NEW target (not the original) --
// deleting the new target correctly cascades, proving reconciliation ran.
TEST(reassigning_a_ref_field_after_update_is_tracked_for_cascade) {
    Model m;
    const Ref<Account> a1 = make_account(m, "A1");
    const Ref<Account> a2 = make_account(m, "A2");
    const Ref<Order> o = make_order(m, "O1", a1);

    update_field(m, o, [&](Order* p) { p->account = a2; });  // reassign the non-nullable Ref
    CHECK(m.snapshot().find(o)->account == a2);

    const std::size_t killed =
        remove_and_commit(m, a2);  // must cascade-kill o: Ref<> is non-nullable

    CHECK_EQ(killed, std::size_t{2});  // a2 and o
    CHECK(m.snapshot().find(o) == nullptr);
}

// Same idea for a nullable Opt<>: reassigning it moves the reverse-index
// edge to the new target, and the OLD target is no longer treated as a
// referrer -- no phantom edge left behind to corrupt a later cascade.
TEST(reassigning_an_opt_field_after_update_is_tracked_for_cascade) {
    Model m;
    const Ref<Account> a = make_account(m, "A1");
    const Ref<Order> p1 = make_order(m, "P1", a);
    const Ref<Order> p2 = make_order(m, "P2", a);
    const Ref<Order> c = make_order(m, "C", a, p1);  // parent = p1 initially

    update_field(m, c, [&](Order* o) { o->parent = p2; });  // reassign the nullable Opt<>
    CHECK(m.snapshot().find(c)->parent == Opt<Order>(p2));

    remove_and_commit(m, p2);  // must null c's parent: the index must track p2, not p1
    CHECK(m.snapshot().find(c) != nullptr);  // Opt<>: survives
    CHECK(!m.snapshot().find(c)->parent);    // ...nulled

    // p1 must be untouched by any of this -- proves the OLD edge (c -> p1)
    // was correctly dropped rather than left as a phantom referrer.
    remove_and_commit(m, p1);
    CHECK(m.snapshot().find(c) != nullptr);
}

// Regression guard: see the in-body comment. Two update() calls to the
// same id in ONE transaction must reconcile against the last COMMITTED
// state, never an intermediate, never-published in-transaction value.
TEST(repeated_updates_in_one_transaction_diff_against_the_pre_transaction_state) {
    // Two update() calls on the same id before a single commit must still
    // reconcile against the state as of the LAST commit, not an intermediate,
    // never-published clone from partway through this transaction.
    Model m;
    const Ref<Account> a1 = make_account(m, "A1");
    const Ref<Account> a2 = make_account(m, "A2");
    const Ref<Account> a3 = make_account(m, "A3");
    const Ref<Order> o = make_order(m, "O1", a1);

    {
        Transaction txn = m.begin();
        txn.update(o)->account = a2;  // first update this transaction
        txn.update(o)->account = a3;  // second update, same transaction, same id
        commit_ok(m, txn);
    }
    CHECK(m.snapshot().find(o)->account == a3);

    CHECK_EQ(remove_and_commit(m, a1), std::size_t{1});  // just a1; o must be unaffected
    CHECK(m.snapshot().find(o) != nullptr);

    CHECK_EQ(remove_and_commit(m, a2), std::size_t{1});  // just a2; o must be unaffected
    CHECK(m.snapshot().find(o) != nullptr);

    CHECK_EQ(remove_and_commit(m, a3), std::size_t{2});  // a3 and o
    CHECK(m.snapshot().find(o) == nullptr);
}

// Transaction::update() on an id already deleted in the committed state
// returns null rather than resurrecting or aliasing it.
TEST(update_returns_null_for_a_dead_id) {
    Model m;
    const Ref<Account> a = make_account(m, "A1");
    const Ref<Order> o = make_order(m, "O1", a);
    remove_and_commit(m, o);

    Transaction txn = m.begin();
    CHECK(txn.update(o) == nullptr);
    CHECK(!txn.exists(o));
}

// Try to break invariant 1 (Ref<> never dangles) by racing a create against
// a remove of its OWN target inside one transaction: apply_transaction_
// contents() installs creates (and their referrers_ edges) BEFORE it
// resolves remove_intents_ (see that function's own ordering comment), so
// the cascade BFS that removes `a` sees the brand-new Order as a referrer
// too, even though the Order never existed before this very commit. If the
// ordering were reversed, this would publish a dangling Ref<Account> that
// no reader could ever have seen coming.
TEST(create_referencing_a_target_removed_in_the_same_transaction_cascades_it_too) {
    Model m;
    const Ref<Account> a = make_account(m, "A1");

    Transaction txn = m.begin();
    auto o = std::make_unique<Order>();
    o->code = "NEW";
    o->account = a;  // brand-new object, non-nullable Ref<> to a
    const Ref<Order> local = txn.create(std::move(o));
    txn.remove(a);  // same transaction ALSO removes the very thing it just pointed at

    const CommitResult res = m.try_commit(txn);
    CHECK(res.status == CommitStatus::Committed);
    std::size_t killed = 0;
    for (const Change& c : res.changes)
        if (c.kind == ChangeKind::Deleted) ++killed;
    CHECK_EQ(killed, std::size_t{2});  // a, and the brand-new order that referenced it

    const Ref<Order> real = res.to_real(local);
    Snapshot s = m.snapshot();
    CHECK(s.find(a) == nullptr);
    CHECK(s.find(real) == nullptr);  // never visible to any reader -- born and cascaded in one commit
    CHECK_EQ(s.size(), std::size_t{0});
}

// Same break attempt, aimed at an EXISTING committed object instead of a
// brand-new one: apply_update() reconciles referrers_ against the update's
// FINAL value (invariant 9) before the remove_intents_ loop runs, so the
// cascade BFS sees the repointed edge -- not the stale one -- and kills the
// updated object too, via the target it was reassigned to THIS transaction.
TEST(update_repointing_to_a_target_removed_in_the_same_transaction_cascades_the_repointed_object_too) {
    Model m;
    const Ref<Account> a1 = make_account(m, "A1");
    const Ref<Account> a2 = make_account(m, "A2");
    const Ref<Order> o = make_order(m, "O1", a1);

    Transaction txn = m.begin();
    txn.update(o)->account = a2;  // repoint o's non-nullable Ref<> to a2
    txn.remove(a2);               // same transaction removes a2

    const CommitResult res = m.try_commit(txn);
    CHECK(res.status == CommitStatus::Committed);
    std::size_t killed = 0;
    for (const Change& c : res.changes)
        if (c.kind == ChangeKind::Deleted) ++killed;
    CHECK_EQ(killed, std::size_t{2});  // a2, and o (now pointing at it)

    Snapshot s = m.snapshot();
    CHECK(s.find(a2) == nullptr);
    CHECK(s.find(o) == nullptr);   // cascaded via the NEW edge, not the stale a1 one
    CHECK(s.find(a1) != nullptr);  // untouched -- o no longer points at it by the time of the cascade
}


TEST(a_create_may_reference_itself_within_its_own_transaction) {
    Model m;
    Transaction txn = m.begin();
    const Ref<Link> l = txn.create(std::make_unique<Link>());
    txn.update(l)->label = "SELF";
    txn.update(l)->next = l;  // non-nullable self-loop, local id

    CommitResult res = m.try_commit(txn);
    CHECK(res.status == CommitStatus::Committed);
    const Ref<Link> real = res.to_real(l);
    Snapshot s = m.snapshot();
    CHECK(s.find(real) != nullptr);
    CHECK(s.find(real)->next.raw() == real.raw());  // resolved to itself, not mismapped
}

TEST(a_create_may_reference_a_later_create_in_the_same_transaction) {
    Model m;
    Transaction txn = m.begin();
    // The Order is created FIRST, its non-nullable account filled in with a
    // local id minted SECOND -- rejected before the pre-mint pass existed,
    // now equivalent to the backward form.
    const Ref<Order> o = txn.create(std::make_unique<Order>());
    txn.update(o)->code = "FWD";
    auto acc = std::make_unique<Account>();
    acc->name = "LATER";
    const Ref<Account> a = txn.create(std::move(acc));
    txn.update(o)->account = a;

    CommitResult res = m.try_commit(txn);
    CHECK(res.status == CommitStatus::Committed);
    Snapshot s = m.snapshot();
    const Order* op = s.find(res.to_real(o));
    CHECK(op != nullptr);
    CHECK_EQ(s.resolve(op->account).name, std::string("LATER"));
}

TEST(two_creates_forming_a_non_nullable_cycle_commit_and_cascade_as_one) {
    Model m;
    Transaction txn = m.begin();
    const Ref<Link> l1 = txn.create(std::make_unique<Link>());
    const Ref<Link> l2 = txn.create(std::make_unique<Link>());
    txn.update(l1)->label = "L1";
    txn.update(l1)->next = l2;
    txn.update(l2)->label = "L2";
    txn.update(l2)->next = l1;  // l1 <-> l2, both edges non-nullable

    CommitResult res = m.try_commit(txn);
    CHECK(res.status == CommitStatus::Committed);
    const Ref<Link> r1 = res.to_real(l1);
    const Ref<Link> r2 = res.to_real(l2);
    {
        Snapshot s = m.snapshot();
        CHECK(s.find(r1)->next.raw() == r2.raw());
        CHECK(s.find(r2)->next.raw() == r1.raw());
    }

    // A non-nullable cycle lives and dies as a unit: removing either link
    // cascades to the other (and the BFS's visited set terminates the loop).
    const std::size_t killed = remove_and_commit(m, r1);
    CHECK_EQ(killed, std::size_t{2});
    Snapshot s = m.snapshot();
    CHECK(s.find(r1) == nullptr);
    CHECK(s.find(r2) == nullptr);
}

// remove() of a local create that another pending object references DEFERS
// instead of cancelling (see Transaction::remove_impl): the create installs
// at apply time and is then removed by the same cascade BFS a committed id
// gets -- so an Opt<> referrer survives with the field nulled, exactly as
// if the victim had been committed first.
TEST(removing_a_referenced_local_create_cascades_at_commit_like_a_committed_remove) {
    Model m;
    const Ref<Account> acct = make_account(m, "A1");

    Transaction txn = m.begin();
    const Ref<Order> victim = txn.create(std::make_unique<Order>());
    txn.update(victim)->code = "VICTIM";
    txn.update(victim)->account = acct;
    auto o = std::make_unique<Order>();
    o->code = "SURVIVOR";
    o->account = acct;
    o->parent = victim;  // Opt<>: nulled by the cascade, not killed
    const Ref<Order> survivor = txn.create(std::move(o));
    txn.remove(victim);
    CHECK(!txn.exists(victim));            // masked immediately, like a real remove intent
    CHECK(txn.update(victim) == nullptr);  // and no longer writable

    const CommitResult res = m.try_commit(txn);
    CHECK(res.status == CommitStatus::Committed);
    const Ref<Order> real_victim = res.to_real(victim);
    const Ref<Order> real_survivor = res.to_real(survivor);

    // The victim was genuinely created then removed within the one commit:
    // both events appear in the changeset, and the published state has no
    // trace of it.
    std::size_t victim_created = 0, victim_deleted = 0;
    for (const Change& c : res.changes) {
        if (c.id == real_victim.raw() && c.kind == ChangeKind::Created) ++victim_created;
        if (c.id == real_victim.raw() && c.kind == ChangeKind::Deleted) ++victim_deleted;
    }
    CHECK_EQ(victim_created, std::size_t{1});
    CHECK_EQ(victim_deleted, std::size_t{1});

    Snapshot s = m.snapshot();
    CHECK(s.find(real_victim) == nullptr);
    const Order* sp = s.find(real_survivor);
    CHECK(sp != nullptr);
    CHECK(!sp->parent);  // nulled by the cascade, same as a committed-victim remove
}

// The Ref<> (non-nullable) flavor, with a transitive hop: victim <- A <- B,
// all three pending in the same transaction. Removing the victim drags the
// whole chain down, exactly as if all three had been committed first.
TEST(removing_a_referenced_local_create_cascades_transitively_through_pending_creates) {
    Model m;
    Transaction txn = m.begin();
    const Ref<Link> victim = txn.create(std::make_unique<Link>());
    txn.update(victim)->label = "VICTIM";
    txn.update(victim)->next = victim;  // self-loop: satisfies its own non-nullable field
    const Ref<Link> a = txn.create(std::make_unique<Link>());
    txn.update(a)->label = "A";
    txn.update(a)->next = victim;
    const Ref<Link> b = txn.create(std::make_unique<Link>());
    txn.update(b)->label = "B";
    txn.update(b)->next = a;
    txn.remove(victim);

    const CommitResult res = m.try_commit(txn);
    CHECK(res.status == CommitStatus::Committed);
    Snapshot s = m.snapshot();
    CHECK(s.find(res.to_real(victim)) == nullptr);
    CHECK(s.find(res.to_real(a)) == nullptr);  // Ref<> referrer: died with the victim
    CHECK(s.find(res.to_real(b)) == nullptr);  // and transitively, its own referrer
    CHECK_EQ(s.size(), std::size_t{0});
}

// The referenced-or-not decision is taken AT the remove() call: a ref
// written toward a local id AFTER its create was already cancelled is a
// build bug, and still rejects as Invalid (the unmapped-local path).
TEST(a_ref_added_after_a_local_create_was_cancelled_is_rejected_as_invalid) {
    Model m;
    const Ref<Account> acct = make_account(m, "A1");

    Transaction txn = m.begin();
    const Ref<Order> victim = txn.create(std::make_unique<Order>());
    txn.remove(victim);  // unreferenced at this point -> cancelled outright
    auto o = std::make_unique<Order>();
    o->code = "DANGLER";
    o->account = acct;
    o->parent = victim;  // written AFTER the cancel: nothing will ever map it
    txn.create(std::move(o));

    const CommitResult res = m.try_commit(txn);
    CHECK(res.status == CommitStatus::Invalid);
    CHECK(res.error.has_value());
    CHECK(!res.error->bad_target);                  // a cancelled create has no real id to report
    CHECK_EQ(m.snapshot().size(), std::size_t{1});  // only the pre-existing account
}

