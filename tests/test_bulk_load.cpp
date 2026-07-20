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

//
// None of these tests hold a Snapshot or Transaction across a commit_bulk_without_undo()
// call -- that's the exclusive-access precondition documented on
// Model::commit_bulk_without_undo() itself, and violating it is undefined behavior, not a
// checked error (see the assert in the implementation, which is a tripwire
// for misuse during development, not a guard these tests should lean on).

TEST(bulk_load_into_a_fresh_model_installs_everything_with_cross_object_local_refs) {
    Model m;
    BulkTransaction t = m.begin_bulk();
    auto acc = std::make_unique<Account>();
    acc->name = "A1";
    const Ref<Account> a = t.create(std::move(acc));
    auto ord = std::make_unique<Order>();
    ord->code = "O1";
    ord->account = a;  // forward reference to another object in the SAME batch
    const Ref<Order> o = t.create(std::move(ord));

    CommitResult r = m.commit_bulk_without_undo(t);
    CHECK(r.status == CommitStatus::Committed);
    const Ref<Account> real_a = r.to_real(a);
    const Ref<Order> real_o = r.to_real(o);

    Snapshot s = m.snapshot();
    CHECK(s.find(real_a) != nullptr);
    CHECK(s.find(real_o) != nullptr);
    CHECK_EQ(s.find(real_o)->account.raw(), real_a.raw());
    CHECK(s.find_by_key<&Account::name>("A1") != nullptr);
}

// update() lets a forward reference (to an object this batch hasn't created
// yet) get fixed up after the fact, instead of predicting a future local id
// by hand the way bulk_load_rejects_an_out_of_range_local_ref_with_nothing_
// mutated below constructs one manually.
TEST(bulk_transaction_update_fixes_up_a_forward_reference_after_the_fact) {
    Model m;
    BulkTransaction t = m.begin_bulk();
    auto ord = std::make_unique<Order>();
    ord->code = "O1";  // account left null -- the Account doesn't exist yet
    const Ref<Order> o = t.create(std::move(ord));

    auto acc = std::make_unique<Account>();
    acc->name = "A1";
    const Ref<Account> a = t.create(std::move(acc));

    t.update(o)->account = a;  // now that `a` exists, fix up the forward ref

    CommitResult r = m.commit_bulk_without_undo(t);
    CHECK(r.status == CommitStatus::Committed);
    const Ref<Account> real_a = r.to_real(a);
    const Ref<Order> real_o = r.to_real(o);

    Snapshot s = m.snapshot();
    CHECK_EQ(s.find(real_o)->account.raw(), real_a.raw());
}

TEST(bulk_transaction_update_returns_null_for_an_id_this_batch_never_created) {
    Model m;
    BulkTransaction t = m.begin_bulk();
    auto acc = std::make_unique<Account>();
    acc->name = "A1";
    const Ref<Account> a = t.create(std::move(acc));

    CHECK(t.update(Ref<Account>(Id{kLocalIdBit | 999u, 1})) == nullptr);  // out of range
    CHECK(t.update(a) != nullptr);  // sanity: a valid local id still resolves
}

TEST(bulk_load_wipes_all_pre_existing_data) {
    Model m;
    make_account(m, "old1");
    make_account(m, "old2");
    CHECK_EQ(m.snapshot().find_by_key<&Account::name>("old1") != nullptr, true);

    BulkTransaction t = m.begin_bulk();
    auto acc = std::make_unique<Account>();
    acc->name = "new1";
    t.create(std::move(acc));
    CommitResult r = m.commit_bulk_without_undo(t);
    CHECK(r.status == CommitStatus::Committed);

    Snapshot s = m.snapshot();
    CHECK(s.find_by_key<&Account::name>("old1") == nullptr);
    CHECK(s.find_by_key<&Account::name>("old2") == nullptr);
    CHECK(s.find_by_key<&Account::name>("new1") != nullptr);
    std::size_t count = 0;
    s.for_each<Account>([&](const Account&) { ++count; });
    CHECK_EQ(count, std::size_t{1});
}

TEST(bulk_load_rejects_an_out_of_range_local_ref_with_nothing_mutated) {
    Model m;
    make_account(m, "survivor");
    const std::uint64_t before = m.current_version();

    BulkTransaction t = m.begin_bulk();
    auto ord = std::make_unique<Order>();
    ord->code = "O1";
    // Points past the end of this batch (only one object is being created)
    // -- after the wipe, nothing else could ever satisfy it.
    ord->account = Ref<Account>(Id{kLocalIdBit | 999u, 1});
    t.create(std::move(ord));

    CommitResult r = m.commit_bulk_without_undo(t);
    CHECK(r.status == CommitStatus::Invalid);
    CHECK_EQ(m.current_version(), before);  // no wipe, no publish -- rejected before any mutation
    CHECK(m.snapshot().find_by_key<&Account::name>("survivor") != nullptr);
}

TEST(bulk_load_rejects_a_null_non_nullable_ref) {
    Model m;
    const std::uint64_t before = m.current_version();

    BulkTransaction t = m.begin_bulk();
    auto ord = std::make_unique<Order>();
    ord->code = "O1";  // ord->account (Ref<Account>, non-nullable) left null
    t.create(std::move(ord));

    CommitResult r = m.commit_bulk_without_undo(t);
    CHECK(r.status == CommitStatus::Invalid);
    CHECK_EQ(m.current_version(), before);
}

TEST(bulk_load_rejects_a_reference_to_a_pre_wipe_real_id) {
    Model m;
    const Ref<Account> old_real = make_account(m, "pre_wipe");

    BulkTransaction t = m.begin_bulk();
    auto ord = std::make_unique<Order>();
    ord->code = "O1";
    ord->account = old_real;  // a REAL id, not local -- can't survive the wipe
    t.create(std::move(ord));

    CommitResult r = m.commit_bulk_without_undo(t);
    CHECK(r.status == CommitStatus::Invalid);
}

TEST(normal_transaction_and_find_by_key_work_correctly_after_a_bulk_load) {
    Model m;
    {
        BulkTransaction t = m.begin_bulk();
        auto acc = std::make_unique<Account>();
        acc->name = "C1";
        t.create(std::move(acc));
        CHECK(m.commit_bulk_without_undo(t).status == CommitStatus::Committed);
    }

    const Ref<Account> a2 =
        make_account(m, "C2");  // an ordinary Transaction, on top of the bulk load
    update_field(m, a2, [](Account* a) { a->balance = 42; });

    Snapshot s = m.snapshot();
    CHECK(s.find_by_key<&Account::name>("C1") != nullptr);
    CHECK(s.find_by_key<&Account::name>("C2") != nullptr);
    CHECK_EQ(s.find(a2)->balance, 42);
    std::size_t count = 0;
    s.for_each<Account>([&](const Account&) { ++count; });
    CHECK_EQ(count, std::size_t{2});
}

// commit_bulk_without_undo() mints its whole remap table before installing anything, so
// a batch may contain forward references, self-loops, and cycles -- the
// same order-independence the ordinary Transaction path gets from its
// pre-mint pass (see apply_transaction_contents). BulkTransaction has no
// update(), so a forward or self edge is written using the local id the
// target WILL get -- deterministic by contract (kLocalIdBit | position in
// creation order; see BulkTransaction::create's doc comment).
TEST(bulk_load_accepts_forward_references_self_loops_and_cycles) {
    Model m;
    auto local = [](std::uint32_t position) { return Ref<Link>(Id{kLocalIdBit | position, 1}); };

    BulkTransaction t = m.begin_bulk();
    auto self = std::make_unique<Link>();
    self->label = "SELF";
    self->next = local(0);  // position 0: itself -- non-nullable self-loop
    const Ref<Link> sl = t.create(std::move(self));
    CHECK(sl.raw() == local(0).raw());  // the predicted id is the minted one

    auto c1 = std::make_unique<Link>();
    c1->label = "C1";
    c1->next = local(2);  // forward: position 2 doesn't exist yet
    const Ref<Link> l1 = t.create(std::move(c1));
    auto c2 = std::make_unique<Link>();
    c2->label = "C2";
    c2->next = local(1);  // backward: closes the non-nullable 2-cycle
    const Ref<Link> l2 = t.create(std::move(c2));

    CommitResult r = m.commit_bulk_without_undo(t);
    CHECK(r.status == CommitStatus::Committed);
    const Ref<Link> rs = r.to_real(sl);
    const Ref<Link> r1 = r.to_real(l1);
    const Ref<Link> r2 = r.to_real(l2);
    {
        Snapshot s = m.snapshot();
        CHECK(s.find(rs)->next.raw() == rs.raw());  // self-loop resolved to itself
        CHECK(s.find(r1)->next.raw() == r2.raw());  // cycle edges resolved crosswise
        CHECK(s.find(r2)->next.raw() == r1.raw());
    }

    // The bulk-built non-nullable cycle lives and dies as a unit, exactly
    // like the Transaction-built one -- referrers_ came out of the no-log
    // install path in the state the cascade BFS expects.
    const std::size_t killed = remove_and_commit(m, r1);
    CHECK_EQ(killed, std::size_t{2});
    Snapshot s = m.snapshot();
    CHECK(s.find(r1) == nullptr);
    CHECK(s.find(r2) == nullptr);
    CHECK(s.find(rs) != nullptr);  // the self-loop was never part of that cycle
}

// The reverse index (referrers_) has to come out of commit_bulk_without_undo()'s no-log
// installation path in exactly the state cascade delete expects -- this
// exercises that by cascading a non-nullable ref and nulling a nullable one,
// entirely on bulk-loaded data.
TEST(cascade_delete_works_correctly_on_bulk_loaded_data) {
    Model m;
    BulkTransaction t = m.begin_bulk();
    auto acc = std::make_unique<Account>();
    acc->name = "Hub";
    const Ref<Account> a = t.create(std::move(acc));
    auto other = std::make_unique<Account>();           // unrelated -- keeps grandchild's own
    other->name = "Other";                              // non-nullable Ref<Account> satisfied after
    const Ref<Account> b = t.create(std::move(other));  // `a` is removed, so only `parent` nulls
    auto child = std::make_unique<Order>();
    child->code = "child";
    child->account = a;  // non-nullable: dies when `a` is removed
    const Ref<Order> c = t.create(std::move(child));
    auto grandchild = std::make_unique<Order>();
    grandchild->code = "grandchild";
    grandchild->account = b;
    grandchild->parent = c;  // nullable: survives, but parent gets nulled
    const Ref<Order> g = t.create(std::move(grandchild));

    CommitResult r = m.commit_bulk_without_undo(t);
    CHECK(r.status == CommitStatus::Committed);
    const Ref<Account> real_a = r.to_real(a);
    const Ref<Order> real_c = r.to_real(c);
    const Ref<Order> real_g = r.to_real(g);

    const std::size_t killed = remove_and_commit(m, real_a);
    CHECK_EQ(killed, std::size_t{2});  // Hub + child cascade; grandchild survives

    Snapshot s = m.snapshot();
    CHECK(s.find(real_c) == nullptr);
    CHECK(s.find(real_g) != nullptr);
    CHECK(s.find(real_g)->parent.raw() == Id{});  // nulled, not cascaded
}

