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
    CHECK_EQ(s.find(real_o)->account.id(), real_a.id());
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
    CHECK_EQ(s.find(real_o)->account.id(), real_a.id());
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

// BulkTransaction::size() is a direct reflection of objects_.size() (see
// create()'s own comment: "no remove() to cancel an entry and leave a
// hole"), so it should track every create() one-for-one with no batching or
// lag, the same running-count property the sibling test above sets up but
// never itself asserts on.
TEST(bulk_transaction_size_reports_pending_object_count) {
    Model m;
    BulkTransaction t = m.begin_bulk();
    CHECK_EQ(t.size(), std::size_t{0});

    auto a1 = std::make_unique<Account>();
    a1->name = "A1";
    const Ref<Account> a = t.create(std::move(a1));
    CHECK_EQ(t.size(), std::size_t{1});

    auto a2 = std::make_unique<Account>();
    a2->name = "A2";
    t.create(std::move(a2));
    CHECK_EQ(t.size(), std::size_t{2});

    auto ord = std::make_unique<Order>();
    ord->code = "O1";
    ord->account = a;
    t.create(std::move(ord));
    CHECK_EQ(t.size(), std::size_t{3});
}

// The BulkTransaction analog of
// a_local_id_from_one_transaction_does_not_resolve_in_a_different_transaction
// (test_transactions.cpp): BulkTransaction::update_raw() is a plain bounds
// check against THIS batch's own objects_ (see its doc comment -- no
// per-Model tagging at all), so a local id minted by a DIFFERENT, smaller
// BulkTransaction is simply out of range rather than aliasing into this
// batch's own local table.
TEST(bulk_transaction_update_on_an_id_from_a_different_bulk_transaction_returns_null) {
    Model m;
    BulkTransaction t1 = m.begin_bulk();
    auto a0 = std::make_unique<Account>();
    a0->name = "A0";
    t1.create(std::move(a0));  // t1 local index 0
    auto a1 = std::make_unique<Account>();
    a1->name = "A1";
    const Ref<Account> a1_local = t1.create(std::move(a1));  // t1 local index 1

    BulkTransaction t2 = m.begin_bulk();
    auto own = std::make_unique<Account>();
    own->name = "OWN";
    const Ref<Account> own_local = t2.create(std::move(own));  // t2's only object: local index 0

    // t1's index-1 id is out of range for t2 (whose objects_ has size 1) --
    // never touches t2's own object at index 0.
    CHECK(t2.update(a1_local) == nullptr);
    CHECK(t2.update(own_local) != nullptr);  // sanity: t2's own local id still resolves
}

// commit_bulk_without_undo()'s doc comment: "if any object's Ref<>/Opt<>
// fails to resolve WITHIN this batch ... rejects as CommitStatus::Invalid
// with NOTHING touched: the whole batch is validated before any mutation
// begins." This exercises that with a MIXED batch -- one object whose
// forward ref gets legitimately fixed up via update() (so a naive
// object-at-a-time validator might be fooled into accepting it) alongside a
// second, unrelated object left with a genuinely dangling ref -- and checks
// that the fixed-up object doesn't get partially installed either.
TEST(bulk_load_rejects_a_batch_where_a_fixed_up_forward_ref_coexists_with_a_genuinely_dangling_one) {
    Model m;
    const std::uint64_t before = m.current_version();

    BulkTransaction t = m.begin_bulk();

    // Object 1: forward ref to an Account created later in the same batch,
    // legitimately fixed up via update() -- same pattern as
    // bulk_transaction_update_fixes_up_a_forward_reference_after_the_fact.
    auto ord1 = std::make_unique<Order>();
    ord1->code = "FIXED";
    const Ref<Order> o1 = t.create(std::move(ord1));  // account left null for now

    auto acc = std::make_unique<Account>();
    acc->name = "A1";
    const Ref<Account> a = t.create(std::move(acc));
    t.update(o1)->account = a;  // now legitimately valid

    // Object 2: a completely separate Order left with a genuinely dangling
    // ref -- never created anywhere in this batch.
    auto ord2 = std::make_unique<Order>();
    ord2->code = "DANGLING";
    ord2->account = Ref<Account>(Id{kLocalIdBit | 999u, 1});
    t.create(std::move(ord2));

    CommitResult r = m.commit_bulk_without_undo(t);
    CHECK(r.status == CommitStatus::Invalid);
    CHECK_EQ(m.current_version(), before);  // no wipe, no publish

    // Nothing partially installed -- not even the object whose forward ref
    // was legitimately fixed up.
    Snapshot s = m.snapshot();
    CHECK_EQ(s.size(), std::size_t{0});
    CHECK(s.find_by_key<&Account::name>("A1") == nullptr);
    CHECK(s.find_by_key<&Order::computed_key>("ord:FIXED") == nullptr);
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

TEST(bulk_load_rejects_a_duplicate_define_keys_value_within_the_batch) {
    Model m;
    make_account(m, "survivor");
    const std::uint64_t before = m.current_version();

    BulkTransaction t = m.begin_bulk();
    auto acc1 = std::make_unique<Account>();
    acc1->name = "DUP";
    t.create(std::move(acc1));
    auto acc2 = std::make_unique<Account>();
    acc2->name = "DUP";  // same define_keys() value as acc1, within the same batch
    t.create(std::move(acc2));

    CommitResult r = m.commit_bulk_without_undo(t);
    CHECK(r.status == CommitStatus::Invalid);
    CHECK_EQ(m.current_version(), before);  // no wipe, no publish -- rejected before any mutation
    CHECK(m.snapshot().find_by_key<&Account::name>("survivor") != nullptr);
}

// The duplicate-key check is keyed by field TAG (identifies a (type, field)
// pair), same as by_key_ itself -- so the same string value claimed on two
// DIFFERENT types' define_keys() fields is not a collision at all. Widget's
// computed_key() returns its `key` member unprefixed, so it can be made to
// collide, string-for-string, with Account::name.
TEST(bulk_load_allows_the_same_key_value_across_different_types) {
    Model m;
    BulkTransaction t = m.begin_bulk();
    auto acc = std::make_unique<Account>();
    acc->name = "DUP";
    t.create(std::move(acc));
    auto w = std::make_unique<Widget>();
    w->key = "DUP";  // same string, but Widget::computed_key() is a different field tag
    t.create(std::move(w));

    CommitResult r = m.commit_bulk_without_undo(t);
    CHECK(r.status == CommitStatus::Committed);

    Snapshot s = m.snapshot();
    CHECK(s.find_by_key<&Account::name>("DUP") != nullptr);
    CHECK(s.find_by_key<&Widget::computed_key>("DUP") != nullptr);
}

// Same idea within a single type: Gadget declares THREE define_keys()
// fields (label, serial, computed_key). label and computed_key can be made
// to hold the identical string across two different Gadgets without
// colliding, because the claims map is keyed per-field, not per-type-per-
// value -- label and computed_key are different field tags even though
// both live on Gadget.
TEST(bulk_load_allows_the_same_value_across_different_fields_of_the_same_type) {
    Model m;
    BulkTransaction t = m.begin_bulk();
    auto g1 = std::make_unique<Gadget>();
    g1->label = "gad:X";  // g1's label field == "gad:X"
    g1->serial = 1;
    t.create(std::move(g1));
    auto g2 = std::make_unique<Gadget>();
    g2->label = "X";  // g2's computed_key() ("gad:" + label) == "gad:X" too
    g2->serial = 2;
    t.create(std::move(g2));

    CommitResult r = m.commit_bulk_without_undo(t);
    CHECK(r.status == CommitStatus::Committed);

    Snapshot s = m.snapshot();
    CHECK(s.find_by_key<&Gadget::label>("gad:X") != nullptr);
    CHECK(s.find_by_key<&Gadget::computed_key>("gad:X") != nullptr);
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
    CHECK(sl.id() == local(0).id());  // the predicted id is the minted one

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
        CHECK(s.find(rs)->next.id() == rs.id());  // self-loop resolved to itself
        CHECK(s.find(r1)->next.id() == r2.id());  // cycle edges resolved crosswise
        CHECK(s.find(r2)->next.id() == r1.id());
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
    CHECK(s.find(real_g)->parent.id() == Id{});  // nulled, not cascaded
}

// commit_bulk_without_undo() installs through a SEPARATE, no-log path
// (add_cached_fields_no_log/add_cached_references_no_log) that has to seed
// Root::by_cached_field/by_cached_reference identically to the ordinary
// apply_create path -- otherwise a bulk-loaded model would look normal for
// find_by_key/cascade but silently have empty cached indexes. This checks
// every lookup family (key, find_by_field's cache-hit AND scan-fallback
// branches via qty/qty_scan, cached-referrer) against bulk-loaded data,
// then confirms a cascade over that same data empties the buckets cleanly
// (no tombstones), exactly like the Transaction-built equivalent in
// test_views.cpp.
TEST(commit_bulk_without_undo_populates_cached_field_scan_field_and_cached_referrer_indexes) {
    Model m;
    BulkTransaction t = m.begin_bulk();
    auto acc = std::make_unique<Account>();
    acc->name = "A1";
    const Ref<Account> a = t.create(std::move(acc));

    auto o1 = std::make_unique<Order>();
    o1->code = "O1";
    o1->account = a;
    o1->qty = 5;
    o1->qty_scan = 5;  // scan-only twin, see Order's own comment in test_types.h
    const Ref<Order> ord1 = t.create(std::move(o1));

    auto o2 = std::make_unique<Order>();
    o2->code = "O2";
    o2->account = a;
    o2->qty = 5;
    o2->qty_scan = 5;
    t.create(std::move(o2));

    auto o3 = std::make_unique<Order>();
    o3->code = "O3";
    o3->account = a;
    o3->qty = 7;
    o3->qty_scan = 7;
    o3->parent = ord1;  // Opt<Order>::parent is deliberately NOT cached -- see Order's own comment
    t.create(std::move(o3));

    const CommitResult r = m.commit_bulk_without_undo(t);
    CHECK(r.status == CommitStatus::Committed);
    const Ref<Account> real_a = r.to_real(a);
    const Ref<Order> real_ord1 = r.to_real(ord1);

    Snapshot s = m.snapshot();
    CHECK(s.find_by_key<&Account::name>("A1") != nullptr);
    CHECK(s.find_by_key<&Order::computed_key>("ord:O1") != nullptr);

    // Scan-fallback branch (qty_scan is scan-only).
    CHECK_EQ(s.find_by_field<&Order::qty_scan>(5).size(), std::size_t{2});
    CHECK_EQ(s.find_by_field<&Order::qty_scan>(7).size(), std::size_t{1});

    // Cache-hit branch (qty/computed_key are cache-declared).
    CHECK_EQ(s.find_by_field<&Order::qty>(5).size(), std::size_t{2});
    CHECK_EQ(s.find_by_field<&Order::qty>(7).size(), std::size_t{1});
    CHECK_EQ(s.find_by_field<&Order::computed_key>("ord:O1").size(), std::size_t{1});

    // account is cached, parent is not -- find_referrers resolves account
    // via the index and parent via the scan fallback, both correctly.
    CHECK_EQ(s.find_referrers<&Order::account>(real_a).size(), std::size_t{3});
    CHECK_EQ(s.find_referrers<&Order::parent>(real_ord1).size(), std::size_t{1});

    // Removing the hub cascades through the non-nullable account edges,
    // taking every Order down with it -- and every bucket above must end up
    // empty, not holding a stale entry for a now-dead id.
    remove_and_commit(m, real_a);
    Snapshot after = m.snapshot();
    CHECK(after.find_by_field<&Order::qty_scan>(5).empty());
    CHECK(after.find_by_field<&Order::qty_scan>(7).empty());
    CHECK(after.find_by_field<&Order::qty>(5).empty());
    CHECK(after.find_by_field<&Order::qty>(7).empty());
    CHECK(after.find_by_field<&Order::computed_key>("ord:O1").empty());
    CHECK(after.find_referrers<&Order::account>(real_a).empty());
}

// commit_bulk_without_undo() is documented to skip PreCommitFn/
// PreTransactionsFn/PostCommitFn entirely and to leave the undo list
// untouched (model.h: "do NOT fire for a bulk commit") -- unlike an ordinary
// try_commit(), which always produces exactly one new undo entry. Checked
// together since both are about what a bulk commit deliberately does NOT do.
TEST(commit_bulk_without_undo_never_fires_hooks_and_leaves_the_undo_list_untouched) {
    Model m;
    // Two ordinary commits FIRST, before any hook is installed, so there's
    // an existing undo list a bulk load must leave completely alone --
    // installing the hooks first would count these two toward the very
    // counters this test checks stay at zero across the bulk load.
    make_account(m, "PRE1");
    make_account(m, "PRE2");
    CHECK_EQ(m.list_undo().size(), std::size_t{2});
    const Model::Diagnostics::Status before_bulk = m.diagnostics();

    int pre_transactions_calls = 0, pre_commit_calls = 0, post_commit_calls = 0;
    m.set_pre_transactions([&](Model&, const Transaction&) { ++pre_transactions_calls; });
    m.set_pre_commit([&](Model&, const Transaction&, const std::vector<Change>&) {
        ++pre_commit_calls;
        return true;
    });
    m.set_post_commit([&](Model&, const Transaction&, const CommitResult&) { ++post_commit_calls; });

    BulkTransaction t = m.begin_bulk();
    auto acc = std::make_unique<Account>();
    acc->name = "BULK";
    t.create(std::move(acc));
    CHECK(m.commit_bulk_without_undo(t).status == CommitStatus::Committed);

    CHECK_EQ(pre_transactions_calls, 0);
    CHECK_EQ(pre_commit_calls, 0);
    CHECK_EQ(post_commit_calls, 0);
    // The bulk wipe clears undo_list_ outright (see its own doc comment) --
    // the two pre-existing entries are gone, not merely "not added to."
    CHECK(m.list_undo().empty());
    // commits_succeeded_ (part of the SAME counter family before_bulk read)
    // is untouched: a bulk load is excluded from ordinary commit accounting.
    CHECK_EQ(m.diagnostics().commits_succeeded, before_bulk.commits_succeeded);

    // Hooks and undo both work normally again on the very next ordinary
    // commit -- the bulk path's exclusion is scoped to itself only.
    const Ref<Account> after_bulk = make_account(m, "AFTER");
    CHECK_EQ(pre_transactions_calls, 1);
    CHECK_EQ(pre_commit_calls, 1);
    CHECK_EQ(post_commit_calls, 1);
    CHECK_EQ(m.list_undo().size(), std::size_t{1});
    CHECK(m.snapshot().find(after_bulk) != nullptr);

    m.set_pre_transactions({});
    m.set_pre_commit({});
    m.set_post_commit({});
}

