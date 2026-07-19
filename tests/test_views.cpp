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


// View<T>::operator[] follows Ref<>/Opt<> fields -- including chained
// hops -- without the caller ever passing a Snapshot explicitly.
TEST(view_traverses_refs_without_plumbing_the_snapshot) {
    Model m;
    const Ref<Account> a = make_account(m, "A1", 77);
    const Ref<Order> p = make_order(m, "P", a, {}, 3);
    make_order(m, "C", a, p, 9);

    Snapshot s = m.snapshot();
    auto c = s.view_by_key<&Order::computed_key>("ord:C");
    CHECK(c.has_value());

    CHECK_EQ((*c)->qty, 9);
    CHECK_EQ((*c)[&Order::account]->name, std::string("A1"));
    CHECK_EQ((*c)[&Order::account]->balance, 77);

    auto parent = (*c)[&Order::parent];
    CHECK(parent.has_value());
    CHECK_EQ((*parent)->code, std::string("P"));
    CHECK_EQ((*parent)[&Order::account]->name, std::string("A1"));
    CHECK(!(*parent)[&Order::parent].has_value());
}

// Snapshot::view()/view_by_key() on a since-deleted handle return an
// empty optional rather than a dangling View.
TEST(view_of_a_stale_handle_is_empty) {
    Model m;
    const Ref<Account> a = make_account(m, "A1");
    const Ref<Order> o = make_order(m, "O1", a);
    remove_and_commit(m, o);

    Snapshot s = m.snapshot();
    CHECK(!s.view(o).has_value());
    CHECK(!s.view_by_key<&Order::computed_key>("ord:O1").has_value());
}

// The scan family returns EVERY match (unlike find_by_key's single
// winner), works on any declared scan field including one that's ALSO a
// define_keys field, and is empty for a field never declared scan.
TEST(find_by_scan_field_returns_every_match_and_only_for_declared_fields) {
    Model m;
    const Ref<Account> a =
        make_account(m, "DUP", 1);  // Account::name: define_keys AND define_scan_fields
    const Ref<Account> b =
        make_account(m, "DUP", 2);  // duplicate name: the unique index keeps only this one
    make_account(m, "OTHER", 3);
    make_order(m, "O1", a, {}, 5);
    make_order(m, "O2", a, {}, 5);
    make_order(m, "O3", a, {}, 7);

    Snapshot s = m.snapshot();

    // The unique-key index sees one winner for a duplicate value...
    CHECK(s.find_by_key<&Account::name>("DUP") == s.find(b));
    // ...the scan family sees every object, on the same field.
    auto dups = s.view_by_scan_field<&Account::name>("DUP");
    CHECK_EQ(dups.size(), std::size_t{2});
    std::int64_t balances = 0;
    for (const auto& v : dups) balances += v->balance;
    CHECK_EQ(balances, std::int64_t{3});  // 1 + 2: both objects, not the winner twice
    CHECK(s.find_by_scan_field<&Account::name>("NOBODY").empty());

    // The find form, on a field define_keys() never mentioned, and the Views
    // traverse like any other.
    CHECK_EQ(s.find_by_scan_field<&Order::qty>(5).size(), std::size_t{2});
    auto q5 = s.view_by_scan_field<&Order::qty>(5);
    CHECK_EQ(q5.size(), std::size_t{2});
    for (const auto& v : q5) CHECK_EQ(v[&Order::account]->balance, std::int64_t{1});

    // A computed (nullary const method) field, same as view_by_key.
    auto o3 = s.view_by_scan_field<&Order::computed_key>("ord:O3");
    CHECK_EQ(o3.size(), std::size_t{1});
    CHECK_EQ(o3[0]->qty, std::int64_t{7});

    // The gate: a field NOT declared in define_scan_fields() is invisible to
    // this family -- empty, even though an account with balance == 1 plainly
    // exists -- exactly as find_by_key is empty for a field define_keys()
    // never mentioned. All three lookup families share that rule.
    CHECK(s.find_by_scan_field<&Account::balance>(1).empty());
    CHECK(s.view_by_scan_field<&Account::balance>(1).empty());
}

// The full lifecycle of a cached (multi-match, indexed) field: create,
// an update that moves an object between buckets, cascade delete
// emptying a bucket cleanly, and versioning like every other index.
TEST(find_by_cached_field_tracks_creates_updates_and_cascades_and_is_versioned) {
    Model m;
    const Ref<Account> a = make_account(m, "A1");
    const Ref<Order> o1 = make_order(m, "O1", a, {}, 5);
    make_order(m, "O2", a, {}, 5);
    make_order(m, "O3", a, {}, 7);

    Snapshot before = m.snapshot();
    CHECK_EQ(before.find_by_cached_field<&Order::qty>(5).size(), std::size_t{2});
    CHECK_EQ(before.find_by_cached_field<&Order::qty>(7).size(), std::size_t{1});
    CHECK(before.find_by_cached_field<&Order::qty>(6).empty());

    // An update moves the object between buckets (reconcile)...
    {
        Transaction txn = m.begin();
        txn.update(o1)->qty = 7;
        commit_ok(m, txn);
    }
    Snapshot mid = m.snapshot();
    CHECK_EQ(mid.find_by_cached_field<&Order::qty>(5).size(), std::size_t{1});
    CHECK_EQ(mid.find_by_cached_field<&Order::qty>(7).size(), std::size_t{2});
    // ...and the index is versioned like everything else: the old snapshot
    // still sees the old buckets.
    CHECK_EQ(before.find_by_cached_field<&Order::qty>(5).size(), std::size_t{2});

    // The view form traverses like any other view.
    auto views = mid.view_by_cached_field<&Order::qty>(7);
    CHECK_EQ(views.size(), std::size_t{2});
    for (const auto& v : views) CHECK_EQ(v[&Order::account]->name, std::string("A1"));

    // A cascade (removing the account kills every order) drops each victim
    // from its bucket, leaving no tombstones behind.
    remove_and_commit(m, a);
    Snapshot after = m.snapshot();
    CHECK(after.find_by_cached_field<&Order::qty>(5).empty());
    CHECK(after.find_by_cached_field<&Order::qty>(7).empty());
}

// The full lifecycle of a cached REFERENCE field, mirroring the cached-
// field test above: agreement with the unindexed scan, reconciliation
// on reassignment, versioning, the View form, and direct removal.
TEST(find_cached_referrers_matches_the_slow_scan_and_tracks_updates_and_is_versioned) {
    Model m;
    const Ref<Account> a1 = make_account(m, "A1");
    const Ref<Account> a2 = make_account(m, "A2");
    const Ref<Order> o1 = make_order(m, "O1", a1);
    make_order(m, "O2", a1);
    make_order(m, "O3", a2);

    Snapshot before = m.snapshot();
    // The indexed and scan forms agree on every account, empty included.
    for (Ref<Account> a : {a1, a2, Ref<Account>(make_account(m, "A3"))}) {
        auto fast = before.find_cached_referrers<&Order::account>(a);
        auto slow = before.find_referrers<&Order::account>(a);
        CHECK_EQ(fast.size(), slow.size());
        for (const Order* o : fast) CHECK(std::find(slow.begin(), slow.end(), o) != slow.end());
    }
    CHECK_EQ(before.find_cached_referrers<&Order::account>(a1).size(), std::size_t{2});

    // Reassigning account moves the entry between buckets (reconcile).
    {
        Transaction txn = m.begin();
        txn.update(o1)->account = a2;
        commit_ok(m, txn);
    }
    Snapshot mid = m.snapshot();
    CHECK_EQ(mid.find_cached_referrers<&Order::account>(a1).size(), std::size_t{1});
    CHECK_EQ(mid.find_cached_referrers<&Order::account>(a2).size(), std::size_t{2});
    // Versioned like everything else: the old snapshot still sees the old split.
    CHECK_EQ(before.find_cached_referrers<&Order::account>(a1).size(), std::size_t{2});

    // The view form traverses like any other view.
    auto views = mid.view_cached_referrers<&Order::account>(a2);
    CHECK_EQ(views.size(), std::size_t{2});
    for (const auto& v : views) CHECK_EQ(v[&Order::account]->name, std::string("A2"));

    // Removing o1 directly (no cascade) drops it from its bucket cleanly.
    remove_and_commit(m, o1);
    CHECK_EQ(m.snapshot().find_cached_referrers<&Order::account>(a2).size(), std::size_t{1});
}

// A cascade-deleted referrer is dropped from its own outgoing bucket in
// Root::by_cached_reference, leaving no tombstone once the bucket empties.
TEST(find_cached_referrers_tracks_cascade_delete) {
    Model m;
    const Ref<Account> a = make_account(m, "A1");
    const Ref<Order> o1 = make_order(m, "O1", a);
    make_order(m, "O2", a);
    CHECK_EQ(m.snapshot().find_cached_referrers<&Order::account>(a).size(), std::size_t{2});

    // Removing an order directly drops it from its own outgoing bucket.
    remove_and_commit(m, o1);
    CHECK_EQ(m.snapshot().find_cached_referrers<&Order::account>(a).size(), std::size_t{1});

    // Deleting the account (Ref<Account> is non-nullable) cascades: the
    // remaining order dies too, and the account's own bucket empties out --
    // no tombstone left behind.
    remove_and_commit(m, a);
    CHECK(m.snapshot().find_cached_referrers<&Order::account>(a).empty());
}

// The cascade-NULL path of Model::reconcile_cached_references -- see
// the in-body comment for why a dedicated (Node) type is needed here,
// since the shared demo types don't cache a nullable reference field.
TEST(find_cached_referrers_tracks_cascade_null) {
    // Order::parent is deliberately NOT cached (see demo/types.h), so the
    // cascade-null path of Model::reconcile_cached_references needs a
    // CACHED nullable field to exercise -- Node, declared just above, exists
    // purely for this.
    Model m;
    const Ref<Node> root = make_node(m, "root");
    const Ref<Node> child = make_node(m, "child", root);
    CHECK_EQ(m.snapshot().find_cached_referrers<&Node::parent>(root).size(), std::size_t{1});

    // Deleting `root` (Opt<Node> is nullable) nulls -- does not kill --
    // `child`, moving child's index entry out of root's bucket.
    remove_and_commit(m, root);
    Snapshot s = m.snapshot();
    CHECK(s.find(child) != nullptr);
    CHECK(!s.find(child)->parent);
    CHECK(s.find_cached_referrers<&Node::parent>(root).empty());
}

// The cached-reference family's own version of the shared
// "undeclared is invisible" rule: the slow scan still finds an
// uncached field's referrers; the index does not, with no silent fallback.
TEST(find_cached_referrers_is_empty_for_a_field_never_declared_cached) {
    Model m;
    const Ref<Account> a = make_account(m, "A1");
    const Ref<Order> hub = make_order(m, "HUB", a);
    make_order(m, "C1", a, hub);

    // parent IS declared in define_references() (so it cascades/nulls
    // correctly) but deliberately NOT in define_cached_references() -- see
    // demo/types.h. The slow scan still finds it; the indexed form is
    // invisible to it, same "undeclared is invisible" rule the other three
    // lookup families share -- there is no silent fallback to the scan.
    CHECK(!m.snapshot().find_referrers<&Order::parent>(hub).empty());
    CHECK(m.snapshot().find_cached_referrers<&Order::parent>(hub).empty());
}

// Undo-log discipline for Root::by_cached_reference: a vetoed
// reassignment must leave the index exactly as it was before the
// attempt, once a later commit actually publishes.
TEST(veto_rollback_restores_the_cached_reference_index) {
    Model m;
    const Ref<Account> a1 = make_account(m, "A1");
    const Ref<Account> a2 = make_account(m, "A2");
    const Ref<Order> o = make_order(m, "O1", a1);

    m.set_pre_commit([](Model&, const Transaction&, const std::vector<Change>&) { return false; });
    Transaction txn = m.begin();
    txn.update(o)->account = a2;
    CHECK(m.try_commit(txn).status == CommitStatus::Vetoed);
    m.set_pre_commit({});

    make_account(m, "UNRELATED");  // publish a fresh root carrying the index
    Snapshot s = m.snapshot();
    CHECK_EQ(s.find_cached_referrers<&Order::account>(a1).size(), std::size_t{1});
    CHECK(s.find_cached_referrers<&Order::account>(a2).empty());
    CHECK(s.find(o)->account == a1);
}

// Same undo-log discipline for Root::by_cached_field: a vetoed value
// change must leave the index exactly as it was before the attempt.
TEST(veto_rollback_restores_the_cached_field_index) {
    Model m;
    const Ref<Account> a = make_account(m, "A1");
    const Ref<Order> o = make_order(m, "O1", a, {}, 5);

    m.set_pre_commit([](Model&, const Transaction&, const std::vector<Change>&) { return false; });
    Transaction txn = m.begin();
    txn.update(o)->qty = 9;
    CHECK(m.try_commit(txn).status == CommitStatus::Vetoed);
    m.set_pre_commit({});

    make_account(m, "UNRELATED");  // publish a fresh root carrying the index
    Snapshot s = m.snapshot();
    CHECK_EQ(s.find_by_cached_field<&Order::qty>(5).size(), std::size_t{1});
    CHECK(s.find_by_cached_field<&Order::qty>(9).empty());
    CHECK_EQ(s.find(o)->qty, std::int64_t{5});
}

// Two Views built from different Snapshots of the same object never mix
// versions: each traverses strictly through the Snapshot it was bound
// to, even when both ultimately describe the same underlying object.
TEST(a_view_always_traverses_its_own_snapshot) {
    Model m;
    const Ref<Account> a = make_account(m, "A1");
    const Ref<Order> p = make_order(m, "P", a);
    const Ref<Order> c = make_order(m, "C", a, p);

    Snapshot v1 = m.snapshot();
    remove_and_commit(m, p);
    Snapshot v2 = m.snapshot();

    auto c1 = v1.view_by_key<&Order::computed_key>("ord:C");
    CHECK(c1.has_value());
    auto par1 = (*c1)[&Order::parent];
    CHECK(par1.has_value());
    CHECK_EQ((*par1)->code, std::string("P"));

    auto c2 = v2.view_by_key<&Order::computed_key>("ord:C");
    CHECK(c2.has_value());
    CHECK(!(*c2)[&Order::parent].has_value());

    CHECK(&(*c1).snapshot() != &(*c2).snapshot());
    CHECK_EQ((*c1)[&Order::account]->name, (*c2)[&Order::account]->name);
    (void)c;
}

// Snapshot::for_each_view<T>() is type-filtered exactly like for_each<T>(),
// and each yielded View can chain into a Ref<> field's own View.
TEST(for_each_view_is_type_filtered) {
    Model m;
    const Ref<Account> a = make_account(m, "A1", 5);
    make_order(m, "O1", a, {}, 1);
    make_order(m, "O2", a, {}, 2);

    Snapshot s = m.snapshot();
    std::int64_t total = 0;
    int seen = 0;
    s.for_each_view<Order>([&](View<Order> o) {
        ++seen;
        total += o->qty;
        total += o[&Order::account]->balance;
    });
    CHECK_EQ(seen, 2);
    CHECK_EQ(total, std::int64_t{1 + 2 + 5 + 5});
}

