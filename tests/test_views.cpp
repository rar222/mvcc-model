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

// find_by_field returns EVERY match (unlike find_by_key's single winner),
// works on any field declared in either lookup family -- including one
// that's ALSO a define_keys field -- and is empty for a field declared in
// neither.
TEST(find_by_field_returns_every_match_and_only_for_declared_fields) {
    Model m;
    const Ref<Account> a = make_account(m, "A1", 1);  // Account::name: define_keys AND
                                                      // define_scan_fields (see Account's
                                                      // own doc comment)
    make_account(m, "OTHER", 3);
    make_order(m, "O1", a, {}, 5);  // also sets qty_scan == 5
    make_order(m, "O2", a, {}, 5);
    make_order(m, "O3", a, {}, 7);

    Snapshot s = m.snapshot();

    // A field can live in more than one lookup family at once: with no
    // duplicate value in play (define_keys() now rejects one -- see
    // field_index_duplicate_value_is_rejected in test_lookup.cpp), the
    // unique-key family and find_by_field's scan-fallback branch simply
    // agree on the same single object.
    CHECK(s.find_by_key<&Account::name>("A1") == s.find(a));
    auto by_name = s.view_by_field<&Account::name>("A1");
    CHECK_EQ(by_name.size(), std::size_t{1});
    CHECK_EQ(by_name[0]->id, a.raw());
    CHECK(s.find_by_field<&Account::name>("NOBODY").empty());

    // find_by_field's real multi-match story: Order::qty_scan is a PURE
    // scan-only field (never define_keys()'d, so genuine duplicates are
    // legal), and two orders legitimately share a value.
    CHECK_EQ(s.find_by_field<&Order::qty_scan>(5).size(), std::size_t{2});
    auto q5 = s.view_by_field<&Order::qty_scan>(5);
    CHECK_EQ(q5.size(), std::size_t{2});
    for (const auto& v : q5) CHECK_EQ(v[&Order::account]->balance, std::int64_t{1});

    // A computed (nullary const method) field, same as view_by_key -- also
    // cache-declared, so this exercises the cache-hit branch.
    auto o3 = s.view_by_field<&Order::computed_key>("ord:O3");
    CHECK_EQ(o3.size(), std::size_t{1});
    CHECK_EQ(o3[0]->qty, std::int64_t{7});

    // The gate: a field NOT declared in EITHER lookup family is invisible --
    // empty, even though an account with balance == 1 plainly exists --
    // exactly as find_by_key is empty for a field define_keys() never
    // mentioned. Every lookup family here shares that rule.
    CHECK(s.find_by_field<&Account::balance>(1).empty());
    CHECK(s.view_by_field<&Account::balance>(1).empty());
}

// The full lifecycle of find_by_field's cache-hit branch (multi-match,
// indexed): create, an update that moves an object between buckets, cascade
// delete emptying a bucket cleanly, and versioning like every other index.
TEST(find_by_field_tracks_creates_updates_and_cascades_and_is_versioned_via_cache) {
    Model m;
    const Ref<Account> a = make_account(m, "A1");
    const Ref<Order> o1 = make_order(m, "O1", a, {}, 5);
    make_order(m, "O2", a, {}, 5);
    make_order(m, "O3", a, {}, 7);

    Snapshot before = m.snapshot();
    CHECK_EQ(before.find_by_field<&Order::qty>(5).size(), std::size_t{2});
    CHECK_EQ(before.find_by_field<&Order::qty>(7).size(), std::size_t{1});
    CHECK(before.find_by_field<&Order::qty>(6).empty());

    // An update moves the object between buckets (reconcile)...
    {
        Transaction txn = m.begin();
        txn.update(o1)->qty = 7;
        commit_ok(m, txn);
    }
    Snapshot mid = m.snapshot();
    CHECK_EQ(mid.find_by_field<&Order::qty>(5).size(), std::size_t{1});
    CHECK_EQ(mid.find_by_field<&Order::qty>(7).size(), std::size_t{2});
    // ...and the index is versioned like everything else: the old snapshot
    // still sees the old buckets.
    CHECK_EQ(before.find_by_field<&Order::qty>(5).size(), std::size_t{2});

    // The view form traverses like any other view.
    auto views = mid.view_by_field<&Order::qty>(7);
    CHECK_EQ(views.size(), std::size_t{2});
    for (const auto& v : views) CHECK_EQ(v[&Order::account]->name, std::string("A1"));

    // A cascade (removing the account kills every order) drops each victim
    // from its bucket, leaving no tombstones behind.
    remove_and_commit(m, a);
    Snapshot after = m.snapshot();
    CHECK(after.find_by_field<&Order::qty>(5).empty());
    CHECK(after.find_by_field<&Order::qty>(7).empty());
}

// The full lifecycle of find_referrers's cache-hit branch, mirroring the
// cache-hit field test above: agreement with its scan-only twin
// account_scan (kept equal by make_order/the update below), reconciliation
// on reassignment, versioning, the View form, and direct removal.
TEST(find_referrers_matches_its_scan_only_twin_and_tracks_updates_and_is_versioned) {
    Model m;
    const Ref<Account> a1 = make_account(m, "A1");
    const Ref<Account> a2 = make_account(m, "A2");
    const Ref<Order> o1 = make_order(m, "O1", a1);
    const Ref<Order> o2 = make_order(m, "O2", a1);
    const Ref<Order> o3 = make_order(m, "O3", a2);
    // account_scan isn't set by make_order (see its own doc comment in
    // test_helpers.cpp) -- set explicitly here since this test never
    // cascade-deletes a1/a2, so the double-processing hazard doesn't apply.
    update_field(m, o1, [&](Order* p) { p->account_scan = a1; });
    update_field(m, o2, [&](Order* p) { p->account_scan = a1; });
    update_field(m, o3, [&](Order* p) { p->account_scan = a2; });

    Snapshot before = m.snapshot();
    // The cache-hit branch (account) and scan-fallback branch (account_scan)
    // agree on every account, empty included.
    for (Ref<Account> a : {a1, a2, make_account(m, "A3")}) {
        auto fast = before.find_referrers<&Order::account>(a);
        auto slow = before.find_referrers<&Order::account_scan>(a);
        CHECK_EQ(fast.size(), slow.size());
    }
    CHECK_EQ(before.find_referrers<&Order::account>(a1).size(), std::size_t{2});

    // Reassigning account moves the entry between buckets (reconcile); keep
    // account_scan equal so the two branches keep agreeing afterward too.
    {
        Transaction txn = m.begin();
        Order* u1 = txn.update(o1);
        u1->account = a2;
        u1->account_scan = a2;
        commit_ok(m, txn);
    }
    Snapshot mid = m.snapshot();
    CHECK_EQ(mid.find_referrers<&Order::account>(a1).size(), std::size_t{1});
    CHECK_EQ(mid.find_referrers<&Order::account>(a2).size(), std::size_t{2});
    CHECK_EQ(mid.find_referrers<&Order::account_scan>(a1).size(), std::size_t{1});
    CHECK_EQ(mid.find_referrers<&Order::account_scan>(a2).size(), std::size_t{2});
    // Versioned like everything else: the old snapshot still sees the old split.
    CHECK_EQ(before.find_referrers<&Order::account>(a1).size(), std::size_t{2});

    // The view form traverses like any other view.
    auto views = mid.view_referrers<&Order::account>(a2);
    CHECK_EQ(views.size(), std::size_t{2});
    for (const auto& v : views) CHECK_EQ(v[&Order::account]->name, std::string("A2"));

    // Removing o1 directly (no cascade) drops it from its bucket cleanly.
    remove_and_commit(m, o1);
    CHECK_EQ(m.snapshot().find_referrers<&Order::account>(a2).size(), std::size_t{1});
}

// for_each_referrers/all_of_referrers agree with find_referrers -- all three
// are built on the same cache-hit-or-scan-fallback dispatch (find_referrers
// itself is built on for_each_referrers). Exercised here on account (cache-
// declared, so this specifically walks the cache-hit branch).
TEST(for_each_and_all_of_referrers_match_find_referrers_via_cache) {
    Model m;
    const Ref<Account> a1 = make_account(m, "A1");
    const Ref<Account> a2 = make_account(m, "A2");
    make_order(m, "O1", a1);
    make_order(m, "O2", a1);
    make_order(m, "O3", a2);
    Snapshot s = m.snapshot();

    std::vector<Id> visited;
    s.for_each_referrers<&Order::account>(a1, [&](const Order& o) { visited.push_back(o.id); });
    CHECK_EQ(visited.size(), std::size_t{2});
    for (const Order* o : s.find_referrers<&Order::account>(a1))
        CHECK(std::find(visited.begin(), visited.end(), o->id) != visited.end());

    // all_of: true when every referrer satisfies the predicate...
    CHECK(s.all_of_referrers<&Order::account>(a1, [](const Order& o) { return o.qty > 0; }));
    // ...and genuinely short-circuits (stops after the first violation,
    // rather than just skipping further calls to the predicate).
    int checked = 0;
    const bool result = s.all_of_referrers<&Order::account>(a1, [&](const Order&) {
        ++checked;
        return false;  // fail immediately
    });
    CHECK(!result);
    CHECK_EQ(checked, 1);

    // Vacuously true for an account with no orders at all.
    CHECK(s.all_of_referrers<&Order::account>(make_account(m, "LONELY"),
                                              [](const Order&) { return false; }));

    // A field declared in define_references() (so it cascades/nulls
    // correctly) but NOT in define_cached_references() -- Order::parent, see
    // tests/test_types.h -- falls back to the scan instead of coming up
    // empty: for_each_referrers/all_of_referrers/find_referrers all find the
    // real referrer via that fallback, same as calling them on any other
    // declared-but-uncached field.
    const Ref<Order> hub = make_order(m, "HUB", a1);
    const Ref<Order> child = make_order(m, "CHILD", a1, hub);  // parent = hub, not cache-indexed
    Snapshot s2 = m.snapshot();

    std::size_t fallback_visits = 0;
    s2.for_each_referrers<&Order::parent>(hub, [&](const Order& o) {
        CHECK(o.id == child.raw());
        ++fallback_visits;
    });
    CHECK_EQ(fallback_visits, std::size_t{1});
    CHECK(!s2.all_of_referrers<&Order::parent>(hub, [](const Order&) { return false; }));
    CHECK_EQ(s2.find_referrers<&Order::parent>(hub).size(), std::size_t{1});  // the leaf it's built on agrees
}

// View-returning forms of for_each_referrers/all_of_referrers, and
// view_referrers built on top of for_each_view_referrers (single-pass, not
// find_referrers followed by a second wrap-in-View loop) -- see
// view_referrers' own doc comment. Exercised on account (cache-hit branch).
TEST(view_forms_of_referrers_bind_to_this_snapshot_via_cache) {
    Model m;
    const Ref<Account> a = make_account(m, "A1");
    make_order(m, "O1", a);
    make_order(m, "O2", a);
    Snapshot s = m.snapshot();

    int seen = 0;
    s.for_each_view_referrers<&Order::account>(a, [&](View<Order> v) {
        CHECK_EQ(v->account, a);
        ++seen;
    });
    CHECK_EQ(seen, 2);

    // all_of_view_referrers: true when everything satisfies the predicate,
    // false (and genuinely stopped, not just a different bool) when
    // something doesn't.
    CHECK(s.all_of_view_referrers<&Order::account>(a, [](View<Order> v) { return v->qty > 0; }));
    int checked = 0;
    const bool stopped = s.all_of_view_referrers<&Order::account>(a, [&](View<Order>) {
        ++checked;
        return false;
    });
    CHECK(!stopped);
    CHECK_EQ(checked, 1);

    const auto views = s.view_referrers<&Order::account>(a);
    CHECK_EQ(views.size(), std::size_t{2});
    for (const auto& v : views) CHECK_EQ(v[&Order::account]->name, std::string("A1"));

    // An uncached field (Order::parent, same as above) falls back to the
    // scan instead of coming up empty -- every view form here finds the
    // real referrer via that fallback, not a crash and not a silent no-op.
    const Ref<Order> hub = make_order(m, "HUB", a);
    const Ref<Order> child = make_order(m, "CHILD", a, hub);
    Snapshot s2 = m.snapshot();

    int fallback_seen = 0;
    s2.for_each_view_referrers<&Order::parent>(hub, [&](View<Order> v) {
        CHECK(v->id == child.raw());
        ++fallback_seen;
    });
    CHECK_EQ(fallback_seen, 1);
    CHECK(!s2.all_of_view_referrers<&Order::parent>(hub, [](View<Order>) { return false; }));
    CHECK_EQ(s2.view_referrers<&Order::parent>(hub).size(), std::size_t{1});
}

// Untyped counterpart of the cache-hit half of find_referrers/for_each_
// referrers/all_of_referrers -- same relationship the cached-FIELD raw trio
// has to find_by_field's own cache-hit half, since Root::by_cached_reference
// has the identical field-tag-keyed shape as Root::by_cached_field (see
// cached_referrer_short_circuit_raw's own doc comment). Exercised here with
// a single type (Order::account) for correctness; the "several unrelated
// types share one tag" scenario itself is already proven for cached FIELDS
// in test_inheritance.cpp -- this is the identical mechanism, just for
// references. Genuinely cache-only, no scan fallback (there's no ClassT
// here to scan with) -- so the undeclared-field case below stays truly
// empty/vacuous, unlike the typed find_referrers<&Order::parent>.
TEST(cached_referrer_raw_trio_matches_the_cache_hit_half_of_the_typed_forms) {
    Model m;
    const Ref<Account> a = make_account(m, "A1");
    const Ref<Order> o1 = make_order(m, "O1", a);
    const Ref<Order> o2 = make_order(m, "O2", a);
    Snapshot s = m.snapshot();

    const void* tag = model::field_tag<&Order::account>();

    const std::vector<const ObjectBase*> found = s.find_cached_referrers_raw(tag, a.raw());
    CHECK_EQ(found.size(), std::size_t{2});
    bool saw_o1 = false, saw_o2 = false;
    for (const ObjectBase* o : found) {
        CHECK(o->tag() == model::type_tag<Order>());
        if (o->id == o1.raw()) saw_o1 = true;
        if (o->id == o2.raw()) saw_o2 = true;
    }
    CHECK(saw_o1);
    CHECK(saw_o2);

    std::size_t visits = 0;
    s.for_each_cached_referrers_raw(tag, a.raw(), [&](const ObjectBase&) { ++visits; });
    CHECK_EQ(visits, std::size_t{2});

    int checked = 0;
    CHECK(!s.all_of_cached_referrers_raw(tag, a.raw(), [&](const ObjectBase&) {
        ++checked;
        return false;
    }));
    CHECK_EQ(checked, 1);  // stopped after the first, didn't visit the second

    // Undeclared field / no referrers: empty and vacuously true, not an error.
    CHECK(s.find_cached_referrers_raw(model::field_tag<&Order::parent>(), a.raw()).empty());
    std::size_t undeclared_visits = 0;
    s.for_each_cached_referrers_raw(model::field_tag<&Order::parent>(), a.raw(),
                                    [&](const ObjectBase&) { ++undeclared_visits; });
    CHECK_EQ(undeclared_visits, std::size_t{0});
    CHECK(s.all_of_cached_referrers_raw(model::field_tag<&Order::parent>(), a.raw(),
                                        [](const ObjectBase&) { return false; }));

    // A target Id that was never even created, on a DECLARED field: also
    // empty/vacuous, not a crash -- a stale/never-existent Id is a legal
    // input, not an error condition.
    CHECK(s.find_cached_referrers_raw(tag, Id{}).empty());
}

// A cascade-deleted referrer is dropped from its own outgoing bucket in
// Root::by_cached_reference, leaving no tombstone once the bucket empties.
TEST(find_referrers_tracks_cascade_delete_via_cache) {
    Model m;
    const Ref<Account> a = make_account(m, "A1");
    const Ref<Order> o1 = make_order(m, "O1", a);
    make_order(m, "O2", a);
    CHECK_EQ(m.snapshot().find_referrers<&Order::account>(a).size(), std::size_t{2});

    // Removing an order directly drops it from its own outgoing bucket.
    remove_and_commit(m, o1);
    CHECK_EQ(m.snapshot().find_referrers<&Order::account>(a).size(), std::size_t{1});

    // Deleting the account (Ref<Account> is non-nullable) cascades: the
    // remaining order dies too, and the account's own bucket empties out --
    // no tombstone left behind.
    remove_and_commit(m, a);
    CHECK(m.snapshot().find_referrers<&Order::account>(a).empty());
}

// The cascade-NULL path of Model::reconcile_cached_references -- see
// the in-body comment for why a dedicated (Node) type is needed here,
// since the shared test types don't cache a nullable reference field.
TEST(find_referrers_tracks_cascade_null_via_cache) {
    // Order::parent is deliberately NOT cached (see tests/test_types.h), so the
    // cascade-null path of Model::reconcile_cached_references needs a
    // CACHED nullable field to exercise -- Node, declared just above, exists
    // purely for this.
    Model m;
    const Ref<Node> root = make_node(m, "root");
    const Ref<Node> child = make_node(m, "child", root);
    CHECK_EQ(m.snapshot().find_referrers<&Node::parent>(root).size(), std::size_t{1});

    // Deleting `root` (Opt<Node> is nullable) nulls -- does not kill --
    // `child`, moving child's index entry out of root's bucket.
    remove_and_commit(m, root);
    Snapshot s = m.snapshot();
    CHECK(s.find(child) != nullptr);
    CHECK(!s.find(child)->parent);
    CHECK(s.find_referrers<&Node::parent>(root).empty());
}

// find_referrers on a field declared in define_references() but NOT in
// define_cached_references() falls back to the scan and finds the real
// referrer -- the opposite of the old two-function API's "no silent
// fallback" guarantee, which this merge deliberately removes (see model.h's
// Object<Derived> class comment).
TEST(find_referrers_falls_back_to_the_scan_for_a_field_never_declared_cached) {
    Model m;
    const Ref<Account> a = make_account(m, "A1");
    const Ref<Order> hub = make_order(m, "HUB", a);
    const Ref<Order> child = make_order(m, "C1", a, hub);

    // parent IS declared in define_references() (so it cascades/nulls
    // correctly) but deliberately NOT in define_cached_references() -- see
    // tests/test_types.h -- so find_referrers on it always takes the scan
    // fallback, and still finds the real referrer.
    const auto found = m.snapshot().find_referrers<&Order::parent>(hub);
    CHECK_EQ(found.size(), std::size_t{1});
    CHECK_EQ(found[0]->id, child.raw());
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
    CHECK_EQ(s.find_referrers<&Order::account>(a1).size(), std::size_t{1});
    CHECK(s.find_referrers<&Order::account>(a2).empty());
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
    CHECK_EQ(s.find_by_field<&Order::qty>(5).size(), std::size_t{1});
    CHECK(s.find_by_field<&Order::qty>(9).empty());
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

// Order::computed_key() is declared in ALL THREE of define_keys(),
// define_scan_fields(), and define_cached_fields() at once (see
// tests/test_types.h), so a single update() that changes what it returns
// must reconcile the key index AND the cached-field index together --
// find_by_field always takes the cache-hit branch for a field declared in
// both, so that's the only branch observable here (the scan-fallback
// branch's "always reflects the live value" behavior is generic and already
// covered via qty_scan elsewhere, not unique to computed_key).
TEST(updating_a_field_declared_in_multiple_lookup_families_reconciles_key_and_cache_together) {
    Model m;
    const Ref<Account> a = make_account(m, "A1");
    const Ref<Order> o = make_order(m, "OLD", a);  // computed_key() == "ord:OLD"

    Snapshot s = m.snapshot();
    CHECK_EQ(s.find_by_key<&Order::computed_key>("ord:OLD"), s.find(o));
    CHECK_EQ(s.find_by_field<&Order::computed_key>("ord:OLD").size(), std::size_t{1});

    update_field(m, o, [](Order* p) { p->code = "NEW"; });  // computed_key() now "ord:NEW"

    s = m.snapshot();
    CHECK(s.find_by_key<&Order::computed_key>("ord:OLD") == nullptr);
    CHECK(s.find_by_field<&Order::computed_key>("ord:OLD").empty());

    CHECK_EQ(s.find_by_key<&Order::computed_key>("ord:NEW"), s.find(o));
    CHECK_EQ(s.find_by_field<&Order::computed_key>("ord:NEW").size(), std::size_t{1});
}

// Same undo-log discipline as veto_rollback_restores_the_field_key_index
// (test_hooks.cpp) and veto_rollback_restores_the_cached_field_index above,
// but on a field that lives in by_field_ AND by_cached_field_ at once
// (Order::computed_key): a vetoed change must leave BOTH indexes exactly as
// they were, not one rolled back and the other not.
TEST(veto_rollback_restores_a_field_declared_in_multiple_lookup_families_simultaneously) {
    Model m;
    const Ref<Account> a = make_account(m, "A1");
    const Ref<Order> o = make_order(m, "OLD", a);  // computed_key() == "ord:OLD"

    m.set_pre_commit([](Model&, const Transaction&, const std::vector<Change>&) { return false; });
    Transaction txn = m.begin();
    txn.update(o)->code = "NEW";
    CHECK(m.try_commit(txn).status == CommitStatus::Vetoed);
    m.set_pre_commit({});

    make_account(m, "UNRELATED");  // publish a fresh root carrying both indexes
    Snapshot s = m.snapshot();
    CHECK_EQ(s.find(o)->computed_key(), std::string("ord:OLD"));

    CHECK_EQ(s.find_by_key<&Order::computed_key>("ord:OLD"), s.find(o));
    CHECK(s.find_by_key<&Order::computed_key>("ord:NEW") == nullptr);

    CHECK_EQ(s.find_by_field<&Order::computed_key>("ord:OLD").size(), std::size_t{1});
    CHECK(s.find_by_field<&Order::computed_key>("ord:NEW").empty());
}

// A cascade-deleted object must drop out of every lookup family it
// participates in AT ONCE -- find_by_key and find_by_field (cache-hit
// branch) on computed_key, plus find_by_field (cache-hit branch) on qty --
// not just whichever one a narrower test happens to check.
TEST(cascade_delete_removes_an_object_from_every_lookup_family_it_participates_in_at_once) {
    Model m;
    const Ref<Account> a = make_account(m, "A1");
    const Ref<Order> o = make_order(m, "O1", a, {}, 5);  // computed_key() == "ord:O1", qty == 5

    Snapshot s = m.snapshot();
    CHECK_EQ(s.find_by_key<&Order::computed_key>("ord:O1"), s.find(o));
    CHECK_EQ(s.find_by_field<&Order::computed_key>("ord:O1").size(), std::size_t{1});
    CHECK_EQ(s.find_by_field<&Order::qty>(5).size(), std::size_t{1});

    remove_and_commit(m, a);  // Order::account is a non-nullable Ref<Account> -- cascades, kills o

    s = m.snapshot();
    CHECK(s.find(o) == nullptr);
    CHECK(s.find_by_key<&Order::computed_key>("ord:O1") == nullptr);
    CHECK(s.find_by_field<&Order::computed_key>("ord:O1").empty());
    CHECK(s.find_by_field<&Order::qty>(5).empty());
}

// The one hole in an otherwise-complete family: for_each_view_by_field on
// the cache-hit branch -- every OTHER view/all_of variant on cache/scan
// fields is covered elsewhere in this file -- visiting every match (never
// stopping early) with each one bound to `s`.
TEST(for_each_view_by_field_visits_every_match_and_binds_views_to_this_snapshot_via_cache) {
    Model m;
    const Ref<Account> a = make_account(m, "A1");
    make_order(m, "O1", a, {}, 5);
    make_order(m, "O2", a, {}, 5);
    make_order(m, "O3", a, {}, 7);
    Snapshot s = m.snapshot();

    int hits = 0;
    s.for_each_view_by_field<&Order::qty>(5, [&](View<Order> v) {
        CHECK_EQ(v->qty, std::int64_t{5});
        CHECK_EQ(v[&Order::account]->name, std::string("A1"));
        ++hits;
    });
    CHECK_EQ(hits, 2);

    // A value with no matches visits nobody.
    int no_hits = 0;
    s.for_each_view_by_field<&Order::qty>(99, [&](View<Order>) { ++no_hits; });
    CHECK_EQ(no_hits, 0);

    // Delegates to for_each_by_field, so it records a lookup exactly like
    // every other entry point in this family.
    const LookupCounts before = m.lookup_stats<&Order::qty>();
    s.for_each_view_by_field<&Order::qty>(7, [](View<Order>) {});
    CHECK_EQ(m.lookup_stats<&Order::qty>().cached_calls, before.cached_calls + 1);
}

// View<T>::operator* dereferences to the same object operator-> reaches
// through -- every other test in this file uses v->field exclusively, so
// operator* itself is otherwise only ever exercised incidentally (inside
// Snapshot::view_by_predicate's own pred(*v), never asserted directly).
TEST(view_operator_star_dereferences_to_the_same_object_as_operator_arrow) {
    Model m;
    const Ref<Account> a = make_account(m, "A1");
    const Ref<Order> o = make_order(m, "O1", a, {}, 5);
    Snapshot s = m.snapshot();

    std::optional<View<Order>> v = s.view(o);
    CHECK(v.has_value());
    const Order& via_star = **v;
    CHECK_EQ(&via_star, v->operator->());
    CHECK_EQ(via_star.qty, std::int64_t{5});
}

