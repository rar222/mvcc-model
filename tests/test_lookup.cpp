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


// Snapshot::for_each<T>() visits only objects of exactly that type, even
// when other types coexist in the same model.
TEST(for_each_is_type_filtered) {
    Model m;
    const Ref<Account> a1 = make_account(m, "A1");
    const Ref<Account> a2 = make_account(m, "A2");
    make_order(m, "O1", a1);
    make_order(m, "O2", a2);
    make_order(m, "O3", a1);

    Snapshot s = m.snapshot();
    int orders = 0, accounts = 0;
    s.for_each<Order>([&](const Order& o) {
        CHECK(!o.code.empty());
        ++orders;
    });
    s.for_each<Account>([&](const Account& a) {
        CHECK(!a.name.empty());
        ++accounts;
    });
    CHECK_EQ(orders, 3);
    CHECK_EQ(accounts, 2);
    CHECK_EQ(s.size(), std::size_t{5});
}

// find_by_key<Field> is type-safe: a value that matches under a
// DIFFERENT type's key (or the wrong field on the same value) returns
// null rather than a miscast object.
TEST(a_typed_lookup_of_the_wrong_type_is_null_not_a_bad_cast) {
    Model m;
    const Ref<Account> a = make_account(m, "A1");
    const Ref<Order> o = make_order(m, "O1", a);

    Snapshot s = m.snapshot();
    CHECK(s.find_by_key<&Order::computed_key>("ord:O1") != nullptr);
    CHECK(s.find_by_key<&Account::name>("ord:O1") == nullptr);
    CHECK(s.find_by_key<&Order::computed_key>("A1") == nullptr);
    CHECK(s.find_by_key<&Account::name>("A1") != nullptr);
    (void)o;
}

// Two different types can share the identical raw key string without
// colliding -- define_keys() indexes per (type, field), not one global
// namespace -- and deleting one type's entry leaves the other untouched.
TEST(external_keys_are_scoped_per_type_not_global) {
    Model m;
    const Ref<Account> a = make_account(m, "A1");
    const Ref<Order> o = make_order(m, "X", a);     // computed_key() == "ord:X"
    const Ref<Widget> w = make_widget(m, "ord:X");  // same raw string, different type

    Snapshot s = m.snapshot();
    CHECK_EQ(s.find_by_key<&Order::computed_key>("ord:X")->id, o.raw());
    CHECK_EQ(s.find_by_key<&Widget::computed_key>("ord:X")->id, w.raw());
    CHECK(s.find_by_key<&Account::name>("ord:X") == nullptr);
    CHECK_EQ(s.size(), std::size_t{3});

    remove_and_commit(m, o);
    s = m.snapshot();
    CHECK(s.find_by_key<&Order::computed_key>("ord:X") == nullptr);
    CHECK(s.find_by_key<&Widget::computed_key>("ord:X") != nullptr);
}

// find_all()/find_all_view() run a predicate scan restricted to one
// type, in both the raw-pointer and the View-returning forms.
TEST(find_all_runs_an_arbitrary_predicate_over_one_type) {
    Model m;
    make_account(m, "A1", 50);
    make_account(m, "A2", 150);
    make_account(m, "A3", 250);

    Snapshot s = m.snapshot();
    const auto rich = s.find_all<Account>([](const Account& a) { return a.balance >= 100; });
    CHECK_EQ(rich.size(), std::size_t{2});
    for (const Account* a : rich) CHECK(a->balance >= 100);

    const auto rich_views =
        s.find_all_view<Account>([](const Account& a) { return a.balance >= 100; });
    CHECK_EQ(rich_views.size(), std::size_t{2});
    for (const View<Account>& v : rich_views) CHECK(v->balance >= 100);

    const auto none = s.find_all<Account>([](const Account&) { return false; });
    CHECK(none.empty());
}

// The unindexed reverse-lookup family (for_each_referrer/find_referrers
// and their View forms) correctly reports every referrer of a target
// across both Ref<> and Opt<> fields, and empty for an unreferenced one.
TEST(find_referrers_answers_who_points_at_me) {
    Model m;
    const Ref<Account> a1 = make_account(m, "A1");
    const Ref<Account> a2 = make_account(m, "A2");
    const Ref<Order> p = make_order(m, "P", a1);
    const Ref<Order> c1 = make_order(m, "C1", a1, p);
    const Ref<Order> c2 = make_order(m, "C2", a2, p);
    make_order(m, "OTHER", a2);

    Snapshot s = m.snapshot();

    const auto a1_orders = s.find_referrers<&Order::account>(a1);
    CHECK_EQ(a1_orders.size(), std::size_t{2});
    for (const Order* o : a1_orders) CHECK(o->account == a1);

    const auto children = s.find_referrers<&Order::parent>(p);
    CHECK_EQ(children.size(), std::size_t{2});
    std::vector<Id> ids;
    for (const Order* o : children) ids.push_back(o->id);
    CHECK(std::find(ids.begin(), ids.end(), c1.raw()) != ids.end());
    CHECK(std::find(ids.begin(), ids.end(), c2.raw()) != ids.end());

    const Ref<Account> lonely = make_account(m, "LONELY");
    s = m.snapshot();
    CHECK(s.find_referrers<&Order::account>(lonely).empty());

    auto va1 = s.view(a1);
    int seen = 0;
    va1->for_each_referrer<&Order::account>([&](View<Order> v) {
        CHECK(v->account == a1);
        ++seen;
    });
    CHECK_EQ(seen, 2);

    const auto view_children = s.view(*s.find(p)).find_referrers<&Order::parent>();
    CHECK_EQ(view_children.size(), std::size_t{2});
}

// find_by_key works for both a string field and an arithmetic field, and
// returns null for an undeclared field or a value that was never set.
TEST(find_by_key_looks_up_define_keys_declared_fields) {
    Model m;
    const Ref<Gadget> g1 = make_gadget(m, "Widget", 111);
    const Ref<Gadget> g2 = make_gadget(m, "Sprocket", 222);

    Snapshot s = m.snapshot();
    CHECK_EQ(s.find_by_key<&Gadget::label>("Widget")->id, g1.raw());
    CHECK_EQ(s.find_by_key<&Gadget::serial>(222)->id, g2.raw());
    CHECK(s.find_by_key<&Gadget::label>("nope") == nullptr);
    CHECK(s.find_by_key<&Gadget::serial>(999) == nullptr);
    CHECK(s.find_by_key<&Order::code>("anything") == nullptr);
}

// Reassigning a define_keys()-declared field via update() moves its
// index entry: the old value stops resolving, the new value resolves to
// the same object.
TEST(field_index_tracks_a_reassigned_indexed_field) {
    Model m;
    const Ref<Gadget> g = make_gadget(m, "Widget", 111);
    CHECK(m.snapshot().find_by_key<&Gadget::label>("Widget") != nullptr);

    update_field(m, g, [](Gadget* p) { p->label = "Renamed"; });

    Snapshot s = m.snapshot();
    CHECK(s.find_by_key<&Gadget::label>("Widget") == nullptr);
    CHECK_EQ(s.find_by_key<&Gadget::label>("Renamed")->id, g.raw());
    CHECK_EQ(s.find_by_key<&Gadget::serial>(111)->id, g.raw());
}

// Removing an object drops every one of its define_keys() index
// entries -- they don't linger as stale, dangling lookups.
TEST(field_index_is_cleaned_up_on_removal) {
    Model m;
    const Ref<Gadget> g = make_gadget(m, "Widget", 111);
    remove_and_commit(m, g);

    Snapshot s = m.snapshot();
    CHECK(s.find_by_key<&Gadget::label>("Widget") == nullptr);
    CHECK(s.find_by_key<&Gadget::serial>(111) == nullptr);
}

// Two objects sharing the same indexed value: the second create is rejected
// as CommitStatus::Invalid rather than silently overwriting the first one's
// index entry.
TEST(field_index_duplicate_value_is_rejected) {
    Model m;
    const Ref<Gadget> g1 = make_gadget(m, "Same", 1);

    Transaction txn = m.begin();
    auto g2 = std::make_unique<Gadget>();
    g2->label = "Same";
    g2->serial = 2;
    txn.create(std::move(g2));
    const CommitResult res = m.try_commit(txn);
    CHECK(res.status == CommitStatus::Invalid);
    CHECK(res.error.has_value());
    CHECK(!res.error->bad_target);  // key collisions have no single target to classify against base

    Snapshot s = m.snapshot();
    CHECK_EQ(s.find_by_key<&Gadget::label>("Same")->id, g1.raw());  // first write still stands
    CHECK(s.find_by_key<&Gadget::serial>(2) == nullptr);            // rejected create never installed
}

// The SAME key, reused after the original holder is deleted, is fine: a
// duplicate is only ever a live collision, never "this value was ever used
// before."
TEST(a_deleted_keys_value_can_be_reclaimed_by_a_different_record) {
    Model m;
    const Ref<Gadget> g1 = make_gadget(m, "Same", 1);
    remove_and_commit(m, g1);
    const Ref<Gadget> g2 = make_gadget(m, "Same", 2);  // different record, same label

    Snapshot s = m.snapshot();
    CHECK_EQ(s.find_by_key<&Gadget::label>("Same")->id, g2.raw());
    CHECK(s.find(g1) == nullptr);
}

// Two creates claiming the same key WITHIN one transaction are rejected too
// -- there's no "current holder" to check against, since both claims are
// equally new, but it's still two different objects wanting one value.
TEST(field_index_duplicate_value_within_one_transaction_is_rejected) {
    Model m;
    Transaction txn = m.begin();
    auto g1 = std::make_unique<Gadget>();
    g1->label = "Same";
    g1->serial = 1;
    txn.create(std::move(g1));
    auto g2 = std::make_unique<Gadget>();
    g2->label = "Same";
    g2->serial = 2;
    txn.create(std::move(g2));

    const CommitResult res = m.try_commit(txn);
    CHECK(res.status == CommitStatus::Invalid);
    CHECK(res.error.has_value());

    Snapshot s = m.snapshot();
    CHECK(s.find_by_key<&Gadget::label>("Same") == nullptr);  // neither create ever installed
}

// A same-transaction "swap": g_a moves OFF its original key onto a new one,
// while g_b moves ONTO that now-vacated key, in the same Transaction. This
// must succeed regardless of which of the two updates the model happens to
// apply first internally (Transaction::local_updated_ has no defined
// iteration order) -- see Model::validate_field_key_uniqueness and
// Model::reconcile_field_keys's conditional erase.
TEST(a_key_vacated_and_reclaimed_in_the_same_transaction_is_not_a_collision) {
    Model m;
    const Ref<Gadget> g_a = make_gadget(m, "KeyA", 1);
    const Ref<Gadget> g_b = make_gadget(m, "KeyB", 2);

    Transaction txn = m.begin();
    txn.update(g_a)->label = "KeyA-moved";
    txn.update(g_b)->label = "KeyA";  // takes over g_a's original label
    const CommitResult res = m.try_commit(txn);
    CHECK(res.status == CommitStatus::Committed);

    Snapshot s = m.snapshot();
    CHECK(s.find_by_key<&Gadget::label>("KeyA") != nullptr);
    CHECK_EQ(s.find_by_key<&Gadget::label>("KeyA")->id, g_b.raw());
    CHECK_EQ(s.find_by_key<&Gadget::label>("KeyA-moved")->id, g_a.raw());
    CHECK(s.find_by_key<&Gadget::label>("KeyB") == nullptr);  // g_b vacated it
}

// A two-hop cascade: deleting an Account kills an Order (Ref<>), which
// in turn nulls -- not kills -- a second Order's Opt<> pointing at it,
// while that second Order's own Ref<> edge stays intact.
TEST(opt_survives_a_cascade_that_ref_does_not) {
    Model m;
    const Ref<Account> a1 = make_account(m, "A1");
    const Ref<Account> a2 = make_account(m, "A2");
    const Ref<Order> p = make_order(m, "P", a1);
    const Ref<Order> c = make_order(m, "C", a2, p);  // Ref -> a2, Opt -> p

    remove_and_commit(m, a1);  // kills p (Ref<Account>), which nulls c.parent (Opt<Order>)

    Snapshot s = m.snapshot();
    CHECK(s.find(a1) == nullptr);
    CHECK(s.find(p) == nullptr);
    CHECK(s.find(c) != nullptr);
    CHECK(!s.find(c)->parent);
    CHECK_EQ(s.resolve(s.find(c)->account).name, std::string("A2"));
}


// Root::by_field (the define_keys index) is versioned like the object
// store itself: an older Snapshot keeps answering find_by_key as of its
// own version, even as later commits add and remove entries.
TEST(the_index_is_versioned_like_everything_else) {
    Model m;
    const Ref<Account> a = make_account(m, "A1");
    make_order(m, "O1", a);
    Snapshot v1 = m.snapshot();

    make_order(m, "O2", a);
    Snapshot v2 = m.snapshot();

    const Ref<Order> o1 = Ref<Order>(m.snapshot().find_by_key<&Order::computed_key>("ord:O1")->id);
    remove_and_commit(m, o1);
    Snapshot v3 = m.snapshot();

    CHECK(v1.find_by_key<&Order::computed_key>("ord:O1") != nullptr);
    CHECK(v1.find_by_key<&Order::computed_key>("ord:O2") == nullptr);
    CHECK(v2.find_by_key<&Order::computed_key>("ord:O1") != nullptr);
    CHECK(v2.find_by_key<&Order::computed_key>("ord:O2") != nullptr);
    CHECK(v3.find_by_key<&Order::computed_key>("ord:O1") == nullptr);
    CHECK(v3.find_by_key<&Order::computed_key>("ord:O2") != nullptr);
}

// A long randomized sequence of interleaved creates/removes leaves the
// surviving objects' index entries exactly consistent with what
// for_each<T>()/find() report -- no drift after heavy churn.
TEST(index_survives_heavy_churn_and_stays_consistent) {
    Model m;
    const Ref<Account> a = make_account(m, "A1");

    std::vector<Ref<Order>> live;
    std::mt19937 rng(5);
    int next = 0;
    for (int i = 0; i < 400; ++i) {
        if (rng() % 2 || live.empty()) {
            live.push_back(make_order(m, "O" + std::to_string(next++), a));
        } else {
            const std::size_t idx = rng() % live.size();
            remove_and_commit(m, live[idx]);
            live.erase(live.begin() + idx);
        }
    }

    Snapshot s = m.snapshot();
    for (const Ref<Order>& r : live) {
        const Order* o = s.find(r);
        CHECK(o != nullptr);
        CHECK(s.find_by_key<&Order::computed_key>(o->computed_key()) == o);
    }
    CHECK_EQ(s.size(), live.size() + 1);
}


TEST(lookup_stats_is_zero_before_the_first_call_for_a_field) {
    Model m;
    make_account(m, "A1");
    const LookupCounts c = m.lookup_stats<&Order::qty>();
    CHECK_EQ(c.cached_calls, std::uint64_t{0});
    CHECK_EQ(c.uncached_calls, std::uint64_t{0});
    CHECK(m.lookup_diagnostics().stats.empty());
}

// find_by_cached_field/find_by_scan_field share one field (Order::qty is
// declared in BOTH define_cached_fields() and define_scan_fields()), so
// this exercises the counts staying INDEPENDENT per family on the same
// field, not just per field.
TEST(lookup_stats_counts_cached_and_uncached_calls_independently) {
    Model m;
    const Ref<Account> a = make_account(m, "A1");
    make_order(m, "O1", a, Opt<Order>{}, 5);
    Snapshot s = m.snapshot();

    (void)s.find_by_cached_field<&Order::qty>(5);
    (void)s.find_by_cached_field<&Order::qty>(5);
    (void)s.find_by_cached_field<&Order::qty>(7);  // no match -- still a CALL
    (void)s.find_by_scan_field<&Order::qty>(5);

    const LookupCounts c = m.lookup_stats<&Order::qty>();
    CHECK_EQ(c.cached_calls, std::uint64_t{3});
    CHECK_EQ(c.uncached_calls, std::uint64_t{1});
}

// for_each_by_scan_field/for_each_by_cached_field are find_by_scan_field/
// find_by_cached_field's no-vector siblings -- same matches, delivered via
// callback instead of a returned std::vector.
TEST(for_each_by_scan_field_and_for_each_by_cached_field_visit_every_match) {
    Model m;
    const Ref<Account> a = make_account(m, "A1");
    make_order(m, "O1", a, Opt<Order>{}, 5);
    make_order(m, "O2", a, Opt<Order>{}, 5);
    make_order(m, "O3", a, Opt<Order>{}, 7);
    Snapshot s = m.snapshot();

    int scan_hits = 0;
    s.for_each_by_scan_field<&Order::qty>(5, [&](const Order& o) {
        CHECK_EQ(o.qty, std::int64_t{5});
        ++scan_hits;
    });
    CHECK_EQ(scan_hits, 2);

    int cached_hits = 0;
    s.for_each_by_cached_field<&Order::qty>(5, [&](const Order& o) {
        CHECK_EQ(o.qty, std::int64_t{5});
        ++cached_hits;
    });
    CHECK_EQ(cached_hits, 2);

    int no_match_hits = 0;
    s.for_each_by_scan_field<&Order::qty>(99, [&](const Order&) { ++no_match_hits; });
    CHECK_EQ(no_match_hits, 0);
}

// all_of_by_scan_field/all_of_by_cached_field: std::all_of's boolean
// contract (vacuously true on no matches, false as soon as one match fails
// pred) -- see the declarations' own comment for why pred simply stops
// being CALLED after the first failure rather than the underlying scan
// stopping early.
TEST(all_of_by_scan_field_and_all_of_by_cached_field_match_std_all_of_semantics) {
    Model m;
    const Ref<Account> a = make_account(m, "A1");
    make_order(m, "O1", a, Opt<Order>{}, 5);
    make_order(m, "O2", a, Opt<Order>{}, 5);
    make_order(m, "O3", a, Opt<Order>{}, 7);
    Snapshot s = m.snapshot();

    CHECK(s.all_of_by_scan_field<&Order::qty>(99, [](const Order&) { return false; }));  // vacuous
    CHECK(s.all_of_by_scan_field<&Order::qty>(5, [](const Order& o) { return o.qty == 5; }));
    CHECK(s.all_of_by_cached_field<&Order::qty>(5, [](const Order& o) { return o.qty == 5; }));

    int calls = 0;
    const bool result = s.all_of_by_cached_field<&Order::qty>(5, [&](const Order&) {
        ++calls;
        return false;  // fails on the FIRST match -- pred should not run again
    });
    CHECK(!result);
    CHECK_EQ(calls, 1);
}

// View-returning forms of the same two functions.
TEST(for_each_view_by_scan_field_and_all_of_view_by_cached_field_bind_to_this_snapshot) {
    Model m;
    const Ref<Account> a = make_account(m, "A1");
    make_order(m, "O1", a, Opt<Order>{}, 5);
    make_order(m, "O2", a, Opt<Order>{}, 5);
    Snapshot s = m.snapshot();

    int hits = 0;
    s.for_each_view_by_scan_field<&Order::qty>(5, [&](View<Order> v) {
        CHECK_EQ(v->qty, std::int64_t{5});
        ++hits;
    });
    CHECK_EQ(hits, 2);

    CHECK(s.all_of_view_by_cached_field<&Order::qty>(5, [](View<Order> v) { return v->qty == 5; }));
    CHECK(!s.all_of_view_by_cached_field<&Order::qty>(5, [](View<Order> v) { return v->qty != 5; }));
    CHECK(s.all_of_view_by_scan_field<&Order::qty>(99, [](View<Order>) { return false; }));  // vacuous
}

// A single transaction that both retires an old match (update qty away from
// 5) and installs a fresh one (create with qty 5) at once -- find_by_scan_field
// and find_by_cached_field must land on the exact same surviving pair, not
// just agree on count, proving the cached index's erase-on-reassign and
// insert-on-create both take effect within one commit rather than one
// lagging the other.
TEST(scan_and_cached_field_agree_after_a_mutate_and_a_create_in_one_transaction) {
    Model m;
    const Ref<Account> a = make_account(m, "A1");
    const Ref<Order> o1 = make_order(m, "O1", a, Opt<Order>{}, 5);
    const Ref<Order> o2 = make_order(m, "O2", a, Opt<Order>{}, 5);

    Snapshot s0 = m.snapshot();
    CHECK_EQ(s0.find_by_scan_field<&Order::qty>(5).size(), std::size_t{2});
    CHECK_EQ(s0.find_by_cached_field<&Order::qty>(5).size(), std::size_t{2});
    CHECK(s0.find(o1) != nullptr);  // baseline: o1's Id resolves, qty still 5
    CHECK_EQ(s0.find(o1)->qty, std::int64_t{5});

    Transaction txn = m.begin();
    txn.update(o1)->qty = 9;  // o1 drops out of the qty==5 match set
    auto o3 = std::make_unique<Order>();
    o3->code = "O3";
    o3->account = a;
    o3->qty = 5;  // o3 joins the qty==5 match set
    const Ref<Order> local3 = txn.create(std::move(o3));
    const CommitResult res = commit_ok(m, txn);
    const Ref<Order> o3_real = res.to_real(local3);

    Snapshot s1 = m.snapshot();

    // o1's own Id still resolves in s1 -- proving update() mutated the SAME
    // logical object rather than deleting it and something else (e.g. o3)
    // landing in a recycled slot. A remove+recreate would bump Id::gen and
    // make o1.raw() a stale Id that find() rejects; see invariant 5.
    const Order* o1_in_s1 = s1.find(o1);
    CHECK(o1_in_s1 != nullptr);
    CHECK_EQ(o1_in_s1->qty, std::int64_t{9});

    const auto scan = s1.find_by_scan_field<&Order::qty>(5);
    const auto cached = s1.find_by_cached_field<&Order::qty>(5);
    CHECK_EQ(scan.size(), std::size_t{2});
    CHECK_EQ(cached.size(), std::size_t{2});

    for (const auto& matches : {scan, cached}) {
        std::vector<Id> ids;
        for (const Order* o : matches) ids.push_back(o->id);
        CHECK(std::find(ids.begin(), ids.end(), o2.raw()) != ids.end());
        CHECK(std::find(ids.begin(), ids.end(), o3_real.raw()) != ids.end());
        CHECK(std::find(ids.begin(), ids.end(), o1.raw()) == ids.end());
    }
}

// find_referrers<Field> is a thin wrapper around for_each_referrer<Field> --
// the count must land once per CALL to the public API, not once per object
// for_each_referrer happens to visit internally.
TEST(lookup_stats_counts_find_referrers_once_not_once_per_visited_object) {
    Model m;
    const Ref<Account> a = make_account(m, "A1");
    make_order(m, "O1", a);
    make_order(m, "O2", a);
    make_order(m, "O3", a);
    Snapshot s = m.snapshot();

    const auto found = s.find_referrers<&Order::account>(a);
    CHECK_EQ(found.size(), std::size_t{3});  // visited three objects...
    const LookupCounts c = m.lookup_stats<&Order::account>();
    CHECK_EQ(c.uncached_calls, std::uint64_t{1});  // ...but that is ONE call

    (void)s.find_cached_referrers<&Order::account>(a);
    CHECK_EQ(m.lookup_stats<&Order::account>().cached_calls, std::uint64_t{1});
}

// Two DISTINCT Models must not share counts, even for the identical field --
// each Model's field_lookup_counts_ is its own map, keyed within that Model
// only.
TEST(lookup_stats_is_independent_per_model) {
    Model m1;
    Model m2;
    make_account(m1, "A1");
    make_account(m2, "A1");
    Snapshot s1 = m1.snapshot();
    Snapshot s2 = m2.snapshot();

    (void)s1.find_by_scan_field<&Account::name>("A1");
    (void)s1.find_by_scan_field<&Account::name>("A1");

    CHECK_EQ(m1.lookup_stats<&Account::name>().uncached_calls, std::uint64_t{2});
    CHECK_EQ(m2.lookup_stats<&Account::name>().uncached_calls, std::uint64_t{0});
}

// The property the atomic-counter redesign exists for: counts recorded from
// MANY threads, concurrently, are all visible from whichever thread later
// calls lookup_stats() -- no per-thread staging, no explicit flush call
// anywhere in this test.
TEST(lookup_stats_aggregates_concurrent_calls_from_every_thread_with_no_flush) {
    Model m;
    const Ref<Account> a = make_account(m, "A1");
    make_order(m, "O1", a, Opt<Order>{}, 5);

    constexpr int kThreads = 8;
    constexpr int kCallsPerThread = 500;
    std::vector<std::thread> threads;
    for (int t = 0; t < kThreads; ++t) {
        threads.emplace_back([&m] {
            Snapshot s = m.snapshot();
            for (int i = 0; i < kCallsPerThread; ++i) (void)s.find_by_cached_field<&Order::qty>(5);
        });
    }
    for (auto& th : threads) th.join();

    CHECK_EQ(m.lookup_stats<&Order::qty>().cached_calls,
             static_cast<std::uint64_t>(kThreads) * kCallsPerThread);
}

TEST(lookup_diagnostics_is_labeled_by_declaring_type) {
    Model m;
    const Ref<Account> a = make_account(m, "A1");
    make_order(m, "O1", a, Opt<Order>{}, 5);
    Snapshot s = m.snapshot();
    (void)s.find_by_cached_field<&Order::qty>(5);
    (void)s.find_by_scan_field<&Order::qty>(9);

    const Model::LookupDiagnostics diag = m.lookup_diagnostics();
    CHECK_EQ(diag.stats.size(), std::size_t{1});
    // stats stores a raw type_info pointer, not a demangled name (that only
    // happens in to_string(), see its own test below) -- so the precise,
    // non-string way to check "labeled by declaring type" is comparing
    // type_info directly.
    const auto it = diag.stats.begin();
    CHECK(*it->first.type == typeid(Order));
    CHECK_EQ(it->second.cached_calls, std::uint64_t{1});
    CHECK_EQ(it->second.uncached_calls, std::uint64_t{1});
}

TEST(lookup_diagnostics_to_string_is_human_readable) {
    Model m;
    const Ref<Account> a = make_account(m, "A1");
    make_order(m, "O1", a, Opt<Order>{}, 5);
    Snapshot s = m.snapshot();

    CHECK(m.lookup_diagnostics().to_string().find("none") !=
          std::string::npos);  // nothing called yet

    (void)s.find_by_cached_field<&Order::qty>(5);
    (void)s.find_by_cached_field<&Order::qty>(5);
    (void)s.find_by_scan_field<&Order::qty>(5);

    const std::string report = m.lookup_diagnostics().to_string();
    // Order::qty is named ("qty") in test_types.h's define_cached_fields()/
    // define_scan_fields() -- registered the moment the Order above was
    // created, well before either lookup call -- so the report shows the
    // real name instead of falling back to a bare address.
    CHECK(report.find("Order::qty") != std::string::npos);
    CHECK(report.find("@") == std::string::npos);  // no address fallback needed

    std::printf("LookupDiagnostics: %s\n", report.c_str());

    // Numbers are column-aligned (right-justified, padded with spaces) for
    // table readability, so check label-then-eventually-digit rather than
    // pinning the exact width.
    auto value_after = [&](const char* label) -> std::string {
        const auto pos = report.find(label);
        if (pos == std::string::npos) return {};
        std::size_t i = pos + std::strlen(label);
        while (i < report.size() && report[i] == ' ') ++i;
        const std::size_t start = i;
        while (i < report.size() && std::isdigit(static_cast<unsigned char>(report[i]))) ++i;
        return report.substr(start, i - start);
    };
    CHECK_EQ(value_after("cached="), std::string("2"));
    CHECK_EQ(value_after("uncached="), std::string("1"));
}

namespace {
/// A cached field declared WITHOUT a name (the third argument to .key<>()
/// left at its default nullptr) -- isolates the fallback path: naming is
/// opt-in, so a field nobody named must still work, just less legibly.
class Unnamed final : public model::Object<Unnamed> {
public:
    std::int64_t code = 0;
    template <class Self>
    static void define_cached_fields(Self& s, const model::FieldKeyReader& v) {
        v.key<&Unnamed::code>(s.code);  // no name argument
    }
};
}  // namespace

TEST(lookup_diagnostics_falls_back_to_an_address_for_an_unnamed_field) {
    Model m;
    Transaction txn = m.begin();
    auto u = std::make_unique<Unnamed>();
    u->code = 7;
    txn.create(std::move(u));
    commit_ok(m, txn);

    Snapshot s = m.snapshot();
    (void)s.find_by_cached_field<&Unnamed::code>(7);

    const std::string report = m.lookup_diagnostics().to_string();
    CHECK(report.find("Unnamed") != std::string::npos);
    CHECK(report.find("@") != std::string::npos);  // no name registered -- address shown instead
    CHECK(report.find("Unnamed::") == std::string::npos);  // no name to append after "::"
}

// Same named/unnamed split as the two tests above, but through the
// REFERENCE-field entry point (RefIndexReader::index(), feeding
// find_cached_referrers/for_each_referrer) rather than the value-field one
// (FieldKeyReader::key(), feeding find_by_cached_field/find_by_scan_field) --
// a genuinely different code path in record_field_lookup's callers, and one
// with a real unnamed field sitting in test_types.h already: Order::account
// is named ("account") in define_cached_references(), but Order::parent is
// declared in define_references() (so cascade/null still work) and
// deliberately left OUT of define_cached_references() entirely -- see its
// own comment -- so it never reaches RefIndexReader::index() and has no
// registered name. Both fields live on the SAME type, which is exactly the
// disambiguation case FieldStat::field exists for.
TEST(lookup_diagnostics_shows_names_for_reference_fields_too) {
    Model m;
    const Ref<Account> a = make_account(m, "A1");
    const Ref<Order> parent_order = make_order(m, "PARENT", a);
    make_order(m, "CHILD", a, parent_order);
    Snapshot s = m.snapshot();

    (void)s.find_cached_referrers<&Order::account>(a);
    s.for_each_referrer<&Order::parent>(parent_order, [](const Order&) {});

    const std::string report = m.lookup_diagnostics().to_string();
    CHECK(report.find("Order::account") != std::string::npos);   // named
    CHECK(report.find("Order::parent") == std::string::npos);    // not named
    CHECK(report.find("Order @") != std::string::npos);          // falls back to an address instead
}

