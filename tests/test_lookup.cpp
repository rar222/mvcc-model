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


// field_tag<Field> mints its identity from the (type, VALUE) pair of the
// non-type template argument, not from the pointer-to-member's type alone.
// Two distinct std::int64_t fields on the same class share the exact same
// pointer-to-member TYPE (std::int64_t TwoFieldsSameType::*) but are
// different VALUES of that type, so they must still resolve to two
// different, stable addresses -- otherwise every same-typed field on a
// class would collide in by_field_/by_type_ and one would silently shadow
// the other.
TEST(field_tag_is_keyed_by_value_not_just_pointer_to_member_type) {
    struct TwoFieldsSameType {
        std::int64_t a = 0;
        std::int64_t b = 0;
    };
    const void* tag_a = model::field_tag<&TwoFieldsSameType::a>();
    const void* tag_b = model::field_tag<&TwoFieldsSameType::b>();
    CHECK(tag_a != tag_b);
    // Stable: the same Field value, instantiated again (as if from a second
    // call site), resolves to the identical address rather than a fresh one.
    CHECK_EQ(tag_a, model::field_tag<&TwoFieldsSameType::a>());
    CHECK_EQ(tag_b, model::field_tag<&TwoFieldsSameType::b>());
}

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

// range<T>/range_view<T>: additive range-based-for form of for_each<T> --
// same type-filtered walk as for_each_is_type_filtered above, and `break`
// short-circuits without visiting the rest of the population.
TEST(range_and_range_view_visit_exactly_the_objects_for_each_does) {
    Model m;
    const Ref<Account> a1 = make_account(m, "A1");
    const Ref<Account> a2 = make_account(m, "A2");
    make_order(m, "O1", a1);
    make_order(m, "O2", a2);
    make_order(m, "O3", a1);
    Snapshot s = m.snapshot();

    int orders = 0, accounts = 0;
    for (const Order& o : s.range<Order>()) {
        CHECK(!o.code.empty());
        ++orders;
    }
    for (const Account& a : s.range<Account>()) {
        CHECK(!a.name.empty());
        ++accounts;
    }
    CHECK_EQ(orders, 3);
    CHECK_EQ(accounts, 2);

    int view_orders = 0;
    for (View<Order> v : s.range_view<Order>()) {
        CHECK(!v->code.empty());
        ++view_orders;
    }
    CHECK_EQ(view_orders, 3);

    int seen_before_break = 0;
    for (const Order& o : s.range<Order>()) {
        (void)o;
        ++seen_before_break;
        break;
    }
    CHECK_EQ(seen_before_break, 1);

    // A type with no live objects: empty range, not a crash.
    Model empty_m;
    Snapshot empty_s = empty_m.snapshot();
    int none = 0;
    for (const Order& o : empty_s.range<Order>()) {
        (void)o;
        ++none;
    }
    CHECK_EQ(none, 0);
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

// find_by_predicate()/view_by_predicate() run a predicate scan restricted to
// one type, in both the raw-pointer and the View-returning forms.
TEST(find_by_predicate_runs_an_arbitrary_predicate_over_one_type) {
    Model m;
    make_account(m, "A1", 50);
    make_account(m, "A2", 150);
    make_account(m, "A3", 250);

    Snapshot s = m.snapshot();
    const auto rich = s.find_by_predicate<Account>([](const Account& a) { return a.balance >= 100; });
    CHECK_EQ(rich.size(), std::size_t{2});
    for (const Account* a : rich) CHECK(a->balance >= 100);

    const auto rich_views =
        s.view_by_predicate<Account>([](const Account& a) { return a.balance >= 100; });
    CHECK_EQ(rich_views.size(), std::size_t{2});
    for (const View<Account>& v : rich_views) CHECK(v->balance >= 100);

    const auto none = s.find_by_predicate<Account>([](const Account&) { return false; });
    CHECK(none.empty());
}

// The unindexed reverse-lookup family (for_each_referrers/find_referrers
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
    va1->for_each_referrers<&Order::account>([&](View<Order> v) {
        CHECK(v->account == a1);
        ++seen;
    });
    CHECK_EQ(seen, 2);

    const auto view_children = s.view(*s.find(p)).find_referrers<&Order::parent>();
    CHECK_EQ(view_children.size(), std::size_t{2});
}

// all_of_referrers/all_of_view_referrers -- previously missing entirely (only
// the scan/cached-field families had an all_of_* sibling; the referrer
// family didn't). Both are built on the same referrer_short_circuit
// for_each_referrers itself now delegates to, so this also checks the walk
// genuinely stops early rather than just skipping further predicate calls.
TEST(all_of_referrers_and_all_of_view_referrers_short_circuit) {
    Model m;
    const Ref<Account> a = make_account(m, "A1");
    make_order(m, "O1", a, Opt<Order>{}, /*qty=*/1);
    make_order(m, "O2", a, Opt<Order>{}, /*qty=*/2);
    Snapshot s = m.snapshot();

    CHECK(s.all_of_referrers<&Order::account>(a, [](const Order& o) { return o.qty > 0; }));
    CHECK(!s.all_of_referrers<&Order::account>(a, [](const Order& o) { return o.qty > 1; }));

    int checked = 0;
    const bool stopped = s.all_of_referrers<&Order::account>(a, [&](const Order&) {
        ++checked;
        return false;  // fail on the very first referrer visited
    });
    CHECK(!stopped);
    CHECK_EQ(checked, 1);  // never reached the second

    // Vacuously true: an account with no orders violates nothing.
    CHECK(s.all_of_referrers<&Order::account>(make_account(m, "LONELY"), [](const Order&) { return false; }));

    // View form: same short-circuit contract, through a View<Order> instead
    // -- including that it genuinely stops (not just a different boolean).
    CHECK(s.all_of_view_referrers<&Order::account>(a, [](View<Order> v) { return v->qty > 0; }));
    CHECK(!s.all_of_view_referrers<&Order::account>(a, [](View<Order> v) { return v->qty > 1; }));

    int checked_view = 0;
    const bool stopped_view = s.all_of_view_referrers<&Order::account>(a, [&](View<Order>) {
        ++checked_view;
        return false;
    });
    CHECK(!stopped_view);
    CHECK_EQ(checked_view, 1);

    CHECK(s.all_of_view_referrers<&Order::account>(make_account(m, "LONELY2"),
                                                  [](View<Order>) { return false; }));
}

// The View-returning referrer API's "no referrers" case -- for_each_view_
// referrer/view_referrers must visit/return nothing, not crash or return a
// stale/garbage view, for a target that genuinely has none.
TEST(view_referrers_forms_are_empty_for_an_unreferenced_target) {
    Model m;
    const Ref<Account> lonely = make_account(m, "LONELY");
    Snapshot s = m.snapshot();

    int seen = 0;
    s.for_each_view_referrers<&Order::account>(lonely, [&](View<Order>) { ++seen; });
    CHECK_EQ(seen, 0);

    CHECK(s.view_referrers<&Order::account>(lonely).empty());
}

// range_referrers/range_view_referrers: additive range-based-for form of
// find_referrers/for_each_referrers/all_of_referrers -- works on ANY Ref<>/
// Opt<> field, declared cached (account) or not (parent), same as the
// callback forms; must visit the exact same matches, and `break` must
// short-circuit the underlying walk.
TEST(range_referrers_visits_the_same_matches_as_find_referrers) {
    Model m;
    const Ref<Account> a1 = make_account(m, "A1");
    const Ref<Account> a2 = make_account(m, "A2");
    const Ref<Order> p = make_order(m, "P", a1);
    make_order(m, "C1", a1, p);
    make_order(m, "C2", a2, p);
    make_order(m, "OTHER", a2);
    Snapshot s = m.snapshot();

    int a1_hits = 0;
    for (const Order& o : s.range_referrers<&Order::account>(a1)) {
        CHECK(o.account == a1);
        ++a1_hits;
    }
    CHECK_EQ(a1_hits, 2);

    int child_hits = 0;
    for (const Order& o : s.range_referrers<&Order::parent>(p)) {
        CHECK(o.parent == p);
        ++child_hits;
    }
    CHECK_EQ(child_hits, 2);

    int view_hits = 0;
    for (View<Order> v : s.range_view_referrers<&Order::account>(a1)) {
        CHECK(v->account == a1);
        ++view_hits;
    }
    CHECK_EQ(view_hits, 2);

    int seen_before_break = 0;
    for (const Order& o : s.range_referrers<&Order::account>(a1)) {
        (void)o;
        ++seen_before_break;
        break;
    }
    CHECK_EQ(seen_before_break, 1);

    const Ref<Account> lonely = make_account(m, "LONELY");
    s = m.snapshot();
    int none = 0;
    for (const Order& o : s.range_referrers<&Order::account>(lonely)) {
        (void)o;
        ++none;
    }
    CHECK_EQ(none, 0);
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

// find_by_field is cache-first, scan-fallback -- qty is declared cached, its
// scan-only twin qty_scan is not (see Order's own comment in test_types.h),
// so calling find_by_field on each exercises the two branches independently
// and each field's OWN lookup_stats stay separate from the other's.
TEST(lookup_stats_counts_cache_hits_and_scan_fallbacks_independently) {
    Model m;
    const Ref<Account> a = make_account(m, "A1");
    make_order(m, "O1", a, Opt<Order>{}, 5);  // sets qty_scan == qty == 5 too
    Snapshot s = m.snapshot();

    (void)s.find_by_field<&Order::qty>(5);
    (void)s.find_by_field<&Order::qty>(5);
    (void)s.find_by_field<&Order::qty>(7);  // no match -- still a CALL
    (void)s.find_by_field<&Order::qty_scan>(5);

    const LookupCounts qty_c = m.lookup_stats<&Order::qty>();
    CHECK_EQ(qty_c.cached_calls, std::uint64_t{3});
    CHECK_EQ(qty_c.uncached_calls, std::uint64_t{0});
    const LookupCounts qty_scan_c = m.lookup_stats<&Order::qty_scan>();
    CHECK_EQ(qty_scan_c.cached_calls, std::uint64_t{0});
    CHECK_EQ(qty_scan_c.uncached_calls, std::uint64_t{1});
}

// for_each_by_field is find_by_field's no-vector sibling -- same matches,
// delivered via callback instead of a returned std::vector. Exercised on
// both qty (cache-hit branch) and its scan-only twin qty_scan (scan-
// fallback branch), on identical data, to prove both branches visit every
// match correctly.
TEST(for_each_by_field_visits_every_match_via_either_branch) {
    Model m;
    const Ref<Account> a = make_account(m, "A1");
    make_order(m, "O1", a, Opt<Order>{}, 5);
    make_order(m, "O2", a, Opt<Order>{}, 5);
    make_order(m, "O3", a, Opt<Order>{}, 7);
    Snapshot s = m.snapshot();

    int scan_hits = 0;
    s.for_each_by_field<&Order::qty_scan>(5, [&](const Order& o) {
        CHECK_EQ(o.qty_scan, std::int64_t{5});
        ++scan_hits;
    });
    CHECK_EQ(scan_hits, 2);

    int cached_hits = 0;
    s.for_each_by_field<&Order::qty>(5, [&](const Order& o) {
        CHECK_EQ(o.qty, std::int64_t{5});
        ++cached_hits;
    });
    CHECK_EQ(cached_hits, 2);

    int no_match_hits = 0;
    s.for_each_by_field<&Order::qty_scan>(99, [&](const Order&) { ++no_match_hits; });
    CHECK_EQ(no_match_hits, 0);
}

// all_of_by_field: std::all_of's boolean contract (vacuously true on no
// matches, false as soon as one match fails pred) -- see the declaration's
// own comment for why pred simply stops being CALLED after the first
// failure rather than the underlying walk stopping early. Exercised on both
// the scan-fallback branch (qty_scan) and the cache-hit branch (qty).
TEST(all_of_by_field_matches_std_all_of_semantics_via_either_branch) {
    Model m;
    const Ref<Account> a = make_account(m, "A1");
    make_order(m, "O1", a, Opt<Order>{}, 5);
    make_order(m, "O2", a, Opt<Order>{}, 5);
    make_order(m, "O3", a, Opt<Order>{}, 7);
    Snapshot s = m.snapshot();

    CHECK(s.all_of_by_field<&Order::qty_scan>(99, [](const Order&) { return false; }));  // vacuous
    CHECK(s.all_of_by_field<&Order::qty_scan>(5, [](const Order& o) { return o.qty_scan == 5; }));
    CHECK(s.all_of_by_field<&Order::qty>(5, [](const Order& o) { return o.qty == 5; }));

    int calls = 0;
    const bool result = s.all_of_by_field<&Order::qty>(5, [&](const Order&) {
        ++calls;
        return false;  // fails on the FIRST match -- pred should not run again
    });
    CHECK(!result);
    CHECK_EQ(calls, 1);
}

// range_by_field/range_view_by_field: additive range-based-for form of
// find_by_field/for_each_by_field/all_of_by_field -- must visit the exact
// same matches, `break` must short-circuit, and a nullary-const-method
// Field (computed_key) must work the same way a data member Field (qty)
// does. Primary walk uses qty_scan to exercise the scan-fallback branch
// (where FieldRange actually filters per element); computed_key is
// cache-declared, exercising the cache-hit branch (no filtering) instead.
TEST(range_by_field_visits_the_same_matches_as_for_each_by_field) {
    Model m;
    const Ref<Account> a = make_account(m, "A1");
    const Ref<Order> o1 = make_order(m, "O1", a, Opt<Order>{}, 5);
    make_order(m, "O2", a, Opt<Order>{}, 5);
    make_order(m, "O3", a, Opt<Order>{}, 7);
    Snapshot s = m.snapshot();

    int range_hits = 0;
    for (const Order& o : s.range_by_field<&Order::qty_scan>(5)) {
        CHECK_EQ(o.qty_scan, std::int64_t{5});
        ++range_hits;
    }
    CHECK_EQ(range_hits, 2);

    int no_match_hits = 0;
    for (const Order& o : s.range_by_field<&Order::qty_scan>(99)) {
        (void)o;
        ++no_match_hits;
    }
    CHECK_EQ(no_match_hits, 0);

    int seen_before_break = 0;
    for (const Order& o : s.range_by_field<&Order::qty_scan>(5)) {
        (void)o;
        ++seen_before_break;
        break;
    }
    CHECK_EQ(seen_before_break, 1);

    int view_hits = 0;
    for (View<Order> v : s.range_view_by_field<&Order::qty_scan>(5)) {
        CHECK_EQ(v->qty_scan, std::int64_t{5});
        ++view_hits;
    }
    CHECK_EQ(view_hits, 2);

    // Field naming a nullary const method (computed_key), not a data member
    // -- exercises FieldRange::matches' other branch
    // (std::is_member_object_pointer_v == false), reached here via the
    // cache-hit path since computed_key is tagged LookupType::Cache.
    int computed_hits = 0;
    for (const Order& o : s.range_by_field<&Order::computed_key>("ord:O1")) {
        CHECK(o.id == o1.raw());
        ++computed_hits;
    }
    CHECK_EQ(computed_hits, 1);
}

// A field that exists on the type but was never declared in EITHER lookup
// family must behave exactly like find_by_field's own "undeclared is
// invisible" rule -- vacuously empty, even for a value (0) that genuinely
// matches some live object's actual field (a freshly made_account's balance
// defaults to 0). Proves range_by_field's declared-check runs, not just the
// value filter.
TEST(range_by_field_is_empty_for_a_field_never_declared_in_either_family) {
    Model m;
    make_account(m, "A1");  // balance defaults to 0 -- would "match" if unfiltered
    Snapshot s = m.snapshot();

    // Sanity: find_by_field itself is already empty here -- this test is
    // about range_by_field matching that, not establishing it fresh.
    CHECK(s.find_by_field<&Account::balance>(0).empty());

    int hits = 0;
    for (const Account& a : s.range_by_field<&Account::balance>(0)) {
        (void)a;
        ++hits;
    }
    CHECK_EQ(hits, 0);
    CHECK_EQ(s.find_by_field<&Account::balance>(0).size(), std::size_t{0});
}

// range_by_field's cache-hit branch: additive range-based-for form of
// find_by_field/for_each_by_field/all_of_by_field on a cache-declared field
// -- must visit the exact same matches as for_each_by_field, and `break`
// must short-circuit (stop the underlying bucket walk, not just skip
// further loop bodies) the same way all_of_by_field's pred returning false
// does.
TEST(range_by_field_visits_the_same_matches_as_for_each_by_field_via_cache) {
    Model m;
    const Ref<Account> a = make_account(m, "A1");
    make_order(m, "O1", a, Opt<Order>{}, 5);
    make_order(m, "O2", a, Opt<Order>{}, 5);
    make_order(m, "O3", a, Opt<Order>{}, 7);
    Snapshot s = m.snapshot();

    int range_hits = 0;
    for (const Order& o : s.range_by_field<&Order::qty>(5)) {
        CHECK_EQ(o.qty, std::int64_t{5});
        ++range_hits;
    }
    CHECK_EQ(range_hits, 2);

    // No matches for this value: empty range, no special-casing at the call
    // site -- same "undeclared/no-match is empty" contract as find_by_field.
    int no_match_hits = 0;
    for (const Order& o : s.range_by_field<&Order::qty>(99)) {
        (void)o;
        ++no_match_hits;
    }
    CHECK_EQ(no_match_hits, 0);

    // break stops the walk itself, not just further loop bodies -- exactly
    // what all_of_by_field's early return false gives you.
    int seen_before_break = 0;
    for (const Order& o : s.range_by_field<&Order::qty>(5)) {
        (void)o;
        ++seen_before_break;
        break;
    }
    CHECK_EQ(seen_before_break, 1);
}

// range_view_by_field: same matches, View<Order> bound to `s`.
TEST(range_view_by_field_binds_views_to_this_snapshot) {
    Model m;
    const Ref<Account> a = make_account(m, "A1");
    make_order(m, "O1", a, Opt<Order>{}, 5);
    make_order(m, "O2", a, Opt<Order>{}, 5);
    Snapshot s = m.snapshot();

    int hits = 0;
    for (View<Order> v : s.range_view_by_field<&Order::qty>(5)) {
        CHECK_EQ(v->qty, std::int64_t{5});
        ++hits;
    }
    CHECK_EQ(hits, 2);
}

// range_referrers/range_view_referrers on the cache-hit branch: same bucket
// (Root::by_cached_reference) find_referrers/for_each_referrers already
// read, via range-based for instead of a callback.
TEST(range_referrers_visits_the_same_matches_as_find_referrers_via_cache) {
    Model m;
    const Ref<Account> a1 = make_account(m, "A1");
    const Ref<Account> a2 = make_account(m, "A2");
    make_order(m, "O1", a1);
    make_order(m, "O2", a1);
    make_order(m, "O3", a2);
    Snapshot s = m.snapshot();

    const auto expected = s.find_referrers<&Order::account>(a1);
    CHECK_EQ(expected.size(), std::size_t{2});

    int hits = 0;
    for (const Order& o : s.range_referrers<&Order::account>(a1)) {
        CHECK(o.account == a1);
        ++hits;
    }
    CHECK_EQ(hits, static_cast<int>(expected.size()));

    int view_hits = 0;
    for (View<Order> v : s.range_view_referrers<&Order::account>(a1)) {
        CHECK(v->account == a1);
        ++view_hits;
    }
    CHECK_EQ(view_hits, static_cast<int>(expected.size()));

    // An account referenced by nothing: empty range.
    const Ref<Account> lonely = make_account(m, "LONELY");
    s = m.snapshot();
    int lonely_hits = 0;
    for (const Order& o : s.range_referrers<&Order::account>(lonely)) {
        (void)o;
        ++lonely_hits;
    }
    CHECK_EQ(lonely_hits, 0);
}

// range_by_field/range_referrers must record a cache-hit lookup call the
// same way find_by_field/find_referrers do when the field/reference is
// declared cached -- see LookupCounts' own doc comment: it exists to let a
// caller decide, from ACTUAL usage, whether a cached field/reference is
// worth its write-side cost. A range-for call site that didn't count here
// would make that diagnostic silently undercount usage for anyone who
// switched to it.
TEST(range_by_field_and_range_referrers_record_lookup_stats) {
    Model m;
    const Ref<Account> a = make_account(m, "A1");
    make_order(m, "O1", a, Opt<Order>{}, 5);
    Snapshot s = m.snapshot();

    for (const Order& o : s.range_by_field<&Order::qty>(5)) (void)o;
    for (const Order& o : s.range_by_field<&Order::qty>(5)) (void)o;
    const LookupCounts qty_counts = m.lookup_stats<&Order::qty>();
    CHECK_EQ(qty_counts.cached_calls, std::uint64_t{2});

    for (const Order& o : s.range_referrers<&Order::account>(a)) (void)o;
    const LookupCounts account_counts = m.lookup_stats<&Order::account>();
    CHECK_EQ(account_counts.cached_calls, std::uint64_t{1});
}

// View-returning forms of for_each_by_field/all_of_by_field, one on each
// branch: for_each_view_by_field on the scan-fallback twin qty_scan,
// all_of_view_by_field on the cache-declared qty.
TEST(for_each_view_by_field_and_all_of_view_by_field_bind_to_this_snapshot) {
    Model m;
    const Ref<Account> a = make_account(m, "A1");
    make_order(m, "O1", a, Opt<Order>{}, 5);
    make_order(m, "O2", a, Opt<Order>{}, 5);
    Snapshot s = m.snapshot();

    int hits = 0;
    s.for_each_view_by_field<&Order::qty_scan>(5, [&](View<Order> v) {
        CHECK_EQ(v->qty_scan, std::int64_t{5});
        ++hits;
    });
    CHECK_EQ(hits, 2);

    CHECK(s.all_of_view_by_field<&Order::qty>(5, [](View<Order> v) { return v->qty == 5; }));
    CHECK(!s.all_of_view_by_field<&Order::qty>(5, [](View<Order> v) { return v->qty != 5; }));
    CHECK(s.all_of_view_by_field<&Order::qty_scan>(99, [](View<Order>) { return false; }));  // vacuous
}

// A single transaction that both retires an old match (update qty away from
// 5) and installs a fresh one (create with qty 5) at once -- find_by_field
// on the cache-declared qty and on its scan-only twin qty_scan must land on
// the exact same surviving pair, not just agree on count, proving the
// cached index's erase-on-reassign and insert-on-create both take effect
// within one commit rather than one lagging the other (and that the scan
// fallback, walking the live population directly, agrees with it).
TEST(cache_hit_and_scan_fallback_agree_after_a_mutate_and_a_create_in_one_transaction) {
    Model m;
    const Ref<Account> a = make_account(m, "A1");
    const Ref<Order> o1 = make_order(m, "O1", a, Opt<Order>{}, 5);  // qty_scan == 5 too
    const Ref<Order> o2 = make_order(m, "O2", a, Opt<Order>{}, 5);

    Snapshot s0 = m.snapshot();
    CHECK_EQ(s0.find_by_field<&Order::qty>(5).size(), std::size_t{2});
    CHECK_EQ(s0.find_by_field<&Order::qty_scan>(5).size(), std::size_t{2});
    CHECK(s0.find(o1) != nullptr);  // baseline: o1's Id resolves, qty still 5
    CHECK_EQ(s0.find(o1)->qty, std::int64_t{5});

    Transaction txn = m.begin();
    Order* u1 = txn.update(o1);
    u1->qty = 9;       // o1 drops out of the qty==5 match set
    u1->qty_scan = 9;  // ...and its scan-only twin, kept equal
    auto o3 = std::make_unique<Order>();
    o3->code = "O3";
    o3->account = a;
    o3->qty = 5;       // o3 joins the qty==5 match set
    o3->qty_scan = 5;
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

    const auto cache_hit = s1.find_by_field<&Order::qty>(5);
    const auto scan_fallback = s1.find_by_field<&Order::qty_scan>(5);
    CHECK_EQ(cache_hit.size(), std::size_t{2});
    CHECK_EQ(scan_fallback.size(), std::size_t{2});

    std::vector<Id> cache_hit_ids, scan_fallback_ids;
    for (const Order* o : cache_hit) cache_hit_ids.push_back(o->id);
    for (const Order* o : scan_fallback) scan_fallback_ids.push_back(o->id);
    for (std::vector<Id>* ids : {&cache_hit_ids, &scan_fallback_ids}) {
        CHECK(std::find(ids->begin(), ids->end(), o2.raw()) != ids->end());
        CHECK(std::find(ids->begin(), ids->end(), o3_real.raw()) != ids->end());
        CHECK(std::find(ids->begin(), ids->end(), o1.raw()) == ids->end());
    }
}

// find_referrers<Field> is a thin wrapper around for_each_referrers<Field> --
// the count must land once per CALL to the public API, not once per object
// for_each_referrers happens to visit internally. account is cache-declared
// (hits the index); account_scan is its scan-only twin (falls back), so the
// two calls below exercise each branch's own counter independently.
TEST(lookup_stats_counts_find_referrers_once_not_once_per_visited_object) {
    Model m;
    const Ref<Account> a = make_account(m, "A1");
    const Ref<Order> o1 = make_order(m, "O1", a);
    const Ref<Order> o2 = make_order(m, "O2", a);
    const Ref<Order> o3 = make_order(m, "O3", a);
    // account_scan isn't set by make_order (see its own doc comment in
    // test_helpers.cpp) -- set explicitly here since this test never
    // cascade-deletes `a`, so the double-processing hazard doesn't apply.
    for (Ref<Order> o : {o1, o2, o3}) update_field(m, o, [&](Order* p) { p->account_scan = a; });
    Snapshot s = m.snapshot();

    const auto found = s.find_referrers<&Order::account_scan>(a);
    CHECK_EQ(found.size(), std::size_t{3});  // visited three objects...
    const LookupCounts c = m.lookup_stats<&Order::account_scan>();
    CHECK_EQ(c.uncached_calls, std::uint64_t{1});  // ...but that is ONE call

    (void)s.find_referrers<&Order::account>(a);
    CHECK_EQ(m.lookup_stats<&Order::account>().cached_calls, std::uint64_t{1});
}

// Regression: for_each_view_referrers/view_referrers must ALSO record a
// lookup. A prior version of for_each_view_referrers (then still named
// for_each_referrer_view) re-walked for_each_view<ClassT> and re-checked the
// match itself instead of delegating to for_each_referrers<Field> -- so it
// never called register_field_lookup, and every lookup made through the
// View-returning referrer API was silently invisible to lookup_stats().
// Delegating (see for_each_view_referrers's own doc comment) fixes that.
// Uses account_scan (View<T>'s referrer accessors forward straight to
// Snapshot's merged for_each_referrers/find_referrers, so a cache-declared
// field would hit the index branch instead of the one this test is about).
TEST(lookup_stats_counts_calls_made_through_the_view_returning_referrer_api_too) {
    Model m;
    const Ref<Account> a = make_account(m, "A1");
    const Ref<Order> o1 = make_order(m, "O1", a);
    const Ref<Order> o2 = make_order(m, "O2", a);
    // account_scan isn't set by make_order -- see test_helpers.cpp's own
    // doc comment; set explicitly since this test never cascade-deletes `a`.
    for (Ref<Order> o : {o1, o2}) update_field(m, o, [&](Order* p) { p->account_scan = a; });
    Snapshot s = m.snapshot();

    auto va = s.view(a);
    int seen = 0;
    va->for_each_referrers<&Order::account_scan>([&](View<Order>) { ++seen; });
    CHECK_EQ(seen, 2);
    CHECK_EQ(m.lookup_stats<&Order::account_scan>().uncached_calls, std::uint64_t{1});

    const auto found = s.view(*s.find(a)).find_referrers<&Order::account_scan>();
    CHECK_EQ(found.size(), std::size_t{2});
    CHECK_EQ(m.lookup_stats<&Order::account_scan>().uncached_calls, std::uint64_t{2});
}

// register_field_lookup() fires from every for_each_*/all_of_* entry point:
// for_each_by_field/all_of_by_field and for_each_referrers/all_of_referrers.
// The regression above (for_each_view_referrers once silently skipped it via
// an independent re-walk) shows exactly this class of bug is real -- this
// test pins all four, once via the cache-hit branch (qty/account) and once
// via the scan-fallback branch (qty_scan/account_scan), including that a
// short-circuited all_of_* still records exactly one call even though it
// stops after the first match.
TEST(lookup_stats_records_a_call_from_every_for_each_and_all_of_entry_point_once_each) {
    Model m;
    const Ref<Account> a = make_account(m, "A1");
    make_order(m, "O1", a, Opt<Order>{}, 5);
    make_order(m, "O2", a, Opt<Order>{}, 5);
    Snapshot s = m.snapshot();

    const LookupCounts qty_scan0 = m.lookup_stats<&Order::qty_scan>();
    s.for_each_by_field<&Order::qty_scan>(5, [](const Order&) {});
    CHECK_EQ(m.lookup_stats<&Order::qty_scan>().uncached_calls, qty_scan0.uncached_calls + 1);

    s.all_of_by_field<&Order::qty_scan>(5, [](const Order&) { return true; });
    CHECK_EQ(m.lookup_stats<&Order::qty_scan>().uncached_calls, qty_scan0.uncached_calls + 2);

    const LookupCounts qty0 = m.lookup_stats<&Order::qty>();
    s.for_each_by_field<&Order::qty>(5, [](const Order&) {});
    CHECK_EQ(m.lookup_stats<&Order::qty>().cached_calls, qty0.cached_calls + 1);

    s.all_of_by_field<&Order::qty>(5, [](const Order&) { return true; });
    CHECK_EQ(m.lookup_stats<&Order::qty>().cached_calls, qty0.cached_calls + 2);

    const LookupCounts acct_scan0 = m.lookup_stats<&Order::account_scan>();
    s.for_each_referrers<&Order::account_scan>(a, [](const Order&) {});
    CHECK_EQ(m.lookup_stats<&Order::account_scan>().uncached_calls, acct_scan0.uncached_calls + 1);

    s.all_of_referrers<&Order::account_scan>(a, [](const Order&) { return true; });
    CHECK_EQ(m.lookup_stats<&Order::account_scan>().uncached_calls, acct_scan0.uncached_calls + 2);

    const LookupCounts acct0 = m.lookup_stats<&Order::account>();
    s.for_each_referrers<&Order::account>(a, [](const Order&) {});
    CHECK_EQ(m.lookup_stats<&Order::account>().cached_calls, acct0.cached_calls + 1);

    int visits = 0;
    const bool stopped = s.all_of_referrers<&Order::account>(a, [&](const Order&) {
        ++visits;
        return false;  // stop after the first match
    });
    CHECK(!stopped);
    CHECK_EQ(visits, 1);
    CHECK_EQ(m.lookup_stats<&Order::account>().cached_calls, acct0.cached_calls + 2);
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

    (void)s1.find_by_field<&Account::name>("A1");
    (void)s1.find_by_field<&Account::name>("A1");

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
            for (int i = 0; i < kCallsPerThread; ++i) (void)s.find_by_field<&Order::qty>(5);
        });
    }
    for (auto& th : threads) th.join();

    CHECK_EQ(m.lookup_stats<&Order::qty>().cached_calls,
             static_cast<std::uint64_t>(kThreads) * kCallsPerThread);
}

// qty (cache-declared) and qty_scan (its scan-only twin) each resolve
// deterministically via one branch, so a single field can no longer show
// both nonzero cached_calls AND uncached_calls the way one merged field
// used to under the old two-function API -- this now checks two entries,
// one per branch.
TEST(lookup_diagnostics_is_labeled_by_declaring_type) {
    Model m;
    const Ref<Account> a = make_account(m, "A1");
    make_order(m, "O1", a, Opt<Order>{}, 5);
    Snapshot s = m.snapshot();
    (void)s.find_by_field<&Order::qty>(5);
    (void)s.find_by_field<&Order::qty_scan>(9);

    const Model::LookupDiagnostics diag = m.lookup_diagnostics();
    CHECK_EQ(diag.stats.size(), std::size_t{2});
    // stats stores a raw type_info pointer, not a demangled name (that only
    // happens in to_string(), see its own test below) -- so the precise,
    // non-string way to check "labeled by declaring type" is comparing
    // type_info directly.
    int cached_entries = 0, uncached_entries = 0;
    for (const auto& [key, counts] : diag.stats) {
        CHECK(*key.type == typeid(Order));
        if (counts.cached_calls > 0) {
            CHECK_EQ(counts.cached_calls, std::uint64_t{1});
            CHECK_EQ(counts.uncached_calls, std::uint64_t{0});
            ++cached_entries;
        } else {
            CHECK_EQ(counts.uncached_calls, std::uint64_t{1});
            ++uncached_entries;
        }
    }
    CHECK_EQ(cached_entries, 1);
    CHECK_EQ(uncached_entries, 1);
}

TEST(lookup_diagnostics_to_string_is_human_readable) {
    Model m;
    const Ref<Account> a = make_account(m, "A1");
    make_order(m, "O1", a, Opt<Order>{}, 5);
    Snapshot s = m.snapshot();

    CHECK(m.lookup_diagnostics().to_string().find("none") !=
          std::string::npos);  // nothing called yet

    (void)s.find_by_field<&Order::qty>(5);
    (void)s.find_by_field<&Order::qty>(5);
    (void)s.find_by_field<&Order::qty_scan>(5);

    const std::string report = m.lookup_diagnostics().to_string();
    // Order::qty/qty_scan are named ("qty"/"qty_scan") in test_types.h's
    // define_fields() -- registered the moment the Order above was created,
    // well before either lookup call -- so the report shows the real names
    // instead of falling back to a bare address.
    CHECK(report.find("Order::qty") != std::string::npos);
    CHECK(report.find("Order::qty_scan") != std::string::npos);
    CHECK(report.find("@") == std::string::npos);  // no address fallback needed

    std::printf("LookupDiagnostics: %s\n", report.c_str());

    // Numbers are column-aligned (right-justified, padded with spaces) for
    // table readability, so check label-then-eventually-digit rather than
    // pinning the exact width. Rows are sorted by total calls descending
    // (see LookupDiagnostics::to_string()), so qty's row (2 calls) comes
    // before qty_scan's (1 call); "Order::qty " (trailing space) anchors
    // qty's own row specifically, since "Order::qty_scan" would otherwise
    // also match a bare "Order::qty" search.
    auto value_after = [&](const std::string& text, const char* label) -> std::string {
        const auto pos = text.find(label);
        if (pos == std::string::npos) return {};
        std::size_t i = pos + std::strlen(label);
        while (i < text.size() && text[i] == ' ') ++i;
        const std::size_t start = i;
        while (i < text.size() && std::isdigit(static_cast<unsigned char>(text[i]))) ++i;
        return text.substr(start, i - start);
    };
    const std::size_t qty_row_start = report.find("Order::qty ");
    const std::size_t qty_scan_row_start = report.find("Order::qty_scan");
    CHECK(qty_row_start != std::string::npos);
    CHECK(qty_scan_row_start != std::string::npos);
    CHECK(qty_row_start < qty_scan_row_start);  // higher total sorts first
    const std::string qty_row = report.substr(qty_row_start, qty_scan_row_start - qty_row_start);
    const std::string qty_scan_row = report.substr(qty_scan_row_start);
    CHECK_EQ(value_after(qty_row, "cached="), std::string("2"));
    CHECK_EQ(value_after(qty_row, "uncached="), std::string("0"));
    CHECK_EQ(value_after(qty_scan_row, "cached="), std::string("0"));
    CHECK_EQ(value_after(qty_scan_row, "uncached="), std::string("1"));
}

namespace {
/// A cached field declared WITHOUT a name (the third argument to .key<>()
/// left at its default nullptr) -- isolates the fallback path: naming is
/// opt-in, so a field nobody named must still work, just less legibly.
class Unnamed final : public model::Object<Unnamed> {
public:
    std::int64_t code = 0;
    template <class Self>
    static void define_fields(Self& s, const model::LookupFieldReader& v) {
        v.key<&Unnamed::code>(s.code, model::LookupType::Cache);  // no name argument
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
    (void)s.find_by_field<&Unnamed::code>(7);

    const std::string report = m.lookup_diagnostics().to_string();
    CHECK(report.find("Unnamed") != std::string::npos);
    CHECK(report.find("@") != std::string::npos);  // no name registered -- address shown instead
    CHECK(report.find("Unnamed::") == std::string::npos);  // no name to append after "::"
}

// Same named/unnamed split as the two tests above, but through the
// REFERENCE-field entry point (CachedRefReader, feeding find_referrers/
// for_each_referrers's cache-hit branch) rather than the value-field one
// (LookupFieldReader::key(), feeding find_by_field) -- a genuinely different
// code path in register_field_lookup's callers, and one with a real unnamed
// field sitting in test_types.h already: Order::account is named ("account")
// and tagged LookupType::Cache in define_references(), but Order::parent is
// also declared in define_references() (so cascade/null still work) and
// deliberately tagged LookupType::Scan instead -- see its own comment -- so
// it never reaches CachedRefReader and has no registered name. Both fields
// live on the SAME type, which is exactly the disambiguation case
// FieldStat::field exists for.
TEST(lookup_diagnostics_shows_names_for_reference_fields_too) {
    Model m;
    const Ref<Account> a = make_account(m, "A1");
    const Ref<Order> parent_order = make_order(m, "PARENT", a);
    make_order(m, "CHILD", a, parent_order);
    Snapshot s = m.snapshot();

    (void)s.find_referrers<&Order::account>(a);
    s.for_each_referrers<&Order::parent>(parent_order, [](const Order&) {});

    const std::string report = m.lookup_diagnostics().to_string();
    CHECK(report.find("Order::account") != std::string::npos);   // named
    CHECK(report.find("Order::parent") == std::string::npos);    // not named
    CHECK(report.find("Order @") != std::string::npos);          // falls back to an address instead
}

