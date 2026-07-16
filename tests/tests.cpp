// Dependency-free test harness. No network in some environments, so no gtest
// or Catch2 -- just a macro and a registry.
//
// Run:  ctest --preset default --output-on-failure
//   or: ./build/default/model_tests

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <functional>
#include <map>
#include <memory>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include "demo/types.h"
#include "model/model.h"

using namespace model;
using namespace demo;

// ---------------------------------------------------------------------------
// Harness
// ---------------------------------------------------------------------------

namespace {

struct TestCase {
    const char* name;
    std::function<void()> fn;
};

std::vector<TestCase>& registry() {
    static std::vector<TestCase> r;
    return r;
}

int g_failures = 0;
const char* g_current = "";

struct Registrar {
    Registrar(const char* name, std::function<void()> fn) {
        registry().push_back({name, std::move(fn)});
    }
};

}  // namespace

#define TEST(name)                            \
    static void name();                       \
    static Registrar reg_##name(#name, name); \
    static void name()

#define CHECK(cond)                                                       \
    do {                                                                  \
        if (!(cond)) {                                                    \
            std::printf("  FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
            ++g_failures;                                                 \
        }                                                                 \
    } while (0)

#define CHECK_EQ(a, b)                                                           \
    do {                                                                         \
        auto lhs__ = (a);                                                        \
        auto rhs__ = (b);                                                        \
        if (!(lhs__ == rhs__)) {                                                 \
            std::printf("  FAIL %s:%d: %s == %s\n", __FILE__, __LINE__, #a, #b); \
            ++g_failures;                                                        \
        }                                                                        \
    } while (0)

// ---------------------------------------------------------------------------
// Helpers
//
// A local id returned by Transaction::create() is only meaningful until
// try_commit() returns (see CommitResult::resolve's doc comment). These
// helpers each own a single commit and hand back the REAL, post-commit ref,
// so most tests never have to think about the local/real distinction at all
// -- only the tests in the "Transaction-local semantics" section below,
// which are specifically about that distinction, build multi-op
// Transactions by hand.
// ---------------------------------------------------------------------------

namespace {

CommitResult commit_ok(Model& m, Transaction& txn) {
    CommitResult r = m.try_commit(txn);
    CHECK(r.status == CommitStatus::Committed);
    return r;
}

Ref<Account> make_account(Model& m, const std::string& name, std::int64_t bal = 0) {
    Transaction txn = m.begin();
    auto a = std::make_unique<Account>();
    a->name = name;
    a->balance = bal;
    const Ref<Account> local = txn.create(std::move(a));
    return commit_ok(m, txn).resolve(local);
}

/// A second, unrelated type whose computed_key() has no fixed prefix -- unlike
/// Order, which happens to prefix "ord:" by its own convention. Used to prove
/// the *model* enforces per-type keyspaces, rather than relying on every user
/// type picking disjoint prefixes.
class Widget final : public model::Object<Widget> {
public:
    std::string key;
    std::string computed_key() const { return key; }
    // no outgoing references, so no define_references() to declare
    template <class Self>
    static void define_keys(Self& s, const model::FieldKeyReader& v) {
        v.key<&Widget::computed_key>(s.computed_key());
    }
};

Ref<Widget> make_widget(Model& m, const std::string& key) {
    Transaction txn = m.begin();
    auto w = std::make_unique<Widget>();
    w->key = key;
    const Ref<Widget> local = txn.create(std::move(w));
    return commit_ok(m, txn).resolve(local);
}

/// A type that opts three fields into fast lookup via define_keys(): a plain
/// string, a plain arithmetic field, and computed_key() itself (a nullary
/// const method, not a data member) -- proving a computed key can be indexed
/// and looked up the exact same way as a stored one.
class Gadget final : public model::Object<Gadget> {
public:
    std::string label;
    std::int64_t serial = 0;

    std::string computed_key() const { return "gad:" + label; }

    // no outgoing references, so no define_references() to declare

    template <class Self>
    static void define_keys(Self& s, const model::FieldKeyReader& v) {
        v.key<&Gadget::label>(s.label);
        v.key<&Gadget::serial>(s.serial);
        v.key<&Gadget::computed_key>(s.computed_key());
    }
};

Ref<Gadget> make_gadget(Model& m, const std::string& label, std::int64_t serial) {
    Transaction txn = m.begin();
    auto g = std::make_unique<Gadget>();
    g->label = label;
    g->serial = serial;
    const Ref<Gadget> local = txn.create(std::move(g));
    return commit_ok(m, txn).resolve(local);
}

Ref<Order> make_order(Model& m, const std::string& code, Ref<Account> account,
                      Opt<Order> parent = Opt<Order>{}, std::int64_t qty = 1) {
    Transaction txn = m.begin();
    auto o = std::make_unique<Order>();
    o->code = code;
    o->account = account;
    o->parent = parent;
    o->qty = qty;
    const Ref<Order> local = txn.create(std::move(o));
    return commit_ok(m, txn).resolve(local);
}

/// Runs `f(T*)` inside a fresh Transaction against `r` and commits. `f`
/// writes through the returned pointer exactly like the single-writer
/// sibling's `m.update(r)->field = x;`, just batched into an explicit call
/// here since there's no privileged writer thread to hand a bare pointer to
/// across statements.
template <class T, class F>
void update_field(Model& m, Ref<T> r, F&& f) {
    Transaction txn = m.begin();
    T* p = txn.update(r);
    CHECK(p != nullptr);
    if (p) f(p);
    commit_ok(m, txn);
}

/// Records a remove() intent for `r` and commits, returning the full resolved
/// kill count (including cascade). Mirrors the single-writer sibling's
/// `m.remove(r)` return value, just resolved one commit later.
template <class T>
std::size_t remove_and_commit(Model& m, Ref<T> r) {
    Transaction txn = m.begin();
    txn.remove(r);
    CommitResult res = m.try_commit(txn);
    CHECK(res.status == CommitStatus::Committed);
    std::size_t n = 0;
    for (const Change& c : res.changes)
        if (c.kind == ChangeKind::Deleted) ++n;
    return n;
}

/// Serialize the committed model to a comparable string, for exact-state checks.
std::string state_of(Model& m) {
    Snapshot s = m.snapshot();
    std::map<std::string, std::string> rows;
    s.for_each<Account>([&](const Account& a) {
        rows["acc:" + a.name] = "A(" + a.name + "," + std::to_string(a.balance) + ")";
    });
    s.for_each<Order>([&](const Order& o) {
        rows["ord:" + o.code] = "O(" + o.code + ",q=" + std::to_string(o.qty) +
                                ",a=" + std::to_string(o.account.raw().index) + ":" +
                                std::to_string(o.account.raw().gen) + ",p=" +
                                (o.parent ? std::to_string(o.parent.raw().index) + ":" +
                                                std::to_string(o.parent.raw().gen)
                                          : "-") +
                                ")";
    });
    std::string out = "v=" + std::to_string(s.version());
    for (auto& [k, v] : rows) out += " " + v;
    return out;
}

}  // namespace

// ---------------------------------------------------------------------------
// Basics
// ---------------------------------------------------------------------------

TEST(create_and_find_by_id_and_key) {
    Model m;
    const Ref<Account> a = make_account(m, "A1", 100);
    const Ref<Order> o = make_order(m, "O1", a, {}, 7);

    Snapshot s = m.snapshot();
    CHECK(s.find(a) != nullptr);
    CHECK(s.find(o) != nullptr);
    CHECK_EQ(s.find(a)->balance, 100);  // typed: no cast
    CHECK_EQ(s.find(o)->qty, 7);

    CHECK(s.find_by_key<&Account::name>("A1") != nullptr);
    CHECK(s.find_by_key<&Order::computed_key>("ord:O1") != nullptr);
    CHECK(s.find_by_key<&Order::computed_key>("ord:nope") == nullptr);
    CHECK(s.find_by_key<&Account::name>("ord:O1") == nullptr);  // right key, wrong type
    CHECK_EQ(s.find_by_key<&Account::name>("A1")->id, a.raw());
}

TEST(refs_resolve_through_snapshot) {
    Model m;
    const Ref<Account> a = make_account(m, "A1");
    const Ref<Order> p = make_order(m, "P", a);
    const Ref<Order> c = make_order(m, "C", a, p);

    Snapshot s = m.snapshot();
    const Order* child = s.find(c);

    const Account& acct = s.resolve(child->account);
    CHECK_EQ(acct.id, a.raw());
    CHECK(s.resolve(child->parent) != nullptr);
    CHECK_EQ(s.resolve(child->parent)->id, p.raw());
    CHECK(s.resolve(s.find(p)->parent) == nullptr);
}

TEST(uncommitted_writes_are_invisible_to_readers) {
    Model m;
    const Ref<Account> a = make_account(m, "A1");

    Snapshot before = m.snapshot();
    Transaction txn = m.begin();
    make_order(m, "placeholder", a);  // committed separately, just to touch `before`'s absence
    CHECK(before.find_by_key<&Order::computed_key>("ord:placeholder") == nullptr);

    Ref<Order> local = txn.create(std::make_unique<Order>());
    txn.update(local)->code = "O1";
    txn.update(local)->account = a;
    CHECK(txn.exists(local));  // visible in the transaction's own working set
    CHECK(before.find_by_key<&Order::computed_key>("ord:O1") == nullptr);  // not committed yet

    commit_ok(m, txn);
    CHECK(before.find_by_key<&Order::computed_key>("ord:O1") ==
          nullptr);  // old snapshot never changes
    CHECK(m.snapshot().find_by_key<&Order::computed_key>("ord:O1") != nullptr);
}

// ---------------------------------------------------------------------------
// Snapshot isolation
// ---------------------------------------------------------------------------

TEST(old_snapshot_sees_old_values) {
    Model m;
    const Ref<Account> a = make_account(m, "A1");
    const Ref<Order> o = make_order(m, "O1", a, {}, 1);

    Snapshot v1 = m.snapshot();
    update_field(m, o, [](Order* p) { p->qty = 999; });
    Snapshot v2 = m.snapshot();

    CHECK_EQ(v1.find(o)->qty, 1);  // still alive, still old
    CHECK_EQ(v2.find(o)->qty, 999);
    CHECK(v1.version() < v2.version());
}

TEST(deleted_object_survives_in_older_snapshot) {
    // The reclamation invariant: nothing may be freed while a reader can still
    // reach it. Under ASan, getting this wrong is a use-after-free here.
    Model m;
    const Ref<Account> a = make_account(m, "A1");
    const Ref<Order> o = make_order(m, "O1", a);

    Snapshot old = m.snapshot();
    remove_and_commit(m, o);

    CHECK(m.snapshot().find(o) == nullptr);  // gone from the new version
    CHECK(old.find(o) != nullptr);           // still there in the old one
    CHECK_EQ(old.find(o)->code, std::string("O1"));
    CHECK(m.retired_pending() > 0);  // held back from the free list
}

TEST(reclamation_advances_when_snapshots_are_dropped) {
    Model m;
    const Ref<Account> a = make_account(m, "A1");
    const Ref<Order> o = make_order(m, "O1", a);

    {
        Snapshot pin = m.snapshot();
        remove_and_commit(m, o);
        CHECK(m.retired_pending() > 0);       // pinned by `pin`
        CHECK(m.wait_for_reclamation() > 0);  // still pinned: barrier confirms it can't free
    }
    // `pin` is gone; the next commit's watermark should free the backlog.
    make_order(m, "O2", a);
    CHECK_EQ(m.wait_for_reclamation(), std::size_t{0});
}

TEST(stale_id_does_not_alias_a_recycled_slot) {
    Model m;
    const Ref<Account> a = make_account(m, "A1");
    const Ref<Order> o1 = make_order(m, "O1", a);

    remove_and_commit(m, o1);
    const Ref<Order> o2 = make_order(m, "O2", a);

    Snapshot s = m.snapshot();
    CHECK_EQ(o2.raw().index, o1.raw().index);  // slot was recycled
    CHECK(o2.raw().gen != o1.raw().gen);       // but the generation moved
    CHECK(s.find(o1) == nullptr);              // the stale handle must NOT resolve to O2
    CHECK(s.find(o2) != nullptr);
}

TEST(transaction_local_update_does_not_alias_a_stale_generation_of_the_same_slot) {
    // Regression: Transaction::local_updated_ is keyed by bare slot index (a
    // transaction only ever clones ONE generation of a given slot -- the one
    // alive at base()). peek_impl/update_impl must verify the FULL Id
    // (index + generation), not just the index, before trusting that map --
    // otherwise a transaction that updates slot K, then separately asks
    // about a stale, dead generation of the SAME slot K (e.g. a handle from
    // an aging cache that was never pruned), gets a false "exists" answer
    // aliased to the WRONG object -- exactly the kind of dangling-looking
    // Ref this whole model exists to prevent.
    Model m;
    const Ref<Account> a = make_account(m, "A1");
    const Ref<Order> o1 = make_order(m, "O1", a);
    const std::uint32_t slot = o1.raw().index;

    remove_and_commit(m, o1);
    const Ref<Order> o2 = make_order(m, "O2", a);  // reuses `a` -- nothing else steals the freed slot
    CHECK_EQ(o2.raw().index, slot);
    CHECK(o2.raw().gen != o1.raw().gen);

    Transaction txn = m.begin();
    txn.update(o2)->qty = 42;  // populates local_updated_[slot], keyed by o2's generation

    // A stale handle for the SAME slot, a DIFFERENT (dead) generation, must
    // still be reported as dead -- not aliased to o2's live local clone.
    CHECK(txn.update(o1) == nullptr);
    CHECK(!txn.exists(o1));
    CHECK(txn.peek(o1) == nullptr);

    // The live one is unaffected by the check above.
    CHECK(txn.exists(o2));
    CHECK_EQ(txn.peek(o2)->qty, 42);
}

// ---------------------------------------------------------------------------
// Cascade delete (resolved at try_commit() time, not eagerly)
// ---------------------------------------------------------------------------

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

TEST(reassigning_a_ref_field_after_update_is_tracked_for_cascade) {
    Model m;
    const Ref<Account> a1 = make_account(m, "A1");
    const Ref<Account> a2 = make_account(m, "A2");
    const Ref<Order> o = make_order(m, "O1", a1);

    update_field(m, o, [&](Order* p) { p->account = a2; });  // reassign the non-nullable Ref
    CHECK(m.snapshot().find(o)->account == a2);

    const std::size_t killed = remove_and_commit(m, a2);  // must cascade-kill o: Ref<> is non-nullable

    CHECK_EQ(killed, std::size_t{2});  // a2 and o
    CHECK(m.snapshot().find(o) == nullptr);
}

TEST(reassigning_an_opt_field_after_update_is_tracked_for_cascade) {
    Model m;
    const Ref<Account> a = make_account(m, "A1");
    const Ref<Order> p1 = make_order(m, "P1", a);
    const Ref<Order> p2 = make_order(m, "P2", a);
    const Ref<Order> c = make_order(m, "C", a, p1);  // parent = p1 initially

    update_field(m, c, [&](Order* o) { o->parent = p2; });  // reassign the nullable Opt<>
    CHECK(m.snapshot().find(c)->parent == Opt<Order>(p2));

    remove_and_commit(m, p2);  // must null c's parent: the index must track p2, not p1
    CHECK(m.snapshot().find(c) != nullptr);         // Opt<>: survives
    CHECK(!m.snapshot().find(c)->parent);  // ...nulled

    // p1 must be untouched by any of this -- proves the OLD edge (c -> p1)
    // was correctly dropped rather than left as a phantom referrer.
    remove_and_commit(m, p1);
    CHECK(m.snapshot().find(c) != nullptr);
}

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

TEST(update_returns_null_for_a_dead_id) {
    Model m;
    const Ref<Account> a = make_account(m, "A1");
    const Ref<Order> o = make_order(m, "O1", a);
    remove_and_commit(m, o);

    Transaction txn = m.begin();
    CHECK(txn.update(o) == nullptr);
    CHECK(!txn.exists(o));
}

// ---------------------------------------------------------------------------
// Events
// ---------------------------------------------------------------------------

TEST(subscriber_gets_a_changeset_per_commit) {
    Model m;
    auto sub = m.subscribe(8);

    const Ref<Account> a = make_account(m, "A1");

    Update u;
    CHECK(sub->try_drain(u));
    CHECK_EQ(u.changes.size(), std::size_t{1});
    CHECK_EQ(u.changes[0].id, a.raw());
    CHECK(u.changes[0].kind == ChangeKind::Created);
    CHECK(!u.coalesced);
    CHECK(u.snapshot.find(a) != nullptr);
    CHECK(!sub->try_drain(u));
}

TEST(overflow_coalesces_instead_of_growing) {
    Model m;
    auto sub = m.subscribe(/*depth=*/2);

    const Ref<Account> a = make_account(m, "A1");
    const Ref<Order> o = make_order(m, "O1", a, {}, 1);

    for (int i = 0; i < 20; ++i) {  // never drained -- must not grow unbounded
        update_field(m, o, [&](Order* p) { p->qty = i; });
    }

    int batches = 0;
    bool saw_coalesced = false;
    std::int64_t final_qty = -1;
    Update u;
    while (sub->try_drain(u)) {
        ++batches;
        if (u.coalesced) saw_coalesced = true;
        if (const Order* p = u.snapshot.find(o)) final_qty = p->qty;
    }
    CHECK(batches <= 3);  // bounded, not 22
    CHECK(saw_coalesced);
    CHECK_EQ(final_qty, 19);  // the latest state still arrives
}

TEST(create_then_delete_between_drains_cancels_out) {
    Model m;
    auto sub = m.subscribe(/*depth=*/1);

    const Ref<Account> a = make_account(m, "A1");

    Update drain;
    while (sub->try_drain(drain)) {
    }

    const Ref<Order> o = make_order(m, "O1", a);
    remove_and_commit(m, o);
    update_field(m, a, [](Account* p) { p->balance = 5; });

    int mentions_o = 0, mentions_a = 0;
    Update u;
    while (sub->try_drain(u)) {
        for (const Change& c : u.changes) {
            if (c.id == o.raw()) ++mentions_o;
            if (c.id == a.raw()) ++mentions_a;
        }
    }
    CHECK_EQ(mentions_o, 0);
    CHECK(mentions_a >= 1);
}

// ---------------------------------------------------------------------------
// Typing
// ---------------------------------------------------------------------------

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

TEST(field_index_is_cleaned_up_on_removal) {
    Model m;
    const Ref<Gadget> g = make_gadget(m, "Widget", 111);
    remove_and_commit(m, g);

    Snapshot s = m.snapshot();
    CHECK(s.find_by_key<&Gadget::label>("Widget") == nullptr);
    CHECK(s.find_by_key<&Gadget::serial>(111) == nullptr);
}

TEST(field_index_duplicate_value_is_last_write_wins) {
    Model m;
    const Ref<Gadget> g1 = make_gadget(m, "Same", 1);
    const Ref<Gadget> g2 = make_gadget(m, "Same", 2);

    Snapshot s = m.snapshot();
    CHECK_EQ(s.find_by_key<&Gadget::label>("Same")->id, g2.raw());  // later write wins
    (void)g1;
}

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

// ---------------------------------------------------------------------------
// Generation exhaustion
// ---------------------------------------------------------------------------

TEST(an_exhausted_slot_is_retired_not_reused) {
    Model m;
    const Ref<Account> a = make_account(m, "A1");
    const Ref<Order> o1 = make_order(m, "O1", a);
    const std::uint32_t slot = o1.raw().index;

    remove_and_commit(m, o1);
    m.debug_set_generation(slot, kGenMax);

    const Ref<Order> o2 = make_order(m, "O2", a);

    CHECK_EQ(m.exhausted_slots(), std::size_t{1});
    CHECK(o2.raw().index != slot);
    CHECK(o2.raw().gen != 0);
    CHECK(m.snapshot().find(o2) != nullptr);
}

TEST(a_stale_handle_never_aliases_across_exhaustion) {
    Model m;
    const Ref<Account> a = make_account(m, "A1");
    const Ref<Order> o1 = make_order(m, "O1", a);
    const Ref<Order> stale = o1;
    const std::uint32_t slot = o1.raw().index;

    remove_and_commit(m, o1);
    m.debug_set_generation(slot, kGenMax);
    make_order(m, "O2", a);

    Snapshot s = m.snapshot();
    CHECK(s.find(stale) == nullptr);
}

// ---------------------------------------------------------------------------
// Background reaper + lock-free reads
// ---------------------------------------------------------------------------

TEST(reaper_frees_eventually_after_snapshots_drop) {
    Model m;
    const Ref<Account> a = make_account(m, "A1");
    const Ref<Order> o = make_order(m, "O1", a);

    Snapshot pin = m.snapshot();
    remove_and_commit(m, o);
    CHECK(m.wait_for_reclamation() > 0);  // pinned by `pin`, cannot free yet

    { Snapshot drop = std::move(pin); }
    CHECK_EQ(m.wait_for_reclamation(), std::size_t{0});
}

TEST(a_large_cascade_does_not_block_forever_and_reclaims) {
    Model m;
    const Ref<Account> a = make_account(m, "hub");
    {
        // Scoped: a Transaction pins its base() Snapshot for its entire
        // lifetime, exactly like any reader's snapshot. Left unscoped, it
        // would still be pinning the pre-cascade version below, and
        // wait_for_reclamation() would (correctly) never reach zero.
        Transaction txn = m.begin();
        for (int i = 0; i < 500; ++i) {
            auto o = std::make_unique<Order>();
            o->code = "O" + std::to_string(i);
            o->account = a;
            txn.create(std::move(o));
        }
        commit_ok(m, txn);
    }
    CHECK_EQ(m.snapshot().size(), std::size_t{501});

    const std::size_t killed = remove_and_commit(m, a);  // takes the account + all 500 orders
    CHECK_EQ(killed, std::size_t{501});
    CHECK_EQ(m.snapshot().size(), std::size_t{0});

    CHECK_EQ(m.wait_for_reclamation(), std::size_t{0});
}

// ---------------------------------------------------------------------------
// Pre-commit hook
// ---------------------------------------------------------------------------

TEST(pre_commit_hook_approving_commits_normally) {
    Model m;
    int seen = 0;
    m.set_pre_commit([&](Model&, const std::vector<Change>& changes) {
        seen = static_cast<int>(changes.size());
        return true;
    });

    make_account(m, "A1");
    CHECK_EQ(seen, 1);
    CHECK_EQ(m.snapshot().size(), std::size_t{1});
}

TEST(pre_commit_hook_is_not_invoked_for_an_empty_commit) {
    Model m;
    bool called = false;
    m.set_pre_commit([&](Model&, const std::vector<Change>&) {
        called = true;
        return true;
    });

    Transaction txn = m.begin();  // nothing pending
    CommitResult res = m.try_commit(txn);
    CHECK(res.status == CommitStatus::Committed);
    CHECK(!called);
}

TEST(pre_commit_hook_sees_the_fully_resolved_changeset_including_cascade_deletes) {
    Model m;
    const Ref<Account> a = make_account(m, "A1");
    const Ref<Order> o = make_order(m, "O1", a);

    std::size_t seen_deletes = 0;
    m.set_pre_commit([&](Model&, const std::vector<Change>& changes) {
        for (const Change& c : changes)
            if (c.kind == ChangeKind::Deleted) ++seen_deletes;
        return true;
    });

    const std::size_t killed = remove_and_commit(m, a);  // cascades to o
    CHECK_EQ(killed, std::size_t{2});
    CHECK_EQ(seen_deletes, std::size_t{2});
    (void)o;
}

TEST(pre_commit_hook_veto_unwinds_everything_as_if_try_commit_were_never_called) {
    Model m;
    const Ref<Account> a = make_account(m, "A1");
    const std::string before = state_of(m);

    m.set_pre_commit([](Model&, const std::vector<Change>&) { return false; });

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

TEST(create_with_a_dead_ref_throws) {
    Model m;
    const Ref<Account> a = make_account(m, "A1");
    const Ref<Order> good = make_order(m, "GOOD", a);
    remove_and_commit(m, a);  // a and good are now dead

    Transaction txn = m.begin();
    auto o = std::make_unique<Order>();
    o->code = "BAD";
    o->account = a;  // Ref<Account> to an already-dead account (as of txn's own base)
    txn.create(std::move(o));

    bool threw = false;
    try {
        m.try_commit(txn);
    } catch (const Model::IntegrityError&) {
        threw = true;
    }
    CHECK(threw);
    CHECK(m.snapshot().find(good) == nullptr);  // txn's base already reflects a/good gone
}

TEST(create_with_a_null_nonnullable_ref_throws) {
    Model m;
    Transaction txn = m.begin();
    auto o = std::make_unique<Order>();
    o->code = "X";  // account left default -> null Ref<Account>
    txn.create(std::move(o));

    bool threw = false;
    try {
        m.try_commit(txn);
    } catch (const Model::IntegrityError&) {
        threw = true;
    }
    CHECK(threw);
    CHECK_EQ(m.snapshot().size(), std::size_t{0});
}

// ---------------------------------------------------------------------------
// Persistent secondary index
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// Views
// ---------------------------------------------------------------------------

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

TEST(view_of_a_stale_handle_is_empty) {
    Model m;
    const Ref<Account> a = make_account(m, "A1");
    const Ref<Order> o = make_order(m, "O1", a);
    remove_and_commit(m, o);

    Snapshot s = m.snapshot();
    CHECK(!s.view(o).has_value());
    CHECK(!s.view_by_key<&Order::computed_key>("ord:O1").has_value());
}

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

// ---------------------------------------------------------------------------
// Transaction-local semantics
// ---------------------------------------------------------------------------

TEST(begin_returns_a_transaction_bound_to_the_current_snapshot) {
    Model m;
    make_account(m, "A1");

    Snapshot s = m.snapshot();
    Transaction txn = m.begin();
    CHECK_EQ(txn.base_version(), s.version());
    CHECK_EQ(txn.base().version(), m.current_version());
}

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
    const Ref<Order> o_real = res.resolve(o_local);
    Snapshot s = m.snapshot();
    CHECK(s.find(o_real) != nullptr);
    CHECK_EQ(s.resolve(s.find(o_real)->account).name, std::string("A1"));
}

TEST(a_local_id_from_one_transaction_does_not_resolve_in_a_different_transaction) {
    Model m;
    Transaction txn1 = m.begin();
    auto a = std::make_unique<Account>();
    a->name = "A1";
    const Ref<Account> a_local = txn1.create(std::move(a));  // placeholder, only meaningful inside txn1

    // A second, independent transaction never saw txn1's create -- to txn2,
    // a_local is just some id it never issued and never cloned from base().
    Transaction txn2 = m.begin();
    CHECK(txn2.peek(a_local) == nullptr);
    CHECK(!txn2.exists(a_local));
    CHECK(txn2.update(a_local) == nullptr);

    // Embedding the stray local id in something txn2 commits fails loudly:
    // try_commit()'s remap table (built fresh per attempt) only knows about
    // local ids THIS transaction's own local_created_ produced.
    auto o = std::make_unique<Order>();
    o->code = "O1";
    o->account = a_local;
    txn2.create(std::move(o));

    bool threw = false;
    try {
        m.try_commit(txn2);
    } catch (const Model::IntegrityError&) {
        threw = true;
    }
    CHECK(threw);
    CHECK_EQ(m.snapshot().size(), std::size_t{0});  // neither txn1 nor txn2 ever published anything
}

TEST(a_local_id_that_collides_with_another_transactions_own_local_index_silently_aliases) {
    Model m;

    Transaction txn1 = m.begin();
    auto a = std::make_unique<Account>();
    a->name = "FROM-TXN1";
    const Ref<Account> a_local = txn1.create(std::move(a));  // txn1's local counter: index 0

    Transaction txn2 = m.begin();
    auto own = std::make_unique<Account>();
    own->name = "TXN2-OWN";
    const Ref<Account> own_local = txn2.create(std::move(own));  // txn2's local counter ALSO starts at 0

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
    CHECK_EQ(m.snapshot().find(res.resolve(own_local))->name, std::string("TXN2-OWN"));
    CHECK_EQ(m.snapshot().size(), std::size_t{1});  // only txn2's object was ever real
}

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

TEST(same_transaction_create_referencing_an_existing_object_then_remove_of_that_object_cascades_the_new_object_too) {
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
    CHECK(m.snapshot().find(res.resolve(o_local)) == nullptr);
}

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

// ---------------------------------------------------------------------------
// try_commit correctness: conflicts
// ---------------------------------------------------------------------------

TEST(try_commit_of_a_non_conflicting_transaction_succeeds_and_publishes) {
    Model m;
    Transaction txn = m.begin();
    auto a = std::make_unique<Account>();
    a->name = "A1";
    txn.create(std::move(a));
    CommitResult res = m.try_commit(txn);
    CHECK(res.status == CommitStatus::Committed);
    CHECK(res.snapshot.version() == m.current_version());
    CHECK_EQ(res.changes.size(), std::size_t{1});
}

TEST(two_concurrent_transactions_touching_disjoint_ids_both_succeed) {
    Model m;
    const Ref<Account> a1 = make_account(m, "A1", 1);
    const Ref<Account> a2 = make_account(m, "A2", 2);

    std::atomic<int> committed{0};
    auto worker = [&](Ref<Account> target, std::int64_t new_balance) {
        Transaction txn = m.begin();
        txn.update(target)->balance = new_balance;
        CommitResult res = m.try_commit(txn);
        if (res.status == CommitStatus::Committed) committed.fetch_add(1);
    };

    std::thread t1([&] { worker(a1, 100); });
    std::thread t2([&] { worker(a2, 200); });
    t1.join();
    t2.join();

    CHECK_EQ(committed.load(), 2);
    Snapshot s = m.snapshot();
    CHECK_EQ(s.find(a1)->balance, 100);
    CHECK_EQ(s.find(a2)->balance, 200);
}

TEST(two_concurrent_transactions_updating_the_same_id_one_conflicts_with_id_set_overlap) {
    Model m;
    const Ref<Account> a = make_account(m, "A1", 0);

    Transaction t1 = m.begin(m.snapshot());
    Transaction t2 = m.begin(m.snapshot());
    t1.update(a)->balance = 111;
    t2.update(a)->balance = 222;

    CommitResult r1 = m.try_commit(t1);
    CHECK(r1.status == CommitStatus::Committed);

    CommitResult r2 = m.try_commit(t2);
    CHECK(r2.status == CommitStatus::Conflict);
    CHECK(r2.conflict.has_value());
    CHECK(r2.conflict->reason == ConflictReason::IdSetOverlap);
    CHECK(std::find(r2.conflict->ids.begin(), r2.conflict->ids.end(), a.raw()) !=
          r2.conflict->ids.end());

    // The loser can retry cleanly against the new base.
    Transaction retry = m.begin();
    retry.update(a)->balance = 222;
    CommitResult rr = m.try_commit(retry);
    CHECK(rr.status == CommitStatus::Committed);
    CHECK_EQ(m.snapshot().find(a)->balance, 222);
}

TEST(update_conflicts_with_a_concurrent_remove_of_the_same_id) {
    Model m;
    const Ref<Account> a = make_account(m, "A1");

    Transaction t1 = m.begin(m.snapshot());
    Transaction t2 = m.begin(m.snapshot());
    t1.remove(a);
    t2.update(a)->balance = 5;

    CommitResult r1 = m.try_commit(t1);
    CHECK(r1.status == CommitStatus::Committed);

    CommitResult r2 = m.try_commit(t2);
    CHECK(r2.status == CommitStatus::Conflict);
    CHECK(r2.conflict->reason == ConflictReason::IdSetOverlap);
}

TEST(ref_integrity_is_revalidated_against_latest_not_just_base) {
    // txn A's base has an Account alive. txn B deletes it and commits first.
    // A never touches that id directly, but references it via a brand-new
    // object -- A's commit must still fail (not id-set overlap: A never
    // wrote that id -- but because the Ref<> it just created would dangle
    // against the LATEST state).
    Model m;
    const Ref<Account> a = make_account(m, "A1");

    Snapshot base = m.snapshot();
    Transaction a_txn = m.begin(base);
    Transaction b_txn = m.begin(base);

    b_txn.remove(a);
    CommitResult rb = m.try_commit(b_txn);
    CHECK(rb.status == CommitStatus::Committed);

    auto o = std::make_unique<Order>();
    o->code = "O1";
    o->account = a;  // new Order -> now-dead Account
    a_txn.create(std::move(o));

    bool threw = false;
    CommitResult ra;
    try {
        ra = m.try_commit(a_txn);
    } catch (const Model::IntegrityError&) {
        threw = true;
    }
    CHECK(!threw);  // classified as Conflict, not an escaped exception
    CHECK(ra.status == CommitStatus::Conflict);
    CHECK(ra.conflict->reason == ConflictReason::RefIntegrity);
}

TEST(cascade_delete_resolved_at_commit_time_matches_a_synchronous_bfs) {
    Model m;
    const Ref<Account> hub = make_account(m, "hub");
    Transaction seed = m.begin();
    for (int i = 0; i < 20; ++i) {
        auto o = std::make_unique<Order>();
        o->code = "O" + std::to_string(i);
        o->account = hub;
        seed.create(std::move(o));
    }
    commit_ok(m, seed);

    const std::size_t killed = remove_and_commit(m, hub);
    CHECK_EQ(killed, std::size_t{21});  // hub + all 20 spokes, exactly

    Snapshot s = m.snapshot();
    CHECK_EQ(s.size(), std::size_t{0});
}

// ---------------------------------------------------------------------------
// Changelog / GC
// ---------------------------------------------------------------------------

TEST(changelog_entries_are_pruned_once_no_open_transaction_or_snapshot_needs_them) {
    Model m;
    make_account(m, "A1");

    {
        Snapshot pin = m.snapshot();  // pins the watermark at this version
        for (int i = 0; i < 5; ++i) make_account(m, "X" + std::to_string(i));
        CHECK(m.debug_changelog_size() >= std::size_t{5});
    }
    // `pin` dropped -- each subsequent commit's prune sees a watermark that no
    // longer includes it, so the backlog collapses over the next couple of
    // commits. A commit's own changelog entry always survives ITS OWN prune
    // call (that same commit's Transaction::base() is still registered in
    // live_ at the moment prune runs, inside try_commit() itself) -- it is
    // only pruned once a LATER commit's base has moved past it. So the
    // backlog settles at "the most recent commit's own entry", not zero.
    make_account(m, "after_pin_drop");
    make_account(m, "flush");
    CHECK(m.debug_changelog_size() <= std::size_t{1});
}

// ---------------------------------------------------------------------------
// Concurrency stress
// ---------------------------------------------------------------------------

TEST(concurrent_stress_many_writer_threads_hammering_try_commit_never_corrupts_referrers_or_leaks_a_dangling_ref) {
    Model m;
    std::vector<Ref<Account>> accounts;
    for (int i = 0; i < 4; ++i) accounts.push_back(make_account(m, "A" + std::to_string(i)));
    std::vector<Ref<Order>> seed_orders;
    for (int i = 0; i < 10; ++i)
        seed_orders.push_back(make_order(m, "O" + std::to_string(i), accounts[i % 4]));

    std::atomic<bool> stop{false};
    std::atomic<std::uint64_t> resolved{0};
    std::atomic<std::uint64_t> snaps{0};
    std::atomic<int> readers_ready{0};
    std::atomic<int> next_id{100};

    auto reader = [&] {
        bool signaled = false;
        while (!stop.load(std::memory_order_relaxed)) {
            Snapshot s = m.snapshot();
            ++snaps;
            if (!signaled) {
                readers_ready.fetch_add(1, std::memory_order_relaxed);
                signaled = true;
            }
            s.for_each<Order>([&](const Order& o) {
                const Account& acct = s.resolve(o.account);  // must never dangle
                if (acct.name.empty()) ++g_failures;
                ++resolved;
                (void)s.resolve(o.parent);
            });
        }
    };

    auto writer = [&](unsigned seed) {
        std::mt19937 rng(seed);
        while (!stop.load(std::memory_order_relaxed)) {
            Transaction txn = m.begin();
            const int roll = rng() % 100;
            if (roll < 50) {
                auto o = std::make_unique<Order>();
                o->code = "W" + std::to_string(next_id.fetch_add(1));
                o->account = accounts[rng() % accounts.size()];
                txn.create(std::move(o));
            } else if (roll < 80) {
                if (Order* o = txn.update(seed_orders[rng() % seed_orders.size()]))
                    o->qty = static_cast<std::int64_t>(rng() % 50);
            } else {
                txn.remove(seed_orders[rng() % seed_orders.size()]);
            }
            m.try_commit(txn);  // conflicts are expected and fine; just don't corrupt anything
        }
    };

    std::thread r1(reader), r2(reader);
    while (readers_ready.load(std::memory_order_relaxed) < 2) std::this_thread::yield();

    std::thread w1([&] { writer(11); }), w2([&] { writer(22); }), w3([&] { writer(33); });

    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    stop = true;
    w1.join();
    w2.join();
    w3.join();
    r1.join();
    r2.join();

    CHECK(snaps.load() > 0);
    CHECK(resolved.load() > 0);

    Snapshot final_s = m.snapshot();
    final_s.for_each<Order>(
        [&](const Order& o) { CHECK(final_s.resolve(o.account).id == o.account.raw()); });
}

// ---------------------------------------------------------------------------

int main(int argc, char** argv) {
    const char* filter = argc > 1 ? argv[1] : nullptr;
    int run = 0;

    for (const auto& t : registry()) {
        if (filter && std::string(t.name).find(filter) == std::string::npos) continue;
        g_current = t.name;
        const int before = g_failures;
        std::printf("[ RUN  ] %s\n", t.name);
        t.fn();
        ++run;
        std::printf("[%s] %s\n", g_failures == before ? "  OK  " : " FAIL ", t.name);
    }

    std::printf("\n%d test(s) run, %d check(s) failed\n", run, g_failures);
    return g_failures == 0 ? 0 : 1;
}
