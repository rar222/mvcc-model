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
// try_commit() returns (see CommitResult::to_real's doc comment). These
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
    return commit_ok(m, txn).to_real(local);
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
    return commit_ok(m, txn).to_real(local);
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
    return commit_ok(m, txn).to_real(local);
}

/// Self-referential, with a CACHED nullable reference. Order::parent (the
/// shared demo type's analogous field) is deliberately left OUT of
/// define_cached_references() to demonstrate the "index only what's worth
/// it" tradeoff -- so testing the cascade-NULL maintenance path of
/// Model::reconcile_cached_references needs a field that actually IS cached
/// and nullable. Node exists purely for that.
class Node final : public model::Object<Node> {
public:
    std::string label;
    model::Opt<Node> parent;

    template <class Self, class V>
    static void define_references(Self& s, V&& v) {
        v(model::field_tag<&Node::parent>(), "parent", s.parent);
    }

    template <class Self>
    static void define_cached_references(Self& s, const model::RefIndexReader& v) {
        (void)s;
        v.index<&Node::parent>();
    }
};

Ref<Node> make_node(Model& m, const std::string& label, Opt<Node> parent = Opt<Node>{}) {
    Transaction txn = m.begin();
    auto n = std::make_unique<Node>();
    n->label = label;
    n->parent = parent;
    const Ref<Node> local = txn.create(std::move(n));
    return commit_ok(m, txn).to_real(local);
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
    return commit_ok(m, txn).to_real(local);
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

/// A wider record type for the large-scale mixed-thread stress test below:
/// 5 fields, 2 of them references -- owner (Ref<Account>, non-nullable: the
/// cascade-delete path) and related (Opt<Record>, nullable, self-referential:
/// the cascade-null path) -- so every write there exercises reconcile_
/// referrer_edges across two DIFFERENT edges at once, not just one. owner is
/// also cached (define_cached_references), so the same run stress-tests
/// Root::by_cached_reference concurrently for the first time -- every prior
/// cached-reference test was single-threaded.
class Record final : public model::Object<Record> {
public:
    std::string label;
    std::int64_t value = 0;
    bool flag = false;
    model::Ref<Account> owner;
    model::Opt<Record> related;

    template <class Self, class V>
    static void define_references(Self& s, V&& v) {
        v(model::field_tag<&Record::owner>(), "owner", s.owner);
        v(model::field_tag<&Record::related>(), "related", s.related);
    }

    template <class Self>
    static void define_cached_references(Self& s, const model::RefIndexReader& v) {
        (void)s;
        v.index<&Record::owner>();
    }
};

/// Same pattern as examples/demo.cpp's pick_live: a handle this Transaction's
/// OWN local view still believes is live, checked against `txn` (base plus
/// this transaction's own pending edits) -- not some model-wide notion of
/// existence, since there isn't one. Another thread's concurrent commit can
/// still make this stale by the time try_commit() runs; that race is exactly
/// what the conflict check exists to catch.
template <class T>
Ref<T> pick_live(const Transaction& txn, const std::vector<Ref<T>>& v, std::mt19937& rng) {
    for (int tries = 0; tries < 8 && !v.empty(); ++tries) {
        const Ref<T> r = v[rng() % v.size()];
        if (txn.exists(r)) return r;
    }
    return Ref<T>{};
}

}  // namespace

// ---------------------------------------------------------------------------
// Basics
// ---------------------------------------------------------------------------

// Basic roundtrip: create+commit, then find both by Ref/Id and by a
// define_keys()-declared field, and confirm a typed lookup distinguishes
// two types even when one's key string looks like the other's.
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

// Ref<>/Opt<> resolve correctly through a Snapshot: a non-null Opt
// resolves to the right object, a null Opt (the root order's parent)
// resolves to nullptr -- no special-casing needed by the caller.
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

// A Transaction's local edits -- including a same-transaction local-id
// create -- are invisible to any Snapshot taken before commit, and only
// become visible via a FRESH Snapshot taken after commit.
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

// Snapshot isolation's headline property: a snapshot taken before an
// update keeps reporting the pre-update value even after a later commit
// changes it and publishes a newer version.
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

// An older Snapshot can still find and read a since-deleted object --
// the reclamation invariant (nothing freed while a reader can reach it)
// made directly observable; see the in-body comment for the ASan angle.
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

// wait_for_reclamation() reports objects as pinned while a Snapshot holds
// their version, and reports zero once that snapshot is dropped and the
// next commit moves the watermark past it.
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

// Invariant 5: a stale handle from before a slot was recycled must NOT
// resolve to the new occupant, even though it shares the same slot index
// -- the generation mismatch is what stops the aliasing.
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

// Regression test: see the in-body comment for the exact bug shape --
// Transaction::local_updated_ is keyed by bare slot index, so peek_impl/
// update_impl must also check the FULL Id before trusting a hit.
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
    const Ref<Order> o2 =
        make_order(m, "O2", a);  // reuses `a` -- nothing else steals the freed slot
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

// ---------------------------------------------------------------------------
// Events
// ---------------------------------------------------------------------------

// A subscription receives exactly one Update per commit, carrying that
// commit's own changeset paired with a matching post-commit Snapshot.
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

// Exceeding a subscriber's queue depth coalesces intermediate updates
// into one entry instead of growing the queue unboundedly -- and the
// coalesced delivery still carries the LATEST value, not a stale one.
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

// Subscription::collapse's create+delete cancellation: an object created
// and deleted before the subscriber ever drains never appears in the
// delivered changeset at all, not even as a no-op pair.
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

// Real cross-thread producer/consumer: every other Subscription test above
// calls push()/try_drain()/collapse() synchronously, in one thread -- never
// the actual wait()/shutdown() wake-up path that's the entire point of the
// blocking API. Here a genuinely slow consumer thread (a deliberate sleep
// per item) forces the writer to overflow a small queue and coalesce under
// real timing pressure, not a manually-triggered one; and shutdown() must
// wake the blocked consumer so it exits instead of hanging forever.
TEST(a_slow_subscriber_thread_wakes_from_wait_after_shutdown_and_sees_coalescing) {
    Model m;
    auto sub = m.subscribe(/*queue_depth=*/2);

    std::atomic<int> updates_seen{0};
    std::atomic<bool> saw_coalesced{false};
    std::thread consumer([&] {
        Update u;
        while (sub->wait(u)) {
            updates_seen.fetch_add(1, std::memory_order_relaxed);
            if (u.coalesced) saw_coalesced = true;
            std::this_thread::sleep_for(std::chrono::milliseconds(2));  // deliberately the bottleneck
        }
    });

    constexpr int kCommits = 50;
    for (int i = 0; i < kCommits; ++i) make_account(m, "A" + std::to_string(i));

    m.shutdown();  // must wake the consumer -- wait() drains whatever's left, then returns false
    consumer.join();

    CHECK(updates_seen.load() > 0);
    CHECK(updates_seen.load() <= kCommits);  // never more entries than commits, coalesced or not
    CHECK(saw_coalesced.load());             // the actual proof that overflow/coalescing fired
}

// ---------------------------------------------------------------------------
// Typing
// ---------------------------------------------------------------------------

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

// Two objects sharing the same indexed value: find_by_key resolves to
// whichever was written last, matching the documented contract.
TEST(field_index_duplicate_value_is_last_write_wins) {
    Model m;
    const Ref<Gadget> g1 = make_gadget(m, "Same", 1);
    const Ref<Gadget> g2 = make_gadget(m, "Same", 2);

    Snapshot s = m.snapshot();
    CHECK_EQ(s.find_by_key<&Gadget::label>("Same")->id, g2.raw());  // later write wins
    (void)g1;
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

// ---------------------------------------------------------------------------
// Generation exhaustion
// ---------------------------------------------------------------------------

// A slot whose generation is forced to kGenMax is permanently withdrawn
// from free_slots_ rather than handed out again, and exhausted_slots()
// reports it.
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

// A handle captured before its slot was exhaustion-retired never
// resolves to whatever object a later, freshly allocated DIFFERENT slot
// ends up holding.
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

// wait_for_reclamation() as a barrier: it reports work still pinned
// while a Snapshot is held, and reports fully drained once that Snapshot
// is dropped.
TEST(reaper_frees_eventually_after_snapshots_drop) {
    Model m;
    const Ref<Account> a = make_account(m, "A1");
    const Ref<Order> o = make_order(m, "O1", a);

    Snapshot pin = m.snapshot();
    remove_and_commit(m, o);
    CHECK(m.wait_for_reclamation() > 0);  // pinned by `pin`, cannot free yet

    {
        Snapshot drop = std::move(pin);
    }
    CHECK_EQ(m.wait_for_reclamation(), std::size_t{0});
}

// A single remove() cascading through 500 objects completes and is
// fully reclaimable -- cascade fan-out isn't hard-capped by size, and
// the reaper can clear the whole backlog afterward.
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

// Regression test for the reconcile_out_refs undo-log bug -- see
// the in-body comment. A vetoed reassignment must restore the old edge
// AND remove the phantom new one, or a later cascade sees a lie.
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

// Same undo-log discipline for by_field_: a vetoed rename must leave the
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

    make_account(m, "UNRELATED");  // publish a fresh root carrying by_field_
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
        m.set_pre_commit([](Model&, const Transaction&, const std::vector<Change>&) { return true; });
    m.set_pre_commit({});
    stop = true;
    committer.join();
    CHECK(m.snapshot().size() > 0);
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

// ---------------------------------------------------------------------------
// Pre-transactions (Model::set_pre_transactions / run_pre_transaction)
// ---------------------------------------------------------------------------

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
        CommitResult r = model.run_pre_transaction(pre);
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
        CommitResult r = model.run_pre_transaction(pre);
        CHECK(r.status == CommitStatus::Invalid);
    });

    Transaction txn = m.begin();
    auto a = std::make_unique<Account>();
    a->name = "NEVER_APPLIED";
    const Ref<Account> local = txn.create(std::move(a));

    CommitResult res = m.try_commit(txn);
    CHECK(res.status == CommitStatus::PrecommitConflict);
    CHECK(res.error.has_value());  // forwarded from the failing pre-transaction
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
        CHECK(model.run_pre_transaction(pre1).status == CommitStatus::Committed);

        Transaction pre2 = model.begin();
        auto o = std::make_unique<Order>();
        o->code = "PRE_BAD";  // null non-nullable Ref -> Invalid
        pre2.create(std::move(o));
        CHECK(model.run_pre_transaction(pre2).status == CommitStatus::Invalid);
        // A well-behaved hook stops calling run_pre_transaction() here.
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
        CHECK(model.run_pre_transaction(pre).status == CommitStatus::Committed);
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
        CHECK(model.run_pre_transaction(pre).status == CommitStatus::Committed);
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
        CommitResult r = model.run_pre_transaction(pre);
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

// ---------------------------------------------------------------------------
// Post-commit hook (Model::set_post_commit)
// ---------------------------------------------------------------------------

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
    m.set_post_commit([&](Model&, const Transaction&, const CommitResult& r) { seen_status = r.status; });
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

// ---------------------------------------------------------------------------
// Transaction::id() -- the correlation key shared by all three hooks
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// Persistent secondary index
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// Views
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// Transaction-local semantics
// ---------------------------------------------------------------------------

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
// commit is rejected as Invalid rather than silently misresolving.
TEST(a_local_id_from_one_transaction_does_not_resolve_in_a_different_transaction) {
    Model m;
    Transaction txn1 = m.begin();
    auto a = std::make_unique<Account>();
    a->name = "A1";
    const Ref<Account> a_local =
        txn1.create(std::move(a));  // placeholder, only meaningful inside txn1

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

// With no pending remove() intents, estimate_changes() has nothing to
// estimate -- it's exactly pending_changes(), verbatim.
TEST(estimate_changes_with_no_pending_removes_matches_pending_changes_exactly) {
    Model m;
    const Ref<Account> a = make_account(m, "A1");
    Transaction txn = m.begin();
    auto o = std::make_unique<Order>();
    o->code = "O1";
    o->account = a;
    txn.create(std::move(o));
    txn.update(a)->balance = 99;

    const std::vector<Change> estimate = txn.estimate_changes();
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
TEST(estimate_changes_estimates_a_non_nullable_cascade_delete) {
    Model m;
    const Ref<Account> a = make_account(m, "A1");
    const Ref<Order> o = make_order(m, "O1", a);

    Transaction txn = m.begin();
    txn.remove(a);
    const std::vector<Change> estimate = txn.estimate_changes();

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
TEST(estimate_changes_estimates_a_nullable_cascade_null_as_updated_not_deleted) {
    Model m;
    const Ref<Account> a = make_account(m, "A1");
    const Ref<Order> parent = make_order(m, "P1", a);
    const Ref<Order> child = make_order(m, "C1", a, parent);

    Transaction txn = m.begin();
    txn.remove(parent);
    const std::vector<Change> estimate = txn.estimate_changes();

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
// without_a_data_race: estimate_changes() takes no lock at all (commit_mu_
// is unreachable from Transaction-building code -- invariant 10), so
// calling it repeatedly from one thread must never race with another
// thread hammering try_commit() concurrently. The real assertion here is
// TSan's (ctest --preset tsan).
TEST(estimate_changes_never_races_a_concurrently_hammering_try_commit) {
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
    for (int i = 0; i < 200; ++i) CHECK(!txn.estimate_changes().empty());

    stop = true;
    committer.join();
}

// When nothing else touches the model between building the estimate and
// actually committing, estimate_changes() and the changeset PreCommitFn
// sees are the SAME set of (id, kind, tag) triples. A pure remove() (no
// creates) keeps every id real on both sides, so this is an exact
// comparison, not just a size check.
TEST(estimate_changes_matches_the_pre_commit_hooks_changeset_when_nothing_else_happened) {
    Model m;
    const Ref<Account> a = make_account(m, "A1");
    const Ref<Order> o = make_order(m, "O1", a);
    (void)o;  // exists only so removing `a` has something to cascade into

    Transaction txn = m.begin();
    txn.remove(a);  // cascades to o
    const std::vector<Change> estimate = txn.estimate_changes();

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
TEST(estimate_changes_and_the_pre_commit_hooks_changeset_diverge_after_a_reference_changes) {
    Model m;
    const Ref<Account> a = make_account(m, "A1");
    const Ref<Account> b = make_account(m, "B1");
    const Ref<Order> o = make_order(m, "O1", a);  // o.account == a, for now

    Transaction txn = m.begin();  // base() still sees o.account == a
    txn.remove(a);
    const std::vector<Change> estimate = txn.estimate_changes();

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
    // delete, after it) disagree -- exactly the staleness estimate_changes()'s
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

// ---------------------------------------------------------------------------
// try_commit correctness: conflicts
// ---------------------------------------------------------------------------

// The baseline success path: a straightforward create commits, and the
// returned CommitResult's snapshot/changes reflect that publish.
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

// Two threads committing updates to DIFFERENT ids never conflict with
// each other -- object-write-set OCC only serializes overlapping writes,
// not all concurrent writers.
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

// Two transactions racing an update to the SAME id: the first commit
// wins, the second is rejected as Conflict/IdSetOverlap, and a fresh
// retry against the new base succeeds cleanly.
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

// A concurrent remove() and update() targeting the same id conflict via
// IdSetOverlap exactly like two updates would -- removal isn't a special
// case of the id-overlap check.
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

// The headline multi-writer guarantee: see the in-body comment. A
// transaction that never touched a since-deleted id, but references it
// via a brand-new object, still fails -- against the LATEST state, not base.
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

    // The target was alive at a_txn's base and died to a CONCURRENT commit:
    // classified as a Conflict (retryable contention), not Invalid.
    const CommitResult ra = m.try_commit(a_txn);
    CHECK(ra.status == CommitStatus::Conflict);
    CHECK(ra.conflict->reason == ConflictReason::RefIntegrity);
}

// A hub-and-spoke cascade (1 account, 20 dependent orders) kills exactly
// the expected count in one commit, confirming try_commit()'s deferred
// cascade resolution matches what an eager, synchronous BFS would do.
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

// prune_changelog()'s retention policy: entries accumulate while a
// Snapshot pins an old watermark, then collapse back down once that
// Snapshot drops and later commits move the watermark past them.
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
// Bulk load (begin_bulk() / commit_bulk())
// ---------------------------------------------------------------------------
//
// None of these tests hold a Snapshot or Transaction across a commit_bulk()
// call -- that's the exclusive-access precondition documented on
// Model::commit_bulk() itself, and violating it is undefined behavior, not a
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

    CommitResult r = m.commit_bulk(t);
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

    CommitResult r = m.commit_bulk(t);
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
    CommitResult r = m.commit_bulk(t);
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

    CommitResult r = m.commit_bulk(t);
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

    CommitResult r = m.commit_bulk(t);
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

    CommitResult r = m.commit_bulk(t);
    CHECK(r.status == CommitStatus::Invalid);
}

TEST(normal_transaction_and_find_by_key_work_correctly_after_a_bulk_load) {
    Model m;
    {
        BulkTransaction t = m.begin_bulk();
        auto acc = std::make_unique<Account>();
        acc->name = "C1";
        t.create(std::move(acc));
        CHECK(m.commit_bulk(t).status == CommitStatus::Committed);
    }

    const Ref<Account> a2 = make_account(m, "C2");  // an ordinary Transaction, on top of the bulk load
    update_field(m, a2, [](Account* a) { a->balance = 42; });

    Snapshot s = m.snapshot();
    CHECK(s.find_by_key<&Account::name>("C1") != nullptr);
    CHECK(s.find_by_key<&Account::name>("C2") != nullptr);
    CHECK_EQ(s.find(a2)->balance, 42);
    std::size_t count = 0;
    s.for_each<Account>([&](const Account&) { ++count; });
    CHECK_EQ(count, std::size_t{2});
}

// The reverse index (referrers_) has to come out of commit_bulk()'s no-log
// installation path in exactly the state cascade delete expects -- this
// exercises that by cascading a non-nullable ref and nulling a nullable one,
// entirely on bulk-loaded data.
TEST(cascade_delete_works_correctly_on_bulk_loaded_data) {
    Model m;
    BulkTransaction t = m.begin_bulk();
    auto acc = std::make_unique<Account>();
    acc->name = "Hub";
    const Ref<Account> a = t.create(std::move(acc));
    auto other = std::make_unique<Account>();  // unrelated -- keeps grandchild's own
    other->name = "Other";                    // non-nullable Ref<Account> satisfied after
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

    CommitResult r = m.commit_bulk(t);
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

// ---------------------------------------------------------------------------
// Concurrency stress
// ---------------------------------------------------------------------------

// The project's primary concurrency invariant test (see CLAUDE.md):
// many writer threads racing try_commit() on overlapping state, plus
// readers resolving Refs throughout, must never corrupt referrers_ or
// publish a dangling Ref -- extend this one rather than adding a new
// writer-only stress test where the scenario is genuinely the same.
TEST(
    concurrent_stress_many_writer_threads_hammering_try_commit_never_corrupts_referrers_or_leaks_a_dangling_ref) {
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
    std::atomic<std::uint64_t> accessor_calls{0};

    // A live Subscription for the whole run: proves push()/collapse() (the
    // REAL per-commit path every writer's publish_now() drives, not the
    // synchronous manual calls the dedicated Subscription tests use) survives
    // sustained concurrent commit pressure, with nobody draining it -- forcing
    // repeated overflow/coalescing for the duration of the run.
    auto sub = m.subscribe(/*queue_depth=*/4);
    (void)sub;  // never drained on purpose -- see the comment above

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

    // Exercises every commit_mu_/ver_mu_/reap_mu_-touching PUBLIC accessor
    // from a thread that is neither a reader nor a writer, concurrently with
    // both -- every other test in this file calls these only before or
    // after a concurrent section, never during one. This is exactly the
    // kind of multi-entry-point pressure that would expose a lock-order
    // violation (commit_mu_ -> ver_mu_ -> reap_mu_, never reversed -- see
    // CLAUDE.md invariant 10) that a single-caller test cannot.
    auto accessor_hammer = [&] {
        while (!stop.load(std::memory_order_relaxed)) {
            (void)m.retired_pending();
            (void)m.current_version();
            (void)m.exhausted_slots();
            (void)m.wait_for_reclamation();
            accessor_calls.fetch_add(1, std::memory_order_relaxed);
        }
    };

    std::thread r1(reader), r2(reader);
    while (readers_ready.load(std::memory_order_relaxed) < 2) std::this_thread::yield();

    std::thread w1([&] { writer(11); }), w2([&] { writer(22); }), w3([&] { writer(33); });
    std::thread acc(accessor_hammer);

    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    stop = true;
    w1.join();
    w2.join();
    w3.join();
    r1.join();
    r2.join();
    acc.join();

    m.shutdown();  // closes `sub` -- safe now that every writer/reader thread has stopped

    CHECK(snaps.load() > 0);
    CHECK(resolved.load() > 0);
    CHECK(accessor_calls.load() > 0);

    Snapshot final_s = m.snapshot();
    final_s.for_each<Order>(
        [&](const Order& o) { CHECK(final_s.resolve(o.account).id == o.account.raw()); });
}

// The intersection the individual hook tests above never cover: every hook
// added this session (PreTransactionsFn, PreCommitFn, PostCommitFn)
// installed globally, doing REAL work, at the same time as the flagship
// stress test's actual cascade-producing multi-writer + reader load. Each
// hook has its own dedicated correctness test elsewhere, using simple,
// isolated creates -- this is the only place a hook's own side effects run
// while a cascade from a DIFFERENT thread is actually in flight. Extend
// this one (not the flagship test above, which stays hook-free so it keeps
// testing the base cascade/referrers_ invariant in isolation) if you add a
// fourth hook.
TEST(hooks_survive_heavy_concurrent_cascade_churn_without_corruption) {
    Model m;
    std::vector<Ref<Account>> accounts;
    for (int i = 0; i < 4; ++i) accounts.push_back(make_account(m, "A" + std::to_string(i)));
    std::vector<Ref<Order>> seed_orders;
    for (int i = 0; i < 10; ++i)
        seed_orders.push_back(make_order(m, "O" + std::to_string(i), accounts[i % 4]));

    std::atomic<std::uint64_t> pre_transactions_ran{0};
    std::atomic<std::uint64_t> pre_commit_ran{0};
    std::atomic<std::uint64_t> post_commit_ran{0};
    std::atomic<std::uint64_t> post_commit_committed{0};
    std::atomic<int> next_pretxn_id{0};

    // Every hook does REAL work, not a no-op -- a no-op hook can't expose an
    // interaction bug with a cascade in flight on another thread.
    m.set_pre_transactions([&](Model& model, const Transaction&) {
        pre_transactions_ran.fetch_add(1, std::memory_order_relaxed);
        Transaction pre = model.begin();
        auto a = std::make_unique<Account>();
        a->name = "PRE" + std::to_string(next_pretxn_id.fetch_add(1, std::memory_order_relaxed));
        pre.create(std::move(a));
        (void)model.run_pre_transaction(pre);  // a fresh create can't conflict; always Committed
    });
    m.set_pre_commit([&](Model&, const Transaction&, const std::vector<Change>&) {
        pre_commit_ran.fetch_add(1, std::memory_order_relaxed);
        return true;  // this test is about survival under load, not veto logic
    });
    m.set_post_commit([&](Model&, const Transaction&, const CommitResult& r) {
        post_commit_ran.fetch_add(1, std::memory_order_relaxed);
        if (r.status == CommitStatus::Committed)
            post_commit_committed.fetch_add(1, std::memory_order_relaxed);
    });

    std::atomic<bool> stop{false};
    std::atomic<std::uint64_t> resolved{0};
    std::atomic<int> readers_ready{0};
    std::atomic<int> next_id{100};

    auto reader = [&] {
        bool signaled = false;
        while (!stop.load(std::memory_order_relaxed)) {
            Snapshot s = m.snapshot();
            if (!signaled) {
                readers_ready.fetch_add(1, std::memory_order_relaxed);
                signaled = true;
            }
            s.for_each<Order>([&](const Order& o) {
                const Account& acct = s.resolve(o.account);  // must never dangle
                if (acct.name.empty()) ++g_failures;
                resolved.fetch_add(1, std::memory_order_relaxed);
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
                o->code = "W" + std::to_string(next_id.fetch_add(1, std::memory_order_relaxed));
                o->account = accounts[rng() % accounts.size()];
                txn.create(std::move(o));
            } else if (roll < 80) {
                if (Order* o = txn.update(seed_orders[rng() % seed_orders.size()]))
                    o->qty = static_cast<std::int64_t>(rng() % 50);
            } else {
                txn.remove(seed_orders[rng() % seed_orders.size()]);
            }
            m.try_commit(txn);  // any status is fine; just don't corrupt anything
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

    m.set_pre_transactions({});
    m.set_pre_commit({});
    m.set_post_commit({});

    CHECK(resolved.load() > 0);
    CHECK(pre_transactions_ran.load() > 0);
    CHECK(pre_commit_ran.load() > 0);
    CHECK(post_commit_ran.load() > 0);
    // Every attempt that ran pre_commit_ (i.e. reached apply) also ran
    // post_commit_ afterward -- the two must stay in lockstep even under
    // heavy contention, since post_commit_ fires for every non-empty
    // attempt regardless of its outcome.
    CHECK(post_commit_ran.load() >= pre_commit_ran.load());
    CHECK(post_commit_ran.load() >= post_commit_committed.load());

    Snapshot final_s = m.snapshot();
    final_s.for_each<Order>(
        [&](const Order& o) { CHECK(final_s.resolve(o.account).id == o.account.raw()); });
}

// Up to N threads mixed reader/writer — kThreads = 8, split 4 readers / 4 writers, all launched and
// alive concurrently for the run window (same pattern as the existing stress test, just with a
// configurable thread count and roughly even role split instead of a fixed 2+3). 10,000–100,000
// total objects — seeded 100 Accounts + 25,000 Records (25,100 total), with an explicit CHECK
// asserting the count falls in [10000, 100000] both right after seeding and again at the end (after
// the concurrent churn of creates/removes). 5 fields, 2 references — new type Record: label
// (string), value (int64), flag (bool), owner (Ref<Account>, non-nullable — cascade-delete edge),
// related (Opt<Record>, nullable, self-referential — cascade-null edge). Writers touch all five on
// every create, and reassign both references plus a plain field on every update.
TEST(concurrent_stress_mixed_readers_and_writers_at_scale_across_five_fields_two_of_them_refs) {
    Model m;

    // ---- seed: a corpus in [10000, 100000] objects, well before any thread
    // starts -- large enough that the concurrent phase below is genuinely
    // contending over substantial state, not a handful of hot ids. ---------
    constexpr int kAccounts = 100;
    constexpr int kRecords = 25000;
    std::vector<Ref<Account>> accounts;
    {
        Transaction seed = m.begin();
        std::vector<Ref<Account>> local;
        for (int i = 0; i < kAccounts; ++i) {
            auto a = std::make_unique<Account>();
            a->name = "A" + std::to_string(i);
            local.push_back(seed.create(std::move(a)));
        }
        const CommitResult res = commit_ok(m, seed);
        for (auto& r : local) accounts.push_back(res.to_real(r));
    }

    std::vector<Ref<Record>> records;
    for (int i = 0; i < kRecords; i += 1000) {
        Transaction txn = m.begin();
        std::vector<Ref<Record>> local;
        for (int j = 0; j < 1000 && i + j < kRecords; ++j) {
            auto r = std::make_unique<Record>();
            r->label = "R" + std::to_string(i + j);
            r->value = i + j;
            r->flag = (i + j) % 2 == 0;
            r->owner = accounts[static_cast<std::size_t>((i + j) % accounts.size())];
            local.push_back(txn.create(std::move(r)));
        }
        const CommitResult res = commit_ok(m, txn);
        for (auto& r : local) records.push_back(res.to_real(r));
    }
    const std::size_t seeded = accounts.size() + records.size();
    CHECK(seeded >= std::size_t{10000} && seeded <= std::size_t{100000});

    // ---- concurrent phase: up to kThreads alive at once, mixed roles -----
    constexpr int kThreads = 8;
    constexpr int kReaders = kThreads / 2;
    constexpr int kWriters = kThreads - kReaders;

    std::atomic<bool> stop{false};
    std::atomic<std::uint64_t> snaps{0};
    std::atomic<std::uint64_t> resolved{0};
    std::atomic<int> readers_ready{0};
    std::atomic<int> next_id{1'000'000};

    // Readers scan every Record (both ref fields must always resolve/never
    // crash) AND cross-check the two "who points at this Account?" answers
    // against a fixed target -- the O(#Records) scan (find_referrers) and
    // the O(log n + matches) index (find_cached_referrers) -- against the
    // SAME frozen Snapshot, so they must agree exactly even while other
    // threads race concurrent commits underneath.
    auto reader = [&] {
        bool signaled = false;
        while (!stop.load(std::memory_order_relaxed)) {
            Snapshot s = m.snapshot();
            ++snaps;
            if (!signaled) {
                readers_ready.fetch_add(1, std::memory_order_relaxed);
                signaled = true;
            }
            s.for_each<Record>([&](const Record& r) {
                const Account& owner = s.resolve(r.owner);  // non-nullable: must never dangle
                if (owner.name.empty()) ++g_failures;
                ++resolved;
                (void)s.resolve(r.related);  // nullable: null or a live Record, never UB
            });

            auto scan = s.find_referrers<&Record::owner>(accounts[0]);
            auto idx = s.find_cached_referrers<&Record::owner>(accounts[0]);
            std::sort(scan.begin(), scan.end());
            std::sort(idx.begin(), idx.end());
            CHECK(scan == idx);
        }
    };

    // Writers touch all 5 fields: create sets every field at once (including
    // both refs); update reassigns owner (cascade-edge move) and related
    // (nullable edge move or clear) independently of the plain fields;
    // remove exercises both the cascade-delete (an owner Account dying takes
    // its Records with it -- not attempted here directly, but a removed
    // Record can itself have been a cascade victim) and cascade-null paths
    // (a removed Record that other Records' `related` pointed at).
    auto writer = [&](unsigned seed) {
        std::mt19937 rng(seed);
        while (!stop.load(std::memory_order_relaxed)) {
            Transaction txn = m.begin();
            const int roll = static_cast<int>(rng() % 100);
            if (roll < 35) {
                auto r = std::make_unique<Record>();
                r->label = "W" + std::to_string(next_id.fetch_add(1));
                r->value = static_cast<std::int64_t>(rng() % 1000);
                r->flag = (rng() % 2) == 0;
                r->owner = accounts[rng() % accounts.size()];
                if (rng() % 3 == 0) {
                    if (const Ref<Record> rel = pick_live(txn, records, rng)) r->related = rel;
                }
                txn.create(std::move(r));
            } else if (roll < 80) {
                if (Record* r = txn.update(pick_live(txn, records, rng))) {
                    r->value = static_cast<std::int64_t>(rng() % 1000);
                    r->flag = !r->flag;
                    r->owner = accounts[rng() % accounts.size()];
                    if (rng() % 4 == 0)
                        r->related.reset();
                    else if (const Ref<Record> rel = pick_live(txn, records, rng))
                        r->related = rel;
                }
            } else {
                if (const Ref<Record> r = pick_live(txn, records, rng)) txn.remove(r);
            }
            m.try_commit(txn);  // conflicts are expected and fine; just don't corrupt anything
        }
    };

    std::vector<std::thread> pool;
    for (int i = 0; i < kReaders; ++i) pool.emplace_back(reader);
    while (readers_ready.load(std::memory_order_relaxed) < kReaders) std::this_thread::yield();
    for (int i = 0; i < kWriters; ++i)
        pool.emplace_back([&, i] { writer(static_cast<unsigned>(700 + i)); });

    std::this_thread::sleep_for(std::chrono::milliseconds(300));
    stop = true;
    for (auto& t : pool) t.join();

    CHECK(snaps.load() > 0);
    CHECK(resolved.load() > 0);

    // ---- final consistency: every live Record's non-nullable ref resolves,
    // and the two "who points at this Account?" answers still agree for
    // EVERY account, not just the one readers happened to poll. ------------
    Snapshot final_s = m.snapshot();
    const std::size_t final_size = final_s.size();
    CHECK(final_size >= std::size_t{10000} && final_size <= std::size_t{100000});
    final_s.for_each<Record>(
        [&](const Record& r) { CHECK(final_s.resolve(r.owner).id == r.owner.raw()); });
    for (const Ref<Account>& a : accounts) {
        auto scan = final_s.find_referrers<&Record::owner>(a);
        auto idx = final_s.find_cached_referrers<&Record::owner>(a);
        std::sort(scan.begin(), scan.end());
        std::sort(idx.begin(), idx.end());
        CHECK(scan == idx);
    }
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
