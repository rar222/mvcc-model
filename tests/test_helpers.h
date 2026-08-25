#pragma once
//
// Shared test fixtures (types + factory helpers) used across tests/test_*.cpp.
//
// The non-template helpers (commit_ok, make_account, make_widget, make_gadget,
// make_node, make_order, state_of) are declared here and defined ONCE in
// test_helpers.cpp -- not in an anonymous namespace. An anonymous-namespace
// definition here would give every including TU its own copy, and -Wall
// -Werror's -Wunused-function then fires in every TU that doesn't happen to
// call a given helper (most split files only use a handful of these) --
// exactly the same "each TU needs to see the SAME symbol" reasoning as
// test_harness.h's registry(). The fixture types (Widget/Gadget/Node/Record/
// Link) are ordinary types for the same reason: a function with external
// linkage can't have an anonymous-namespace type in its signature and still
// mean the same thing in every TU.
//
// Templates (update_field/remove_and_commit/pick_live) stay defined here,
// same as any template -- they get per-TU (vague) linkage regardless.

#include <cstdint>
#include <map>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "model/model.h"
#include "test_harness.h"
#include "test_types.h"

using namespace model;

// A local id returned by Transaction::create() is only meaningful until
// try_commit() returns (see CommitResult::to_real's doc comment). These
// helpers each own a single commit and hand back the REAL, post-commit ref,
// so most tests never have to think about the local/real distinction at all
// -- only the tests in the "Transaction-local semantics" section, which are
// specifically about that distinction, build multi-op Transactions by hand.

CommitResult commit_ok(Model& m, Transaction& txn);

Ref<Account> make_account(Model& m, const std::string& name, std::int64_t bal = 0);

/// A second, unrelated type whose computed_key() has no fixed prefix -- unlike
/// Order, which happens to prefix "ord:" by its own convention. Used to prove
/// the *model* enforces per-type keyspaces, rather than relying on every user
/// type picking disjoint prefixes.
class Widget final : public model::Object<Widget> {
public:
    std::string key;
    std::string computed_key() const { return key; }

    /// Diagnostic only -- see example::Account::to_string()'s doc comment
    /// for why `id` comes first.
    std::string to_string() const { return "Widget{id=" + id.to_string() + ", key=" + key + "}"; }

    // no outgoing references, so no define_references() to declare
    template <class Self>
    static void define_keys(Self& s, const model::FieldKeyReader& v) {
        v.key<&Widget::computed_key>(s.computed_key());
    }
};

Ref<Widget> make_widget(Model& m, const std::string& key);

/// A type that opts three fields into fast lookup via define_keys(): a plain
/// string, a plain arithmetic field, and computed_key() itself (a nullary
/// const method, not a data member) -- proving a computed key can be indexed
/// and looked up the exact same way as a stored one.
class Gadget final : public model::Object<Gadget> {
public:
    std::string label;
    std::int64_t serial = 0;

    std::string computed_key() const { return "gad:" + label; }

    /// Diagnostic only -- see example::Account::to_string()'s doc comment
    /// for why `id` comes first.
    std::string to_string() const {
        return "Gadget{id=" + id.to_string() + ", label=" + label + ", serial=" + std::to_string(serial) + "}";
    }

    // no outgoing references, so no define_references() to declare

    template <class Self>
    static void define_keys(Self& s, const model::FieldKeyReader& v) {
        v.key<&Gadget::label>(s.label);
        v.key<&Gadget::serial>(s.serial);
        v.key<&Gadget::computed_key>(s.computed_key());
    }
};

Ref<Gadget> make_gadget(Model& m, const std::string& label, std::int64_t serial);

/// Self-referential, with a CACHED nullable reference. Order::parent (the
/// shared demo type's analogous field) is deliberately tagged
/// RefLookupType::Scan to demonstrate the "index only what's worth it"
/// tradeoff -- so testing the cascade-NULL maintenance path of
/// Model::reconcile_cached_references needs a field that actually IS
/// RefLookupType::Exact and nullable. Node exists purely for that.
class Node final : public model::Object<Node> {
public:
    std::string label;
    model::Opt<Node> parent;

    /// Diagnostic only -- see example::Account::to_string()'s doc comment
    /// for why `id` comes first and parent prints as raw Id::to_string()
    /// instead of being resolved.
    std::string to_string() const {
        return "Node{id=" + id.to_string() + ", label=" + label +
               ", parent=" + (parent ? parent.id().to_string() : "null") + "}";
    }

    template <class Self, class V>
    static void define_references(Self& s, V&& v) {
        v(model::field_tag<&Node::parent>(), s.parent, model::RefLookupType::Exact, "parent");
    }
};

Ref<Node> make_node(Model& m, const std::string& label, Opt<Node> parent = Opt<Node>{});

Ref<Order> make_order(Model& m, const std::string& code, Ref<Account> account,
                      Opt<Order> parent = Opt<Order>{}, std::int64_t qty = 1);

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
std::string state_of(Model& m);

/// A wider record type for the large-scale mixed-thread stress test below:
/// 5 fields, 2 of them references -- owner (Ref<Account>, non-nullable: the
/// cascade-delete path) and related (Opt<Record>, nullable, self-referential:
/// the cascade-null path) -- so every write there exercises reconcile_
/// referrer_edges across two DIFFERENT edges at once, not just one. owner is
/// tagged RefLookupType::Exact, so the same run stress-tests Root::by_cached_
/// reference concurrently for the first time -- every prior cached-
/// reference test was single-threaded.
class Record final : public model::Object<Record> {
public:
    std::string label;
    std::int64_t value = 0;
    bool flag = false;
    model::Ref<Account> owner;
    model::Opt<Record> related;

    /// Scan-only twin of owner (see Order::qty_scan in test_types.h for the
    /// same pattern) -- Opt<>, not Ref<>, so leaving it unset (as every
    /// caller that doesn't care about it does) is never a validation error.
    /// Exists so find_referrers's scan-fallback branch can be exercised on
    /// Record concurrently, now that owner itself is cache-declared and
    /// find_referrers on it always takes the cache-hit branch.
    model::Opt<Account> owner_scan;

    /// Diagnostic only -- see example::Account::to_string()'s doc comment
    /// for why `id` comes first and owner/related print as raw
    /// Id::to_string() instead of being resolved.
    std::string to_string() const {
        return "Record{id=" + id.to_string() + ", label=" + label + ", value=" + std::to_string(value) +
               ", flag=" + (flag ? "true" : "false") + ", owner=" + owner.id().to_string() +
               ", related=" + (related ? related.id().to_string() : "null") + "}";
    }

    template <class Self, class V>
    static void define_references(Self& s, V&& v) {
        v(model::field_tag<&Record::owner>(), s.owner, model::RefLookupType::Exact, "owner");
        v(model::field_tag<&Record::related>(), s.related, model::RefLookupType::Scan, "related");
        v(model::field_tag<&Record::owner_scan>(), s.owner_scan, model::RefLookupType::Scan,
          "owner_scan");
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

/// The only way to express a NON-nullable cycle: no demo type carries a
/// Ref<> to its own type (Order::parent is deliberately Opt<>). With the
/// pre-mint pass, a Ref<Link> cycle is representable -- which also means
/// the non-nullable reference graph is NOT guaranteed to be a DAG, and a
/// cascade entering the cycle kills every link in it. Shared between
/// test_cascade.cpp (pre-mint-pass tests) and test_bulk_load.cpp (which
/// reuses it for the same self-loop/cycle shape via commit_bulk_without_undo()).
class Link final : public model::Object<Link> {
public:
    std::string label;
    model::Ref<Link> next;

    /// Diagnostic only -- see example::Account::to_string()'s doc comment
    /// for why `id` comes first and next prints as raw Id::to_string()
    /// instead of being resolved.
    std::string to_string() const {
        return "Link{id=" + id.to_string() + ", label=" + label + ", next=" + next.id().to_string() + "}";
    }

    template <class Self, class V>
    static void define_references(Self& s, V&& v) {
        v(model::field_tag<&Link::next>(), s.next, model::RefLookupType::Scan, "next");
    }
};
