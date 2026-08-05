#include "test_helpers.h"

CommitResult commit_ok(Model& m, Transaction& txn) {
    CommitResult r = m.try_commit(txn);
    CHECK(r.status == CommitStatus::Committed);
    return r;
}

Ref<Account> make_account(Model& m, const std::string& name, std::int64_t bal) {
    Transaction txn = m.begin();
    auto a = std::make_unique<Account>();
    a->name = name;
    a->balance = bal;
    const Ref<Account> local = txn.create(std::move(a));
    return commit_ok(m, txn).to_real(local);
}

Ref<Widget> make_widget(Model& m, const std::string& key) {
    Transaction txn = m.begin();
    auto w = std::make_unique<Widget>();
    w->key = key;
    const Ref<Widget> local = txn.create(std::move(w));
    return commit_ok(m, txn).to_real(local);
}

Ref<Gadget> make_gadget(Model& m, const std::string& label, std::int64_t serial) {
    Transaction txn = m.begin();
    auto g = std::make_unique<Gadget>();
    g->label = label;
    g->serial = serial;
    const Ref<Gadget> local = txn.create(std::move(g));
    return commit_ok(m, txn).to_real(local);
}

Ref<Node> make_node(Model& m, const std::string& label, Opt<Node> parent) {
    Transaction txn = m.begin();
    auto n = std::make_unique<Node>();
    n->label = label;
    n->parent = parent;
    const Ref<Node> local = txn.create(std::move(n));
    return commit_ok(m, txn).to_real(local);
}

Ref<Order> make_order(Model& m, const std::string& code, Ref<Account> account, Opt<Order> parent,
                      std::int64_t qty) {
    Transaction txn = m.begin();
    auto o = std::make_unique<Order>();
    o->code = code;
    o->account = account;
    o->parent = parent;
    o->qty = qty;
    // qty_scan is a scan-only twin of qty (see Order's own comment in
    // test_types.h) -- kept equal here so every existing caller of
    // make_order gets both the cache-hit and scan-fallback lookup families
    // exercisable on identical data without having to know the twin exists.
    // account_scan is NOT auto-set the same way: it's Opt<>, and giving an
    // object BOTH a non-nullable and a nullable ref to the identical target
    // trips a real (pre-existing, unrelated to this refactor) double-
    // processing bug in Model::remove_raw's cascade BFS -- a referrer with
    // one non-nullable edge forcing its own deletion and a nullable edge to
    // the same dying target gets a spurious cascade-null Update recorded
    // for it moments before its own cascade Delete. Setting account_scan
    // globally here would expose every OTHER test that cascade-deletes an
    // Account through make_order's Orders to that bug. Callers that
    // specifically need account_scan == account (to compare find_referrers'
    // cache-hit and scan-fallback branches on identical data) set it
    // themselves, and must avoid also cascade-deleting that same account
    // while it's set.
    o->qty_scan = qty;
    const Ref<Order> local = txn.create(std::move(o));
    return commit_ok(m, txn).to_real(local);
}

std::string state_of(Model& m) {
    Snapshot s = m.snapshot();
    std::string out = "Snapshot Version=" + std::to_string(s.version());
    s.for_each<Account>([&](const Account& a) { out.append(" "); out.append(a.to_string()); });
    s.for_each<Order>([&](const Order& o) { out.append(" "); out.append(o.to_string()); });
    return out;
}
