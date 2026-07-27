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
