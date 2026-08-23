// Exercises the same breadth of Snapshot/Transaction/Model/BulkTransaction/
// View surface that the generated example/types_extern.h explicitly
// instantiates for Account and Order -- both to prove the technique
// produces a working binary (this runs under CTest, same as demo.cpp) and
// to give a realistic file to measure. types_extern.h/.cpp aren't checked
// in: CMakeLists.txt's "extern_template_demo" section generates them from
// example/types.h via scripts/gen_extern_templates.py on every build (see
// that script's docstring for the mechanism and why full coverage instead
// of a hand-curated list).
//
// Measured (this file, gcc -O2 -g, ccache disabled, min of 8 runs):
//   including example/types.h directly (implicit instantiation): ~2.8s
//   including generated example/types_extern.h (this file's setup): ~1.2s
// About a 57% reduction for a TU that genuinely touches this much of the
// API. Even a minimal TU that only calls create/update/commit/resolve --
// most of this project's other example/bench files -- now sees a real cut
// too (~1.5s implicit vs. ~1.0s extern, min of 5 runs): types_extern.h also
// carries `extern template class Object<Account>`/`Object<Order>` (the CRTP
// base every user type derives from -- see the generator script's
// docstring), which pays off the instant a TU constructs one at all, not
// just for TUs that call the specific Snapshot/Transaction/Model/
// BulkTransaction/View entry points below. (Transaction/BulkTransaction's
// non-template raw() helpers were separately moved into src/model.cpp --
// that's not an extern-template effect, and it benefits implicit-
// instantiation TUs too, which is part of why the implicit number above
// also dropped.) The remaining floor for a TU that constructs an
// Account/Order at all is Account/Order's OWN vtable (no key function,
// since neither declares a virtual of its own) -- fixable the same way,
// but only by giving every user type an out-of-line destructor, which
// changes example/types.h's "just derive from Object<T>" contract for a
// few hundred bytes of payoff; deliberately not done.
//
// Generating full coverage (every structurally reachable entry point, not
// just ones this file happens to call) trades a measured +22% object-file
// size in the generated types_extern.cpp for a guarantee a hand-curated
// list can't make: a field added to types.h can never silently fall back
// to implicit instantiation because someone forgot to update a checked-in
// list. Fine for a demo binary; see the generator script's docstring
// before wiring the same tradeoff into a real project's real types.

#include "example/types_extern.h"

#include <cassert>
#include <cstdio>

using namespace example;
using namespace model;

int main() {
    Model m;
    Transaction txn = m.begin();

    Ref<Account> acct = txn.create(std::make_unique<Account>());
    if (Account* a = txn.update(acct)) {
        a->name = "Widgets Inc";
        a->balance = 100;
    }
    Ref<Order> ord = txn.create(std::make_unique<Order>());
    if (Order* o = txn.update(ord)) {
        o->code = "O1";
        o->account = acct;
        o->account_scan = acct;
        o->qty = 5;
        o->qty_scan = 5;
    }

    assert(txn.exists(acct));
    assert(txn.peek(acct));
    assert(txn.peek_as<Account>(acct.id()));
    assert(!txn.peek_before<Order>(ord.id()));  // fresh create -- no "before" value yet

    CommitResult r = m.try_commit(txn);
    assert(r.status == CommitStatus::Committed);
    acct = r.to_real(acct);
    ord = r.to_real(ord);
    Opt<Order> maybe_ord = ord;
    maybe_ord = r.to_real(maybe_ord);  // Opt<T> overload -- passes through, not itself local here
    assert(maybe_ord && maybe_ord.id() == ord.id());

    Snapshot s = m.snapshot();
    const Account& a2 = s.resolve(acct);
    assert(a2.name == "Widgets Inc");
    const Order* o2 = s.find<Order>(ord);
    assert(o2 && !s.resolve(o2->parent));  // never set -- resolve(Opt<>) is null, not dangling

    assert(s.find_by_key<&Account::name>("Widgets Inc") == &a2);
    assert(s.view_by_key<&Account::name>("Widgets Inc"));
    // Account::name is tagged LookupType::Scan in define_fields() --
    // exercises find_by_field's scan-fallback branch.
    assert(s.find_by_field<&Account::name>("Widgets Inc").size() == 1);
    assert(s.view_by_field<&Account::name>("Widgets Inc").size() == 1);

    assert(s.find_by_key<&Order::computed_key>("ord:O1") == o2);
    assert(s.view_by_key<&Order::computed_key>("ord:O1"));
    // Order::qty/computed_key are tagged LookupType::Exact in define_fields()
    // -- exercises find_by_field's cache-hit branch.
    assert(s.find_by_field<&Order::qty>(5).size() == 1);
    assert(s.find_by_field<&Order::computed_key>("ord:O1").size() == 1);
    assert(s.view_by_field<&Order::qty>(5).size() == 1);
    // Order::qty_scan/account_scan are scan-only twins of qty/account, kept
    // at identical values -- exercises find_by_field/find_referrers' scan-
    // fallback branch on data shaped just like the cache-hit case above.
    assert(s.find_by_field<&Order::qty_scan>(5).size() == 1);
    assert(s.view_by_field<&Order::qty_scan>(5).size() == 1);
    assert(s.find_referrers<&Order::account>(acct).size() == 1);
    assert(s.view_referrers<&Order::account>(acct).size() == 1);
    assert(s.find_referrers<&Order::account_scan>(acct).size() == 1);
    assert(s.view_referrers<&Order::account_scan>(acct).size() == 1);

    // range_* siblings: same lookup families, range-for instead of a vector
    // (see types_extern.h's coverage -- these are template<auto Field>/
    // template<class T>, no Pred/F, so (unlike for_each_*/all_of_*) they're
    // structurally enumerable and belong in the generated extern coverage).
    {
        int n = 0;
        for (const Account& x : s.range_by_field<&Account::name>("Widgets Inc")) { (void)x; ++n; }
        assert(n == 1);
        n = 0;
        for (View<Account> x : s.range_view_by_field<&Account::name>("Widgets Inc")) { (void)x; ++n; }
        assert(n == 1);
    }
    {
        int n = 0;
        for (const Order& x : s.range_by_field<&Order::qty>(5)) { (void)x; ++n; }
        assert(n == 1);
        n = 0;
        for (const Order& x : s.range_by_field<&Order::computed_key>("ord:O1")) { (void)x; ++n; }
        assert(n == 1);
        n = 0;
        for (View<Order> x : s.range_view_by_field<&Order::qty>(5)) { (void)x; ++n; }
        assert(n == 1);
        n = 0;
        for (const Order& x : s.range_by_field<&Order::qty_scan>(5)) { (void)x; ++n; }
        assert(n == 1);
        n = 0;
        for (View<Order> x : s.range_view_by_field<&Order::qty_scan>(5)) { (void)x; ++n; }
        assert(n == 1);
        n = 0;
        for (const Order& x : s.range_referrers<&Order::account>(acct)) { (void)x; ++n; }
        assert(n == 1);
        n = 0;
        for (View<Order> x : s.range_view_referrers<&Order::account>(acct)) { (void)x; ++n; }
        assert(n == 1);
        n = 0;
        for (const Order& x : s.range_referrers<&Order::account_scan>(acct)) { (void)x; ++n; }
        assert(n == 1);
        n = 0;
        for (View<Order> x : s.range_view_referrers<&Order::account_scan>(acct)) { (void)x; ++n; }
        assert(n == 1);
    }
    {
        int n = 0;
        for (const Account& x : s.range<Account>()) { (void)x; ++n; }
        assert(n == 1);
        n = 0;
        for (View<Account> x : s.range_view<Account>()) { (void)x; ++n; }
        assert(n == 1);
        n = 0;
        for (const Order& x : s.range<Order>()) { (void)x; ++n; }
        assert(n == 1);
        n = 0;
        for (View<Order> x : s.range_view<Order>()) { (void)x; ++n; }
        assert(n == 1);
    }

    if (const Account* ap = s.find(acct)) {
        View<Account> av = s.view(*ap);
        assert(av.find_referrers<&Order::account>().size() == 1);
    }
    if (auto ov = s.view(ord)) {
        View<Account> linked = (*ov)[&Order::account];
        assert(linked->name == "Widgets Inc");
        assert(!(*ov)[&Order::parent]);
    }

    // BulkTransaction: same Account/Order instantiations, separate entry
    // points (see types_extern.h's "BulkTransaction" section).
    Model m2;
    BulkTransaction bulk = m2.begin_bulk();
    Ref<Account> bacct = bulk.create(std::make_unique<Account>());
    if (Account* ba = bulk.update(bacct)) ba->name = "Bulk Inc";
    Ref<Order> bord = bulk.create(std::make_unique<Order>());
    if (Order* bo = bulk.update(bord)) {
        bo->code = "B1";
        bo->account = bacct;
    }
    CommitResult br = m2.commit_bulk_without_undo(bulk);
    assert(br.status == CommitStatus::Committed);

    std::puts("extern_template_demo: ok");
    return 0;
}
