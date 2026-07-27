// Exercises the same breadth of Snapshot/Transaction/Model/BulkTransaction/
// View surface that types_extern.h explicitly instantiates for Account and
// Order -- both to prove the technique produces a working binary (this
// runs under CTest, same as demo.cpp) and to give a realistic file to
// measure. See types_extern.h's own comment for the mechanism.
//
// Measured (this file, gcc -O2 -g, ccache disabled, min of 8 runs):
//   including example/types.h directly (implicit instantiation): ~2.8s
//   including example/types_extern.h (this file's actual setup):  ~1.1s
// About a 59% reduction for a TU that genuinely touches this much of the
// API. Even a minimal TU that only calls create/update/commit/resolve --
// most of this project's other example/bench files -- now sees a real cut
// too (~1.5s implicit vs. ~1.0s extern, min of 5 runs): types_extern.h also
// carries `extern template class Object<Account>`/`Object<Order>` (the CRTP
// base every user type derives from -- see its own comment), which pays off
// the instant a TU constructs one at all, not just for TUs that call the
// specific Snapshot/Transaction/Model/BulkTransaction/View entry points
// below. (Transaction/BulkTransaction's non-template raw() helpers were
// separately moved into src/model.cpp -- that's not an extern-template
// effect, and it benefits implicit-instantiation TUs too, which is part of
// why the implicit number above also dropped.) The remaining floor for a
// TU that constructs an Account/Order at all is Account/Order's OWN vtable
// (no key function, since neither declares a virtual of its own) -- fixable
// the same way, but only by giving every user type an out-of-line
// destructor, which changes example/types.h's "just derive from Object<T>"
// contract for a few hundred bytes of payoff; deliberately not done.
// Don't reach for the entry-point-listing part of this pattern below the
// "genuinely touches this much of the API" bar; it's pure maintenance cost
// (a new Field used in a new TU that isn't added to types_extern.cpp just
// silently falls back to implicit instantiation, which is safe but easy to
// mistake for "not working").

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
        o->qty = 5;
    }

    assert(txn.exists(acct));
    assert(txn.peek(acct));
    assert(txn.peek_as<Account>(acct.raw()));
    assert(!txn.peek_before<Order>(ord.raw()));  // fresh create -- no "before" value yet

    CommitResult r = m.try_commit(txn);
    assert(r.status == CommitStatus::Committed);
    acct = r.to_real(acct);
    ord = r.to_real(ord);

    Snapshot s = m.snapshot();
    const Account& a2 = s.resolve(acct);
    assert(a2.name == "Widgets Inc");
    const Order* o2 = s.find<Order>(ord);
    assert(o2 && !s.resolve(o2->parent));  // never set -- resolve(Opt<>) is null, not dangling

    assert(s.find_by_key<&Account::name>("Widgets Inc") == &a2);
    assert(s.view_by_key<&Account::name>("Widgets Inc"));
    assert(s.find_by_scan_field<&Account::name>("Widgets Inc").size() == 1);
    assert(s.view_by_scan_field<&Account::name>("Widgets Inc").size() == 1);

    assert(s.find_by_key<&Order::computed_key>("ord:O1") == o2);
    assert(s.view_by_key<&Order::computed_key>("ord:O1"));
    assert(s.find_by_scan_field<&Order::qty>(5).size() == 1);
    assert(s.find_by_scan_field<&Order::computed_key>("ord:O1").size() == 1);
    assert(s.view_by_scan_field<&Order::qty>(5).size() == 1);
    assert(s.find_by_cached_field<&Order::qty>(5).size() == 1);
    assert(s.find_by_cached_field<&Order::computed_key>("ord:O1").size() == 1);
    assert(s.view_by_cached_field<&Order::qty>(5).size() == 1);
    assert(s.find_referrers<&Order::account>(acct).size() == 1);
    assert(s.find_referrers_view<&Order::account>(acct).size() == 1);
    assert(s.find_cached_referrers<&Order::account>(acct).size() == 1);
    assert(s.view_cached_referrers<&Order::account>(acct).size() == 1);

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
