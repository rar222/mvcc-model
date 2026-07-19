#include "example/types.h"
#include <cassert>
#include <chrono>
#include <cstdio>
#include <memory>
#include <string>
using namespace model; using namespace example;

// The tempting alternative: a View that owns its Snapshot by value, so it's
// storable and can't dangle. Every traversal step copies the Snapshot.
template <class T>
class FatView {
public:
    FatView(Snapshot s, const T& o) : s_(std::move(s)), o_(&o) {}
    const T* operator->() const { return o_; }
    template <class U> FatView<U> operator[](Ref<U> T::*f) const {
        return FatView<U>(s_, s_.resolve(o_->*f));      // copies Snapshot: 2 atomic bumps
    }
private:
    Snapshot s_;   // shared_ptr<const Root> + shared_ptr<Lease>
    const T* o_;
};

template <class F> double time_ms(int reps, F&& f) {
    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < reps; ++i) f();
    auto t1 = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::milli>(t1 - t0).count() / reps;
}

int main() {
    Model m;
    constexpr int kAccounts = 500, kOrders = 100000;

    Transaction txn = m.begin();
    std::vector<Ref<Account>> accts;
    for (int i = 0; i < kAccounts; ++i) {
        auto a = std::make_unique<Account>(); a->name = "A" + std::to_string(i);
        accts.push_back(txn.create(std::move(a)));
    }
    std::vector<Ref<Order>> ords;
    for (int i = 0; i < kOrders; ++i) {
        auto o = std::make_unique<Order>();
        o->code = "O" + std::to_string(i);
        o->account = accts[i % kAccounts];
        if (i > 0 && i % 3 == 0) o->parent = ords[i - 1];
        ords.push_back(txn.create(std::move(o)));
    }
    CommitResult res = m.try_commit(txn);
    assert(res.status == CommitStatus::Committed);
    std::printf("%d orders, %d accounts\n\n", kOrders, kAccounts);

    Snapshot s = m.snapshot();
    volatile std::int64_t sink = 0;

    double raw = time_ms(20, [&] {
        std::int64_t t = 0;
        s.for_each<Order>([&](const Order& o) { t += s.resolve(o.account).balance; });
        sink = t;
    });

    double view = time_ms(20, [&] {
        std::int64_t t = 0;
        s.for_each_view<Order>([&](View<Order> o) { t += (*o[&Order::account]).balance; });
        sink = t;
    });

    double fat = time_ms(20, [&] {
        std::int64_t t = 0;
        s.for_each<Order>([&](const Order& o) {
            FatView<Order> v(s, o);                   // Snapshot copy per object
            t += v[&Order::account]->balance;         // Snapshot copy per hop
        });
        sink = t;
    });

    std::printf("raw   s.resolve(o.account)        %7.2f ms   (baseline)\n", raw);
    std::printf("View  by pointer  (16 bytes)      %7.2f ms   %.2fx\n", view, view / raw);
    std::printf("View  by value    (shared_ptr)    %7.2f ms   %.2fx  <-- refcounts\n", fat, fat / raw);
    (void)sink;
    return 0;
}
