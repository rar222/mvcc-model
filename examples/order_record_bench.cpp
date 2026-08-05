// Stand-alone bench for a type that exercises EVERY commit_mu_-protected
// index, not just by_type_ -- the natural next question after basic_record_
// bench.cpp (which profiled BasicRecord, a type with nothing declared).
// Order (include/example/types.h) touches:
//   by_type_              -- every object, unconditionally (see basic_record_bench.cpp)
//   by_field_              -- computed_key (define_keys())
//   by_cached_field_       -- qty, computed_key (define_fields(), LookupType::Cache)
//   by_cached_reference_   -- account (define_references(), LookupType::Cache)
//   referrers_             -- account (non-nullable), parent (nullable)
// Same shape as large_txn_bench.cpp's bench_bulk_create, pulled into its own
// binary with basic_record_bench.cpp's build/commit phase split and argv
// count override, so it can be profiled (gprof/callgrind) the same way.
//
// Usage: ./order_record_bench [n_orders]   (default 150,000, matching
// large_txn_bench.cpp's own kOrders)

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <string>
#include <vector>

#include "example/types.h"
#include "model/model.h"

using namespace model;
using namespace example;
using Clock = std::chrono::steady_clock;

namespace {

double ms_since(Clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

}  // namespace

int main(int argc, char** argv) {
    const int kOrders = argc > 1 ? std::atoi(argv[1]) : 150'000;
    const int kAccounts = std::max(1, kOrders / 75);  // ratio matches large_txn_bench.cpp

    Model m;

    const auto t_seed0 = Clock::now();
    std::vector<Ref<Account>> accounts;
    {
        Transaction txn = m.begin();
        for (int i = 0; i < kAccounts; ++i) {
            auto a = std::make_unique<Account>();
            a->name = "Acct" + std::to_string(i);
            accounts.push_back(txn.create(std::move(a)));
        }
        const CommitResult res = m.try_commit(txn);
        for (auto& r : accounts) r = res.to_real(r);
    }
    const double seed_ms = ms_since(t_seed0);

    const auto t_build0 = Clock::now();
    Transaction txn = m.begin();
    for (int i = 0; i < kOrders; ++i) {
        auto o = std::make_unique<Order>();
        o->code = "O" + std::to_string(i);
        o->account = accounts[static_cast<std::size_t>(i) % accounts.size()];
        o->qty = i % 50;
        txn.create(std::move(o));
    }
    const double build_ms = ms_since(t_build0);

    const auto t_commit0 = Clock::now();
    const CommitResult res = m.try_commit(txn);
    const double commit_ms = ms_since(t_commit0);

    std::printf("n_orders=%d n_accounts=%d\n", kOrders, kAccounts);
    std::printf("seed accounts                     = %9.2f ms\n", seed_ms);
    std::printf("build  (Order construction + create) = %9.2f ms  (%.1f ns/object)\n", build_ms,
                build_ms * 1e6 / kOrders);
    std::printf("commit (try_commit, all 4 indexes)   = %9.2f ms  (%.1f ns/object)\n", commit_ms,
                commit_ms * 1e6 / kOrders);
    std::printf("status=%s  size=%zu\n", res.status == CommitStatus::Committed ? "Committed" : "FAIL",
                m.snapshot().size());

    return res.status == CommitStatus::Committed ? 0 : 1;
}
