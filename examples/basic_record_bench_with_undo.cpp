// Same as basic_record_bench.cpp, except the commit call is try_commit()
// (the undo-tracking path) instead of try_commit_without_undo() -- isolates
// what keeping an undo entry costs on top of the same per-object apply work
// (the per-object clone-capture in apply_create -- see keep_undo's own
// comment there -- and the pending_undo_/collapse_undo_actions/UndoEntry
// bookkeeping in apply_transaction_contents/publish_now, none of which
// try_commit_without_undo() pays for).
//
// BasicRecord has NO define_keys(), NO define_references(), NO
// define_scan_fields(), NO define_cached_fields() -- see basic_record_
// bench.cpp's own doc comment for why that makes it the cheapest possible
// object shape, isolating the model's own per-object floor rather than this
// type's field overhead.
//
// Usage: ./basic_record_bench_with_undo [count]   (default 1,000,000)

#include <chrono>
#include <cstdio>
#include <limits>
#include <memory>
#include <random>
#include <string>

#include "model/model.h"

#if defined(MODEL_BENCH_CALLGRIND)
#include <valgrind/callgrind.h>
#endif

using namespace model;
using Clock = std::chrono::steady_clock;

namespace {

double ms_since(Clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

class BasicRecord final : public model::Object<BasicRecord> {
public:
    int number = 0;
    std::string text;
};

}  // namespace

int main(int argc, char** argv) {
    const int kCount = argc > 1 ? std::atoi(argv[1]) : 1'000'000;
    Model m;

    // Fixed seed: a random-but-reproducible workload, not a fresh reshuffle
    // per run -- see the PERF_TEST this was copied from for why.
    std::mt19937 rng(12345);
    std::uniform_int_distribution<int> number_dist(std::numeric_limits<int>::min(),
                                                    std::numeric_limits<int>::max());
    std::uniform_int_distribution<int> len_dist(4, 32);
    std::uniform_int_distribution<int> char_dist('a', 'z');

    const auto t_build0 = Clock::now();
    Transaction txn = m.begin();
    for (int i = 0; i < kCount; ++i) {
        auto r = std::make_unique<BasicRecord>();
        r->number = number_dist(rng);
        r->text.resize(static_cast<std::size_t>(len_dist(rng)));
        for (char& c : r->text) c = static_cast<char>(char_dist(rng));
        txn.create(std::move(r));
    }
    const double build_ms = ms_since(t_build0);

#if defined(MODEL_BENCH_CALLGRIND)
    CALLGRIND_ZERO_STATS;
    CALLGRIND_START_INSTRUMENTATION;
#endif
    const auto t_commit0 = Clock::now();
    const CommitResult res = m.try_commit(txn);
    const double commit_ms = ms_since(t_commit0);
#if defined(MODEL_BENCH_CALLGRIND)
    CALLGRIND_STOP_INSTRUMENTATION;
    CALLGRIND_DUMP_STATS;
#endif

    std::printf("count=%d\n", kCount);
    std::printf("build  (random gen + txn.create) = %9.2f ms  (%.1f ns/object)\n", build_ms,
                build_ms * 1e6 / kCount);
    std::printf("commit (try_commit)               = %9.2f ms  (%.1f ns/object)\n", commit_ms,
                commit_ms * 1e6 / kCount);
    std::printf("total                             = %9.2f ms\n", build_ms + commit_ms);
    std::printf("status=%s  size=%zu\n", res.status == CommitStatus::Committed ? "Committed" : "FAIL",
                m.snapshot().size());

    return res.status == CommitStatus::Committed && m.snapshot().size() == static_cast<std::size_t>(kCount)
              ? 0
              : 1;
}
