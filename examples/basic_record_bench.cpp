// Stand-alone copy of tests/performance_tests.cpp's
// one_million_random_basic_records_in_a_single_transaction_without_undo,
// pulled out into its own executable so it can be profiled in isolation
// (gprof/callgrind) without the rest of the performance suite running
// alongside it and without needing -pg/-fno-omit-frame-pointer applied to
// every other test binary in the project.
//
// BasicRecord has NO define_keys(), NO define_references(), NO
// define_fields() -- deliberately nothing declared, so each_ref()/
// each_field_key()/each_field()/each_cached_reference() all fall back to
// Object<>'s own no-op defaults. That
// makes this the cheapest possible object shape for try_commit()'s apply
// phase: no ref to validate (invariant 1), no by_key_/by_cached_field_/
// by_cached_reference_ index entries to insert. The only per-object
// bookkeeping every create() ever pays regardless of declared fields is
// by_type_ (Root's "every Id of this type" set) and the raw slot
// allocation/chunk COW machinery -- so whatever this benchmark measures is
// close to the model's own per-object floor, not this type's overhead.
//
// Manually split into BUILD (random generation + txn.create(), which never
// touches commit_mu_ or any shared state -- invariant 10) and COMMIT
// (try_commit_without_undo(), the one serialized, shared-state-touching
// call) phases, printed separately -- the coarsest instrumentation
// available without editing library internals. Finer-grained "where inside
// apply does the time go" needs an actual profiler (gprof/callgrind); see
// the file this was copied from and the accompanying analysis for that
// breakdown.
//
// Usage: ./basic_record_bench [count]   (default 1,000,000)

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
    const CommitResult res = m.try_commit_without_undo(txn);
    const double commit_ms = ms_since(t_commit0);
#if defined(MODEL_BENCH_CALLGRIND)
    CALLGRIND_STOP_INSTRUMENTATION;
    CALLGRIND_DUMP_STATS;
#endif

    std::printf("count=%d\n", kCount);
    std::printf("build  (random gen + txn.create) = %9.2f ms  (%.1f ns/object)\n", build_ms,
                build_ms * 1e6 / kCount);
    std::printf("commit (try_commit_without_undo)  = %9.2f ms  (%.1f ns/object)\n", commit_ms,
                commit_ms * 1e6 / kCount);
    std::printf("total                             = %9.2f ms\n", build_ms + commit_ms);
    std::printf("status=%s  size=%zu\n", res.status == CommitStatus::Committed ? "Committed" : "FAIL",
                m.snapshot().size());

    return res.status == CommitStatus::Committed && m.snapshot().size() == static_cast<std::size_t>(kCount)
              ? 0
              : 1;
}
