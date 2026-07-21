// Performance test suite: turns this project's own documented complexity
// and memory claims (see CLAUDE.md, DESIGN.md, README.md, and the
// printed-only benchmarks in examples/commit_bench.cpp and
// examples/cached_reference_bench.cpp) into PASS/FAIL assertions across
// several data sizes -- so a regression that turns an O(log n) lookup into
// an O(n) scan, or a memcpy into a per-object deep copy, fails this suite
// instead of only showing up if a human happens to run a benchmark by hand
// and notices.
//
// Three kinds of coverage, per the request that motivated this file:
//   1. The claims themselves, swept across several data sizes.
//   2. Worst-case / degenerate data: the costs CLAUDE.md's own "Known
//      scope boundaries" section documents as unbounded (a hub object
//      with many referrers; unbounded cascade fan-out) -- demonstrated
//      and bounded, not silently accepted as "fine."
//   3. Object PAYLOAD size as its own axis, independent of object COUNT:
//      write cost (clone/copy) should scale with an object's own byte
//      size; read cost (find_by_key) should not.
//
// HOW PASS/FAIL WORKS. No test here asserts a fixed number of seconds.
// Each sweep checks the SHAPE of the curve: every step's timing is
// compared against the PREVIOUS step's measured timing, scaled by what
// the claimed complexity predicts for that step's size ratio (see
// check_scaling), with a +/-20% band by default. That makes every check
// independent of how fast the machine running it happens to be, and
// encodes the actual claim ("this grows like log n") rather than a
// wall-clock budget that a slower machine would fail and a silent
// O(log n)-to-O(n) regression would still pass.
//
// Three claims cannot hold a two-sided 20% band for reasons that are
// physical rather than algorithmic (cache behaviour, fixed per-commit
// overhead, O(log n) index maintenance). Each of those gets an explicitly
// widened tolerance with the measured numbers that justify it recorded at
// the definition -- see Tolerance below. None of them is a bare "it ran
// fast enough" check.
//
// Measurements use best-of-N (see best_of): noise can only push a
// wall-clock reading up, so the minimum across trials is the sample least
// corrupted by scheduling jitter and cold caches. A 20% band is only
// meaningful if its inputs are not themselves 50% noise.
//
// Timing bounds are asserted in the DEFAULT build only. Under asan/tsan
// the suite still runs and prints everything -- that is how this file's
// own fork/VmHWM harness gets checked for memory errors and races -- but
// complexity is not asserted there, because the sizes are 25x smaller and
// the instrumentation overhead is not proportional to algorithmic work.
// See kAssertTimings. Correctness assertions run everywhere.
//
// Dependency-free, same as tests/tests.cpp (see CLAUDE.md: "Don't add
// gtest/Catch2") -- a small self-contained harness, adapted here to print
// a timing line per measurement instead of just OK/FAIL.
//
// Run: ctest --preset default -R performance_tests --output-on-failure
//   or: ./build/default/performance_tests

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <functional>
#include <limits>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "model/model.h"
#include "test_types.h"

#if defined(__linux__)
#include <malloc.h>
#include <sys/wait.h>
#include <unistd.h>

#include <cstdlib>
#define PERF_HAS_MEMORY_SECTION 1
#else
#define PERF_HAS_MEMORY_SECTION 0
#endif

using namespace model;

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

struct Registrar {
    Registrar(const char* name, std::function<void()> fn) {
        registry().push_back({name, std::move(fn)});
    }
};

}  // namespace

#define PERF_TEST(name)                       \
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

namespace {

// ASan's and (especially) TSan's per-allocation and per-memory-access
// instrumentation make the FULL sizes below take minutes instead of
// seconds -- TSan alone did not finish this suite inside 5 minutes at full
// scale. `scaled()` shrinks every data-size constant under either
// sanitizer so the suite still exercises the same comparisons and the same
// relative shape, just fast enough to run there at all; the default build
// (this project's presets never build with NDEBUG, but also never build
// with a sanitizer by default) always runs the full, realistic sizes the
// claims are actually about.
#if defined(__SANITIZE_ADDRESS__) || defined(__SANITIZE_THREAD__)
constexpr bool kSlowSanitizedBuild = true;
#elif defined(__has_feature)
#if __has_feature(address_sanitizer) || __has_feature(thread_sanitizer)
constexpr bool kSlowSanitizedBuild = true;
#else
constexpr bool kSlowSanitizedBuild = false;
#endif
#else
constexpr bool kSlowSanitizedBuild = false;
#endif

int scaled(int n) {
    return kSlowSanitizedBuild ? std::max(20, n / 25) : n;
}

// Timing-derived bounds are asserted ONLY in the default build. Under a
// sanitizer this suite still runs every path (which is the point of
// running it there -- it is how the fork/VmHWM harness and the model code
// it drives get checked for memory errors and races), and still prints
// every measurement, but it does not assert complexity: the data sizes
// are 25x smaller (see scaled()) AND each operation carries 20-50x
// instrumentation overhead that is not proportional to its algorithmic
// work. Observed under TSan at those sizes: find_cached_referrers
// measured 8.59, 8.36 then 5.37 us -- i.e. it got FASTER as n grew,
// purely from instrumentation noise. Asserting a scaling band on that
// would be asserting the sanitizer's overhead profile, not the model's
// complexity. Correctness assertions (cascade killed the right objects,
// the index costs more memory than no index) are NOT gated and run
// everywhere.
constexpr bool kAssertTimings = !kSlowSanitizedBuild;

template <class F>
double time_ms(F&& f) {
    const auto t0 = std::chrono::steady_clock::now();
    f();
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

// Runs `f` (which returns its own elapsed time, in whatever unit) `trials`
// times and keeps the MINIMUM -- the standard way to pull a stable estimate
// out of a wall-clock microbenchmark: OS scheduling jitter, a cache miss, a
// GC-unrelated page fault, etc. can only ever push one particular trial's
// reading UP, never down, so the minimum across several trials is the
// measurement least corrupted by that kind of noise. Every scaling check
// below (check_scaling) depends on this: a +/-20% band around a
// theoretical prediction is only meaningful if the inputs feeding it
// aren't already +/-50% noise.
template <class F>
double best_of(int trials, F&& f) {
    double best = f();
    for (int t = 1; t < trials; ++t) {
        const double v = f();
        if (v < best) best = v;
    }
    return best;
}

// The claimed asymptotic order of an operation, for check_scaling() below.
//
// kNLogN is the one to reach for whenever an operation touches n objects
// and does per-object index maintenance, because every such write goes
// through a path-copying persistent map: n objects x O(log n) per object.
// Cascade delete is the case in this suite (see the cascade test) -- each
// victim drives by_type_ plus drop_field_keys / drop_cached_fields /
// drop_cached_references. Reach for kLinear only when the per-object work
// really is index-free (e.g. try_commit's spine copy, which is a flat run
// of shared_ptr copies).
//
// Bulk create is the other structurally-n-log-n operation here (n creates,
// each doing O(log n) persistent-map inserts), and it is deliberately NOT
// swept: measured across 4x steps it came out x11.3 then x5.7, i.e. 2.5x
// and 1.3x against the n log n prediction, because per-object cost climbs
// from 2.6 us to 10.5 us as the working set leaves cache. The asymptotic
// term is real but is not the dominant one over any range this suite can
// afford, so asserting a band on it would be asserting cache behaviour.
enum class GrowthOrder { kConstant, kLogN, kLinear, kNLogN };

double scaling_factor(GrowthOrder order, int n_prev, int n_cur) {
    switch (order) {
        case GrowthOrder::kConstant:
            return 1.0;
        case GrowthOrder::kLogN:
            return std::log(static_cast<double>(n_cur)) / std::log(static_cast<double>(n_prev));
        case GrowthOrder::kLinear:
            return static_cast<double>(n_cur) / static_cast<double>(n_prev);
        case GrowthOrder::kNLogN:
            return (static_cast<double>(n_cur) * std::log(static_cast<double>(n_cur))) /
                   (static_cast<double>(n_prev) * std::log(static_cast<double>(n_prev)));
    }
    return 1.0;  // unreachable
}

// How much slack to allow around the predicted growth. kDefaultBand -- the
// +/-20% two-sided band -- is the intended check and the one to reach for.
// Every widening below is backed by a measurement recorded here, not by a
// preference for green tests.
//
//   kSmallIndexedRead -- the O(log n) lookups. Their per-call cost is a
//     fraction of a microsecond, and a 2x size step only predicts a x1.07
//     change, so ordinary run-to-run jitter is a larger effect than the
//     signal: measured step factors ran x0.82-x1.18 across repeat runs
//     against that x1.07 prediction -- and, under concurrent machine
//     load, as low as x0.51 (the lookup measured FASTER at the larger
//     size, purely from frequency/cache variance). The band is therefore
//     CEILING-ONLY (lo_mult 0): the regression this suite hunts makes
//     lookups slower, never faster, so a lower edge on a sub-microsecond
//     flat read asserts CPU-frequency stability rather than complexity.
//     (Match counts are held near-constant across the sweep so this
//     measures index DEPTH; letting matches grow with n instead put these
//     readings anywhere from 1.5 to 9.9 us and made them unassertable at
//     any tolerance.)
//
//   kLinearWithIndexOverhead -- cascade delete, checked against an n log n
//     prediction rather than a linear one. Each cascaded victim performs
//     several path-copying persistent-map erases (by_type_, plus
//     drop_field_keys / drop_cached_fields / drop_cached_references --
//     Order declares several indexed fields), so the real cost is
//     superlinear: measured x5.01-x7.74 against a x4.00 pure-linear
//     prediction, i.e. an empirical exponent around n^1.2-n^1.5. Against
//     the n log n prediction (x4.65-x4.78) the measured factors sit at
//     1.10-1.57x. The 2.0x ceiling covers that; a genuine O(n^2)
//     regression would land at ~3.4x and still fail.
//
// Note the step ratios: the write-side sweeps use 4x steps, not 2x. At 2x
// steps a quadratic regression produces x4.00 while observed noise alone
// reached x3.45 -- the two overlap, so the check could not tell them
// apart. At 4x steps linear predicts x4 and quadratic x16, which noise
// cannot bridge.
struct Tolerance {
    double lo_mult;
    double hi_mult;
};

constexpr Tolerance kDefaultBand{0.8, 1.2};              // the +/-20% band
constexpr Tolerance kSmallIndexedRead{0.0, 1.3};         // sub-microsecond: ceiling-only, see above
constexpr Tolerance kLinearWithIndexOverhead{0.7, 2.0};  // n log n + allocator churn

// For a claim whose absolute timings are too noisy for a step band, but
// whose RELATIVE behaviour across the sweep is unmistakable: assert that
// the indexed-vs-unindexed gap WIDENS by at least `min_growth` from the
// smallest size to the largest. That is exactly the "one of these is O(n)
// and the other is not" claim, and it is immune to the cache and
// allocator noise that makes the scan's own per-step timings unstable --
// the measured gap runs from ~5x at the smallest size to ~1000x at the
// largest, so a 3x minimum has enormous headroom while still failing
// outright if the index ever stops being asymptotically better.
void check_gap_widens(const char* label, double small_scan, double small_indexed, double large_scan,
                      double large_indexed, double min_growth) {
    const double small_ratio = small_scan / small_indexed;
    const double large_ratio = large_scan / large_indexed;
    std::printf(
        "    %-24s gap: %.1fx at smallest size -> %.1fx at largest (need >= %.1fx growth)\n", label,
        small_ratio, large_ratio, min_growth);
    if (!kAssertTimings) return;  // see kAssertTimings
    CHECK(large_ratio >= small_ratio * min_growth);
}

// The actual pass/fail mechanism this file uses for every size/fanout
// sweep: for each step, the EXPECTED timing is the PREVIOUS step's own
// observed timing, scaled by what `order` predicts from the ratio between
// consecutive sizes -- never a fixed number of milliseconds. A fixed
// wall-clock bound is either too loose to catch a real regression (an
// O(log n) lookup silently becoming O(n) still finishes in a few ms at
// these sizes) or too tight to survive a slower machine; checking the
// SHAPE of the curve against the previous step does neither -- it is
// independent of the absolute speed of whatever hardware runs it, and it
// directly encodes the actual claim ("this grows like log n", not "this
// takes under 50 ms").
void check_scaling(const char* label, GrowthOrder order, const std::vector<int>& sizes,
                   const std::vector<double>& times, Tolerance tol = kDefaultBand) {
    for (std::size_t i = 1; i < sizes.size(); ++i) {
        const double factor = scaling_factor(order, sizes[i - 1], sizes[i]);
        const double expected = times[i - 1] * factor;
        const double lo = expected * tol.lo_mult;
        const double hi = expected * tol.hi_mult;
        const double actual_factor = times[i - 1] > 0 ? times[i] / times[i - 1] : 0.0;
        std::printf(
            "    %-24s %7d -> %7d : predicted x%.2f, measured x%.2f  (actual=%.6f, allowed "
            "[%.6f, %.6f])\n",
            label, sizes[i - 1], sizes[i], factor, actual_factor, times[i], lo, hi);
        if (!kAssertTimings) continue;  // see kAssertTimings
        CHECK(times[i] >= lo);
        CHECK(times[i] <= hi);
    }
}

// ---------------------------------------------------------------------------
// Setup helpers -- batched creates so seeding a large model is far faster
// than one commit per object, while still going through the real
// create()/try_commit() path (not a backdoor into writer-private state).
// ---------------------------------------------------------------------------

Ref<Account> make_one_account(Model& m, const std::string& name) {
    Transaction txn = m.begin();
    auto a = std::make_unique<Account>();
    a->name = name;
    const Ref<Account> local = txn.create(std::move(a));
    CommitResult res = m.try_commit(txn);
    return res.to_real(local);
}

void seed_accounts(Model& m, int n, std::vector<Ref<Account>>& out) {
    constexpr int kBatch = 2000;
    out.clear();
    out.reserve(static_cast<std::size_t>(n));
    for (int i = 0; i < n; i += kBatch) {
        Transaction txn = m.begin();
        std::vector<Ref<Account>> local;
        for (int j = i; j < std::min(n, i + kBatch); ++j) {
            auto a = std::make_unique<Account>();
            a->name = "A" + std::to_string(j);
            local.push_back(txn.create(std::move(a)));
        }
        CommitResult res = m.try_commit(txn);
        for (auto& r : local) out.push_back(res.to_real(r));
    }
}

// Round-robins `accounts` so referrers spread evenly. `qty_mod` bounds
// Order::qty's range (default 50): a probe value's match count is roughly
// n_orders / qty_mod, so callers isolating an index's O(matches) term from
// its O(log n) term (see the cached-field/cached-referrer tests below)
// scale qty_mod with n_orders to keep matches roughly constant across a
// size sweep -- the same reason examples/cached_reference_bench.cpp scales
// its bucket count with population (bucket_count_for).
void seed_orders(Model& m, const std::vector<Ref<Account>>& accounts, int n_orders,
                 std::vector<Ref<Order>>& out, int qty_mod = 50) {
    constexpr int kBatch = 2000;
    out.clear();
    out.reserve(static_cast<std::size_t>(n_orders));
    for (int i = 0; i < n_orders; i += kBatch) {
        Transaction txn = m.begin();
        std::vector<Ref<Order>> local;
        for (int j = i; j < std::min(n_orders, i + kBatch); ++j) {
            auto o = std::make_unique<Order>();
            o->code = "O" + std::to_string(j);
            o->account = accounts[static_cast<std::size_t>(j) % accounts.size()];
            o->qty = j % qty_mod;
            local.push_back(txn.create(std::move(o)));
        }
        CommitResult res = m.try_commit(txn);
        for (auto& r : local) out.push_back(res.to_real(r));
    }
}

}  // namespace

// ---------------------------------------------------------------------------
// Read-side complexity claims
// ---------------------------------------------------------------------------

// Account::name is declared in BOTH define_keys() (find_by_key, unique,
// O(1)) and define_scan_fields() (find_by_scan_field, O(#accounts) linear
// scan) -- see tests/test_types.h. Same field, same data, same query: any timing
// difference is attributable entirely to the index, not to anything else.
PERF_TEST(find_by_key_is_flat_while_find_by_scan_field_grows_with_population) {
    const std::vector<int> sizes = {scaled(25000), scaled(50000), scaled(100000)};
    std::vector<double> key_us, scan_us;
    for (int n : sizes) {
        Model m;
        std::vector<Ref<Account>> accounts;
        seed_accounts(m, n, accounts);
        const std::string probe = "A" + std::to_string(n / 2);
        Snapshot s = m.snapshot();

        constexpr int kKeyReps = 5000;
        key_us.push_back(best_of(15, [&] {
            return time_ms([&] {
                       for (int i = 0; i < kKeyReps; ++i)
                           (void)s.find_by_key<&Account::name>(probe);
                   }) *
                   1000.0 / kKeyReps;
        }));

        // Fewer repetitions as n grows -- each individual scan call is
        // O(n), so this keeps total wall time roughly bounded while still
        // reporting a stable per-call average.
        const int scan_reps = std::max(5, 200000 / n);
        scan_us.push_back(best_of(5, [&] {
            return time_ms([&] {
                       for (int i = 0; i < scan_reps; ++i)
                           (void)s.find_by_scan_field<&Account::name>(probe);
                   }) *
                   1000.0 / scan_reps;
        }));

        std::printf("  n=%7d  find_by_key=%9.4f us/call   find_by_scan_field=%9.2f us/call\n", n,
                    key_us.back(), scan_us.back());
    }
    check_scaling("find_by_key", GrowthOrder::kConstant, sizes, key_us);
    check_gap_widens("find_by_key vs scan", scan_us.front(), key_us.front(), scan_us.back(),
                     key_us.back(), 3.0);
}

// Same comparison, one level up: Order::qty is in BOTH define_cached_fields
// (indexed, O(log n + matches)) and define_scan_fields (O(#orders) scan).
PERF_TEST(find_by_cached_field_stays_near_flat_while_scan_grows_on_the_same_field) {
    const std::vector<int> sizes = {scaled(25000), scaled(50000), scaled(100000)};
    std::vector<double> cached_us, scan_us;
    for (int n : sizes) {
        Model m;
        std::vector<Ref<Account>> accounts;
        seed_accounts(m, 4, accounts);
        std::vector<Ref<Order>> orders;
        // qty_mod scales with n so match count for `probe_qty` stays ~10
        // regardless of n -- isolates the index's O(log n) term from its
        // O(matches) term (see seed_orders' own comment).
        const int qty_mod = std::max(8, n / 10);
        seed_orders(m, accounts, n, orders, qty_mod);
        const std::int64_t probe_qty = 7;  // always < qty_mod (>= 8), always ~10 matches
        Snapshot s = m.snapshot();

        constexpr int kCachedReps = 2000;
        cached_us.push_back(best_of(15, [&] {
            return time_ms([&] {
                       for (int i = 0; i < kCachedReps; ++i)
                           (void)s.find_by_cached_field<&Order::qty>(probe_qty);
                   }) *
                   1000.0 / kCachedReps;
        }));

        const int scan_reps = std::max(5, 200000 / n);
        scan_us.push_back(best_of(5, [&] {
            return time_ms([&] {
                       for (int i = 0; i < scan_reps; ++i)
                           (void)s.find_by_scan_field<&Order::qty>(probe_qty);
                   }) *
                   1000.0 / scan_reps;
        }));

        std::printf(
            "  n=%7d  find_by_cached_field=%9.4f us/call   find_by_scan_field=%9.2f us/call\n", n,
            cached_us.back(), scan_us.back());
    }
    // Matches held ~constant across the sweep (see qty_mod above), so this
    // isolates the index's O(log n) term from its O(matches) term.
    check_scaling("find_by_cached_field", GrowthOrder::kLogN, sizes, cached_us, kSmallIndexedRead);
    check_gap_widens("cached_field vs scan", scan_us.front(), cached_us.front(), scan_us.back(),
                     cached_us.back(), 3.0);
}

// The reverse-lookup counterpart: Order::account is in define_references()
// (always -- required for cascade) AND define_cached_references() (opt-in
// index). find_referrers is the always-available O(#orders) scan;
// find_cached_referrers is the O(log n + matches) indexed alternative --
// same comparison examples/cached_reference_bench.cpp benchmarks in more
// depth, here as an asserted claim at smaller, CI-friendly sizes.
PERF_TEST(find_cached_referrers_beats_the_scan_and_the_gap_widens_with_population) {
    const std::vector<int> sizes = {scaled(25000), scaled(50000), scaled(100000)};
    std::vector<double> scan_us, indexed_us;
    for (int n : sizes) {
        Model m;
        // Account count scales with n (same reason as bucket_count_for in
        // examples/cached_reference_bench.cpp): match count for `target`
        // stays ~10 regardless of n, isolating the index's O(log n) term
        // from its O(matches) term.
        const int n_accounts = std::max(4, n / 10);
        std::vector<Ref<Account>> accounts;
        seed_accounts(m, n_accounts, accounts);
        std::vector<Ref<Order>> orders;
        seed_orders(m, accounts, n, orders);
        Snapshot s = m.snapshot();
        const Ref<Account> target = accounts[0];

        const int reps = std::max(5, 200000 / n);
        scan_us.push_back(best_of(5, [&] {
            return time_ms([&] {
                       for (int i = 0; i < reps; ++i)
                           (void)s.find_referrers<&Order::account>(target);
                   }) *
                   1000.0 / reps;
        }));

        constexpr int kIndexedReps = 2000;
        indexed_us.push_back(best_of(15, [&] {
            return time_ms([&] {
                       for (int i = 0; i < kIndexedReps; ++i)
                           (void)s.find_cached_referrers<&Order::account>(target);
                   }) *
                   1000.0 / kIndexedReps;
        }));

        std::printf("  n=%7d  scan=%10.2f us/call   indexed=%10.4f us/call   speedup=%8.0fx\n", n,
                    scan_us.back(), indexed_us.back(), scan_us.back() / indexed_us.back());
    }
    // Matches held ~constant across the sweep (account count scales with
    // n above), so this isolates the index's O(log n) term.
    check_scaling("find_cached_referrers", GrowthOrder::kLogN, sizes, indexed_us,
                  kSmallIndexedRead);
    check_gap_widens("cached_referrers vs scan", scan_us.front(), indexed_us.front(),
                     scan_us.back(), indexed_us.back(), 3.0);
}

// ---------------------------------------------------------------------------
// Write-side complexity claims
// ---------------------------------------------------------------------------

// try_commit()'s publish step copies spine_ (O(#chunks) shared_ptr copies,
// not a deep copy -- see DESIGN.md's Structure diagram) and every
// per-type/per-field index (O(#types)/O(#indexed fields), independent of
// model size). #chunks is proportional to model size, so the claim is that
// a single-field update's commit latency grows LINEARLY with model size --
// not flat, but also not proportional to a deep copy of the whole model
// (which would grow far faster than linear).
PERF_TEST(single_field_commit_latency_stays_bounded_as_total_model_size_grows) {
    const std::vector<int> sizes = {scaled(6250), scaled(25000), scaled(100000)};
    std::vector<double> ms_per_commit;
    for (int n : sizes) {
        Model m;
        std::vector<Ref<Account>> accounts;
        seed_accounts(m, n, accounts);

        // Many commits, min-of-several-trials aggregation: the chunk-copy
        // term this test is isolating is small relative to fixed
        // per-commit overhead (lock acquisition, index bookkeeping) at
        // these sizes, so a single (or lightly-averaged) sample is mostly
        // noise -- see best_of()'s own comment.
        constexpr int kReps = 50;
        ms_per_commit.push_back(best_of(5, [&] {
            double total = 0;
            for (int i = 0; i < kReps; ++i) {
                Transaction txn = m.begin();
                txn.update(accounts[static_cast<std::size_t>(i) % accounts.size()])->balance = i;
                total += time_ms([&] { (void)m.try_commit(txn); });
            }
            return total / kReps;
        }));
        std::printf("  n=%7d  single-field commit = %8.5f ms/commit (best-of-5 avg of %d)\n", n,
                    ms_per_commit.back(), kReps);
    }
    // No step band, for the same reason as the hub-referrer test below: the
    // size-dependent term is a minority of what is being measured. A
    // single-field commit at n=100,000 costs 6-15 us in total, of which the
    // O(#chunks) spine copy -- the only part that grows with model size --
    // is roughly 2 us; the rest is fixed (lock, Root allocation, index map
    // copies, changelog append) and overlaps asynchronous reaper work.
    // Measured step factors ranged x1.51-x7.60 against a x4.00 linear
    // prediction across repeat runs, i.e. 0.38x-1.90x of it: no band over
    // that spread would mean anything.
    //
    // The ceiling is what carries the value, and it is aimed at a specific
    // regression: if try_commit ever deep-copied objects instead of copying
    // shared_ptrs, per-commit cost would become proportional to model size
    // with a huge constant -- at n=100,000 that is milliseconds, ~1000x the
    // smallest-size reading. A 20x ceiling over a 16x size increase clears
    // the observed noise and still fails that outright.
    std::printf("    %-24s %7d -> %7d objects (16x): %.5f ms -> %.5f ms (ceiling %.5f)\n",
                "single-field commit", sizes.front(), sizes.back(), ms_per_commit.front(),
                ms_per_commit.back(), ms_per_commit.front() * 20.0);
    if (kAssertTimings) CHECK(ms_per_commit.back() <= ms_per_commit.front() * 20.0);
}

// ---------------------------------------------------------------------------
// Concurrent throughput: does parallel Transaction *building* actually buy
// anything, given commit_mu_ fully serializes every try_commit()'s apply
// step (CLAUDE.md invariant 7)?
// ---------------------------------------------------------------------------

namespace {

// kThreads disjoint groups of kObjectsPerThread Accounts each, seeded in one
// batched commit. Each thread only ever touches its own group, so this
// isolates parallel building/serialized-apply overlap from conflict-retry
// overhead -- the same reason commit_bench.cpp's throughput sweep gives
// each thread its own Account.
constexpr int kThreads = 4;
constexpr int kObjectsPerThread = 100;

void seed_private_groups(Model& m, std::vector<std::vector<Ref<Account>>>& groups) {
    groups.assign(kThreads, {});
    Transaction txn = m.begin();
    std::vector<std::vector<Ref<Account>>> local(kThreads);
    for (int t = 0; t < kThreads; ++t) {
        for (int i = 0; i < kObjectsPerThread; ++i) {
            auto a = std::make_unique<Account>();
            a->name = "T" + std::to_string(t) + "_" + std::to_string(i);
            local[static_cast<std::size_t>(t)].push_back(txn.create(std::move(a)));
        }
    }
    const CommitResult res = m.try_commit(txn);
    for (int t = 0; t < kThreads; ++t)
        for (auto& r : local[static_cast<std::size_t>(t)])
            groups[static_cast<std::size_t>(t)].push_back(res.to_real(r));
}

// Unlike seed_private_groups/four_threads_updating_private_objects_..._
// sequentially above, EVERY iteration creates a brand-new batch of
// `batch_size` objects (globally-unique names, never reused) and then
// updates that same fresh batch -- two commits, not one. The point isn't
// the create/update mechanics; it's the ids: because every iteration's
// batch is disjoint from every other iteration's (this thread's own past
// ones AND every other thread's), publish_now()'s conflict-prune scan
// (which drops an existing undo_list_ entry only when a LATER commit
// touches an overlapping id) never fires ACROSS iterations. Within one
// iteration it fires exactly once -- the update's touched set is the same
// as its own create's, so the create's own entry is invalidated the moment
// the update lands, leaving one surviving UndoEntry per iteration. So an
// UNCAPPED run's undo_list_ grows by one entry every iteration, with
// nothing ever pruning it back down -- unlike the reused-object workload
// above, where the list stays near kThreads entries regardless of any cap.
// That's what makes a cap actually bind here: by the last iteration,
// kThreads * kIters surviving entries is comfortably past both the 1000
// and 100 caps tested below, so the conflict-prune scan's O(list size)
// cost is genuinely smaller under a cap than without one. `batch_size` is
// deliberately small (not kObjectsPerThread=100): the effect under test is
// driven by the NUMBER of undo_list_ entries, not their size, so a small
// batch reaches the same entry counts for a fraction of the clone/commit
// cost -- keeping kIters large enough to clear the 1000 cap affordable.
double bench_concurrent_create_and_update_with_undo_cap(std::size_t max_undo_list_size, int kIters,
                                                        int batch_size, int trials) {
    return best_of(trials, [&] {
        Model m;
        m.set_max_undo_list_size(max_undo_list_size);
        return time_ms([&] {
            std::vector<std::thread> pool;
            for (int t = 0; t < kThreads; ++t) {
                pool.emplace_back([&, t] {
                    for (int it = 0; it < kIters; ++it) {
                        std::vector<Ref<Account>> batch;
                        {
                            Transaction txn = m.begin();
                            std::vector<Ref<Account>> local;
                            for (int i = 0; i < batch_size; ++i) {
                                auto a = std::make_unique<Account>();
                                a->name = "T" + std::to_string(t) + "_" + std::to_string(it) + "_" +
                                          std::to_string(i);
                                local.push_back(txn.create(std::move(a)));
                            }
                            const CommitResult res = m.try_commit(txn);
                            for (auto& r : local) batch.push_back(res.to_real(r));
                        }
                        {
                            Transaction txn = m.begin();
                            for (const auto& r : batch)
                                if (Account* a = txn.update(r)) a->balance = it;
                            m.try_commit(txn);
                        }
                    }
                });
            }
            for (auto& th : pool) th.join();
        });
    });
}

}  // namespace

// 4 threads, each repeatedly building-and-committing a transaction that
// updates all 100 of ITS OWN objects (never another thread's), must finish
// the same total work -- kThreads * kIters transactions, each touching
// kObjectsPerThread objects -- faster than running the identical sequence
// of transactions one after another on a single thread. This isn't a
// claim that try_commit() itself parallelizes: commit_mu_ still serializes
// every one of the 4 applies (see commit_bench.cpp's sub-linear
// throughput-vs-threads sweep, and CLAUDE.md's "Known scope boundaries").
// What DOES parallelize is Transaction building -- txn.update() clones an
// object into the transaction's local overlay without touching commit_mu_
// at all -- so while one thread holds commit_mu_ applying, the other three
// can be off cloning and mutating their own next transaction instead of
// waiting idle, which the fully-sequential run can never do.
PERF_TEST(
    four_threads_updating_private_objects_concurrently_beats_running_the_same_transactions_sequentially) {
    const int kIters = scaled(300);

    Model concurrent_model;
    std::vector<std::vector<Ref<Account>>> concurrent_groups;
    seed_private_groups(concurrent_model, concurrent_groups);

    const double concurrent_ms = best_of(3, [&] {
        return time_ms([&] {
            std::vector<std::thread> pool;
            for (int t = 0; t < kThreads; ++t) {
                pool.emplace_back([&, t] {
                    const auto& objs = concurrent_groups[static_cast<std::size_t>(t)];
                    for (int it = 0; it < kIters; ++it) {
                        Transaction txn = concurrent_model.begin();
                        for (const auto& r : objs)
                            if (Account* a = txn.update(r)) a->balance = it;
                        concurrent_model.try_commit(txn);
                    }
                });
            }
            for (auto& th : pool) th.join();
        });
    });

    Model sequential_model;
    std::vector<std::vector<Ref<Account>>> sequential_groups;
    seed_private_groups(sequential_model, sequential_groups);

    const double sequential_ms = best_of(3, [&] {
        return time_ms([&] {
            for (int t = 0; t < kThreads; ++t) {
                const auto& objs = sequential_groups[static_cast<std::size_t>(t)];
                for (int it = 0; it < kIters; ++it) {
                    Transaction txn = sequential_model.begin();
                    for (const auto& r : objs)
                        if (Account* a = txn.update(r)) a->balance = it;
                    sequential_model.try_commit(txn);
                }
            }
        });
    });

    const double speedup = sequential_ms / concurrent_ms;
    std::printf(
        "  %d threads x %d iters x %d objs/txn:  concurrent=%.2f ms   sequential=%.2f ms   "
        "speedup=%.2fx\n",
        kThreads, kIters, kObjectsPerThread, concurrent_ms, sequential_ms, speedup);

    if (!kAssertTimings) return;  // see kAssertTimings
    // Not asserting anywhere near kThreads-fold: apply is fully serialized,
    // so the ceiling on speedup is nowhere close to 4x. The floor asserted
    // here is deliberately modest -- it only needs to catch a regression
    // that makes concurrent building stop overlapping with serialized
    // apply at all (e.g. a lock accidentally widened to cover building
    // too), not chase a specific ratio.
    CHECK(speedup > 1.15);
}

// Three variants of the SAME kThreads x kIters concurrent shape as the test
// just above (see bench_concurrent_create_and_update_with_undo_cap's own
// comment for why this workload creates-then-updates a fresh batch every
// iteration, rather than reusing kObjectsPerThread objects the way the
// test above does), with the ONLY difference being Model::set_max_undo_
// list_size(), swept at maximum (the historical, uncapped default), 1000,
// and 100 -- each compares its capped run against a fresh uncapped run and
// reports the speedup. kIters * kThreads (1200) comfortably clears both
// caps, so undo_list_ genuinely exceeds them in the uncapped run and both
// caps do real pruning work, not a no-op.
//
// publish_now()'s undo-list maintenance can only do LESS work under a cap
// than without one, never more, so a capped run is never EXPECTED to be
// slower than uncapped: the floor below asserts exactly that (speedup >=
// 1.0, no ceiling -- this isn't a claim about how MUCH faster, just that
// bounding retention never costs anything).
PERF_TEST(
    four_threads_updating_private_objects_with_undo_list_capped_at_1000_is_never_slower_than_uncapped) {
    // kThreads * kIters needs to clear kCap by a wide margin, not just cross
    // it -- at 1200 (kIters=300, the other two tests' iteration count) the
    // uncapped list is only 20% over the cap, so the pruning work saved is a
    // small fraction of total commit cost and gets lost in machine noise
    // (measured flipping below 1.0x across repeat runs at that size). 900
    // gives kThreads * kIters = 3600, 3.6x the cap, a comfortably larger
    // fraction of steady-state list scans to save.
    const int kIters = scaled(900);
    constexpr int kBatchSize = 10;
    constexpr int kTrials = 5;
    constexpr std::size_t kCap = 1000;
    const double uncapped_ms = bench_concurrent_create_and_update_with_undo_cap(
        std::numeric_limits<std::size_t>::max(), kIters, kBatchSize, kTrials);
    const double capped_ms =
        bench_concurrent_create_and_update_with_undo_cap(kCap, kIters, kBatchSize, kTrials);
    const double speedup = uncapped_ms / capped_ms;
    std::printf(
        "  undo list cap=%4zu vs maximum:  uncapped=%7.2f ms   capped=%7.2f ms   speedup=%.2fx\n",
        kCap, uncapped_ms, capped_ms, speedup);
    if (!kAssertTimings) return;  // see kAssertTimings
    CHECK(speedup >= 1.0);
}

PERF_TEST(
    four_threads_updating_private_objects_with_undo_list_capped_at_100_is_never_slower_than_uncapped) {
    const int kIters = scaled(300);
    constexpr int kBatchSize = 10;
    constexpr int kTrials = 3;
    constexpr std::size_t kCap = 100;
    const double uncapped_ms = bench_concurrent_create_and_update_with_undo_cap(
        std::numeric_limits<std::size_t>::max(), kIters, kBatchSize, kTrials);
    const double capped_ms =
        bench_concurrent_create_and_update_with_undo_cap(kCap, kIters, kBatchSize, kTrials);
    const double speedup = uncapped_ms / capped_ms;
    std::printf(
        "  undo list cap=%4zu vs maximum:  uncapped=%7.2f ms   capped=%7.2f ms   speedup=%.2fx\n",
        kCap, uncapped_ms, capped_ms, speedup);
    if (!kAssertTimings) return;  // see kAssertTimings
    CHECK(speedup >= 1.0);
}

PERF_TEST(
    four_threads_updating_private_objects_with_undo_list_capped_at_0_is_never_slower_than_uncapped) {
    const int kIters = scaled(300);
    constexpr int kBatchSize = 10;
    constexpr int kTrials = 3;
    constexpr std::size_t kCap = 0;
    const double uncapped_ms = bench_concurrent_create_and_update_with_undo_cap(
        std::numeric_limits<std::size_t>::max(), kIters, kBatchSize, kTrials);
    const double capped_ms =
        bench_concurrent_create_and_update_with_undo_cap(kCap, kIters, kBatchSize, kTrials);
    const double speedup = uncapped_ms / capped_ms;
    std::printf(
        "  undo list cap=%4zu vs maximum:  uncapped=%7.2f ms   capped=%7.2f ms   speedup=%.2fx\n",
        kCap, uncapped_ms, capped_ms, speedup);
    if (!kAssertTimings) return;  // see kAssertTimings
    CHECK(speedup >= 1.0);
}

// ---------------------------------------------------------------------------
// Worst-case / degenerate data
// ---------------------------------------------------------------------------
//
// CLAUDE.md's "Known scope boundaries" already documents two specific,
// unbounded costs -- not bugs, deliberate tradeoffs -- that this section
// exists to demonstrate and bound, not silently accept:
//   - referrers_ is a vector<RefEdge> scanned linearly (find_if) per
//     target, so mutating/removing ONE referrer of a hub object costs
//     O(#referrers of that hub), regardless of total model size.
//   - Cascade fan-out from a single remove() intent is unbounded: removing
//     one end of a long non-nullable reference CHAIN walks the whole
//     chain, one hop at a time.
// These are two DIFFERENT cost drivers -- fan-OUT (breadth) vs cascade
// DEPTH -- so they get two separate tests below.

namespace {

// A hub Account, non-nullably referenced by exactly `fanout` Orders, plus
// `background` unrelated accounts+orders so the model's TOTAL size doesn't
// collapse to just the hub's own neighborhood -- fanout is the one
// deliberately varying axis.
void seed_hub_with_fanout(Model& m, int fanout, int background, Ref<Account>& hub,
                          std::vector<Ref<Order>>& hub_orders) {
    std::vector<Ref<Account>> bg_accounts;
    seed_accounts(m, background, bg_accounts);
    std::vector<Ref<Order>> bg_orders;
    seed_orders(m, bg_accounts, background, bg_orders);

    hub = make_one_account(m, "HUB");
    const std::vector<Ref<Account>> hub_only{hub};
    seed_orders(m, hub_only, fanout, hub_orders);
}

}  // namespace

// Removing ONE of a hub's many referrers forces drop_out_refs to locate
// that one edge in the hub's referrer vector (a linear find_if) and erase
// it (a vector erase) -- the O(#referrers) cost CLAUDE.md's
// scope-boundaries section documents. Held here as an UPPER bound only,
// which is a deliberate, measured decision rather than a loose default:
//
//   - The cost depends strongly on WHERE the edge sits. Removing the
//     first-created referrer (edge at the FRONT: find_if returns
//     immediately, but erase must memmove the whole tail) and the
//     last-created one (edge at the BACK: full find_if scan, O(1) erase)
//     differ by ~10x on a single cold measurement -- 16 ms vs 1.2 ms at
//     fanout=50,000.
//   - Almost all of that gap is COLD-CACHE first touch, not asymptotics.
//     Taking the best of five successive removals collapses it: the same
//     front-edge case then measures 0.23 ms at fanout=25,000, 0.17 ms at
//     50,000 and 0.33 ms at 100,000 -- no clean growth at all, because in
//     steady state fixed per-commit overhead dominates both the scan and
//     the memmove.
//
// So a two-sided band here would be asserting noise. The upper bound is
// still worth having: it rejects the regression that actually matters
// (this operation going quadratic), while not pretending the linear term
// is separable from commit overhead at any size this suite can afford.
PERF_TEST(worst_case_removing_one_referrer_of_a_hub_scales_with_hub_fanout) {
    const std::vector<int> fanouts = {scaled(6250), scaled(25000), scaled(100000)};
    constexpr int kTrials = 5;
    std::vector<double> times_ms;
    for (int fanout : fanouts) {
        Model m;
        Ref<Account> hub;
        std::vector<Ref<Order>> hub_orders;
        // Seed `fanout + kTrials` referrers, then remove kTrials of them
        // ONE AT A TIME below: kTrials << fanout, so the hub's fanout is
        // still ~`fanout` for every one of those removals, letting us take
        // several trials without rebuilding the whole hub each time.
        seed_hub_with_fanout(m, fanout + kTrials, /*background=*/1000, hub, hub_orders);

        std::vector<double> trial_ms;
        for (int t = 0; t < kTrials; ++t) {
            const Ref<Order> victim = hub_orders.back();
            hub_orders.pop_back();
            trial_ms.push_back(time_ms([&] {
                Transaction txn = m.begin();
                txn.remove(victim);
                (void)m.try_commit(txn);
            }));
        }
        times_ms.push_back(*std::min_element(trial_ms.begin(), trial_ms.end()));
        std::printf("  hub fanout=%6d   remove one referrer = %8.4f ms (best of %d)\n", fanout,
                    times_ms.back(), kTrials);
    }
    // No step band: the measured values here are ~0.1 ms, small enough that
    // run-to-run jitter exceeds the size-dependent term entirely (see the
    // comment above). What IS robust is that a 16x increase in fanout must
    // not produce a blowup: quadratic would be ~256x. A 10x ceiling clears
    // the observed noise by a wide margin and still fails loudly on a real
    // complexity regression.
    std::printf("    %-24s %7d -> %7d fanout (16x): %.4f ms -> %.4f ms (ceiling %.4f)\n",
                "remove one hub referrer", fanouts.front(), fanouts.back(), times_ms.front(),
                times_ms.back(), times_ms.front() * 10.0);
    if (kAssertTimings) CHECK(times_ms.back() <= times_ms.front() * 10.0);
}

// Removing the hub ITSELF (rather than one of its referrers, as the test
// above does) cascades to every Order pointing at it -- so the work is
// proportional to the SIZE OF THE CASCADE, which is the cost CLAUDE.md's
// scope-boundaries section means by "cascade fan-out from a single
// remove() intent is still unbounded."
//
// A note on shape: this measures a HUB (breadth), not a deep chain
// (a -> b -> c -> ..., removing `a` cascading hop by hop). Deep
// non-nullable chains -- and even non-nullable cycles -- ARE expressible
// (the remap table is minted in full before any create applies, in both
// try_commit()'s pre-mint pass and commit_bulk_without_undo(), so same-transaction
// forward references and self-loops resolve; see
// apply_transaction_contents). The hub stays the representative worst
// case regardless: the cascade BFS does the same per-victim work whether
// the victims arrive broad or deep, so cost scales with the cascade's
// SIZE either way, which is what this test measures.
PERF_TEST(worst_case_removing_a_hub_scales_with_the_size_of_its_cascade) {
    const std::vector<int> fanouts = {scaled(1250), scaled(5000), scaled(20000)};
    constexpr int kTrials = 5;
    std::vector<double> times_ms;
    for (int fanout : fanouts) {
        Model m;
        // Removing a hub destroys that whole sub-graph, so the same hub
        // can't be measured twice -- build kTrials INDEPENDENT hubs of
        // equal fanout in one Model, then remove each exactly once.
        std::vector<Ref<Account>> hubs;
        for (int t = 0; t < kTrials; ++t) {
            const Ref<Account> hub = make_one_account(m, "HUB" + std::to_string(t));
            std::vector<Ref<Order>> orders;
            seed_orders(m, std::vector<Ref<Account>>{hub}, fanout, orders);
            hubs.push_back(hub);
        }
        const std::size_t before = m.snapshot().size();

        std::vector<double> trial_ms;
        for (const Ref<Account>& hub : hubs) {
            trial_ms.push_back(time_ms([&] {
                Transaction txn = m.begin();
                txn.remove(hub);
                (void)m.try_commit(txn);
            }));
        }
        times_ms.push_back(*std::min_element(trial_ms.begin(), trial_ms.end()));
        std::printf("  cascade size=%6d   remove hub (full cascade) = %8.4f ms (best of %d)\n",
                    fanout, times_ms.back(), kTrials);

        // Each removal killed its hub AND all `fanout` Orders under it.
        CHECK(m.snapshot().size() == before - static_cast<std::size_t>(fanout + 1) * kTrials);
    }
    check_scaling("remove hub (cascade)", GrowthOrder::kNLogN, fanouts, times_ms,
                  kLinearWithIndexOverhead);
}

// ---------------------------------------------------------------------------
// Object payload size -- a THIRD axis, independent of object COUNT
// ---------------------------------------------------------------------------

namespace {

// No reference fields on purpose: payload-size sensitivity is about
// clone()/copy cost (the COW per-touch cost every create/update pays), not
// the ref-graph machinery, which the tests above already cover.
class Blob final : public model::Object<Blob> {
public:
    std::string key;
    std::vector<std::string> payload;  // the "object size" knob

    template <class Self>
    static void define_keys(Self& s, const model::FieldKeyReader& v) {
        v.key<&Blob::key>(s.key);
    }
};

struct PayloadCase {
    const char* label;
    int strings;
    std::size_t str_len;
};

const std::vector<PayloadCase>& payload_cases() {
    static const std::vector<PayloadCase> cases = {
        {"small (1x16B)", 1, 16},
        {"large (50x256B)", 50, 256},
    };
    return cases;
}

void seed_blobs(Model& m, const PayloadCase& c, int count, std::vector<Ref<Blob>>& out) {
    const std::string filler(c.str_len, 'x');
    constexpr int kBatch = 1000;
    out.clear();
    out.reserve(static_cast<std::size_t>(count));
    for (int i = 0; i < count; i += kBatch) {
        Transaction txn = m.begin();
        std::vector<Ref<Blob>> local;
        for (int j = i; j < std::min(count, i + kBatch); ++j) {
            auto b = std::make_unique<Blob>();
            b->key = "B" + std::to_string(j);
            b->payload.assign(static_cast<std::size_t>(c.strings), filler);
            local.push_back(txn.create(std::move(b)));
        }
        CommitResult res = m.try_commit(txn);
        for (auto& r : local) out.push_back(res.to_real(r));
    }
}

}  // namespace

// WRITE cost (create/update, both of which clone the whole object) scales
// with the object's OWN byte size, holding object COUNT fixed -- a
// different axis than every size-sweep test above, which holds payload
// fixed and varies count.
PERF_TEST(write_cost_scales_with_object_payload_size_at_a_fixed_object_count) {
    const int kCount = scaled(5000);
    std::vector<double> create_ms, update_ms;
    for (const PayloadCase& c : payload_cases()) {
        Model m;
        std::vector<Ref<Blob>> blobs;
        const double t_create = time_ms([&] { seed_blobs(m, c, kCount, blobs); });
        create_ms.push_back(t_create);

        constexpr int kReps = 20;
        double total = 0;
        for (int i = 0; i < kReps; ++i) {
            Transaction txn = m.begin();
            txn.update(blobs[static_cast<std::size_t>(i) % blobs.size()])->key += "!";
            total += time_ms([&] { (void)m.try_commit(txn); });
        }
        update_ms.push_back(total / kReps);

        std::printf("  payload=%-16s  create(%d)=%8.3f ms   single update=%8.5f ms/commit\n",
                    c.label, kCount, t_create, update_ms.back());
    }
    // The large payload (50x the string count, 16x the string length --
    // ~800x the raw byte count) must not be catastrophically (>80x) more
    // expensive to create or clone than the small one at the SAME count.
    if (kAssertTimings) {
        CHECK(create_ms[1] < create_ms[0] * 80.0 + 200.0);
        CHECK(update_ms[1] < update_ms[0] * 80.0 + 5.0);
    }
}

// Contrast with the write-side test above: find_by_key never copies the
// object, so lookup time should stay flat regardless of payload size, at
// the SAME fixed object count.
PERF_TEST(read_lookup_time_is_insensitive_to_object_payload_size) {
    const int kCount = scaled(5000);
    std::vector<double> lookup_us;
    for (const PayloadCase& c : payload_cases()) {
        Model m;
        std::vector<Ref<Blob>> blobs;
        seed_blobs(m, c, kCount, blobs);

        Snapshot s = m.snapshot();
        const std::string probe = "B" + std::to_string(kCount / 2);
        constexpr int kReps = 20000;
        const double t = time_ms([&] {
            for (int i = 0; i < kReps; ++i) (void)s.find_by_key<&Blob::key>(probe);
        });
        lookup_us.push_back(t * 1000.0 / kReps);
        std::printf("  payload=%-16s  find_by_key = %9.4f us/call\n", c.label, lookup_us.back());
    }
    if (kAssertTimings) CHECK(lookup_us[1] < lookup_us[0] * 5.0 + 5.0);
}

// ---------------------------------------------------------------------------
// Memory claim: the cached-reference index costs SOME extra memory, but
// not a runaway amount -- same fork()+VmHWM technique as
// examples/cached_reference_bench.cpp (see that file for the full,
// multi-size sweep; this is the asserted, single-point version of the same
// claim, kept to one size to bound this suite's own runtime).
// ---------------------------------------------------------------------------

#if PERF_HAS_MEMORY_SECTION

namespace {

// Peak resident set size (VmHWM) of THIS process, in KB -- see
// examples/cached_reference_bench.cpp for why VmHWM (the high-water mark)
// rather than current RSS, and why fork() rather than getrusage(RUSAGE_
// CHILDREN) (which reports the max across ALL terminated children, not
// the most recent one).
long peak_rss_kb() {
    FILE* f = std::fopen("/proc/self/status", "r");
    if (!f) return -1;
    long kb = -1;
    char line[256];
    while (std::fgets(line, sizeof line, f)) {
        if (std::sscanf(line, "VmHWM: %ld kB", &kb) == 1) break;
    }
    std::fclose(f);
    return kb;
}

template <class F>
long measure_child_peak_kb(F&& work) {
    int fds[2];
    if (pipe(fds) != 0) return -1;
    // Return whatever this PARENT process's own earlier (in-process, not
    // forked) tests left resident-but-freed in glibc's heap arenas back to
    // the OS before forking. Without this, a child measuring a SMALL true
    // demand (tens of MB) can inherit a much larger already-resident free
    // pool from prior tests via COW and satisfy its entire allocation from
    // it without RSS growing at all -- measured directly: with pollution
    // and no trim, a genuine ~71 MB demand showed up as a ~0.6 MB delta;
    // with this trim, the same demand measures correctly. A demand large
    // enough to exceed any plausible leftover pool (hundreds of MB, as the
    // other test in this section is) isn't affected either way, but nothing
    // here should have to know in advance which regime it's in.
    malloc_trim(0);
    // Flush every open C stream BEFORE forking, not just skip re-flushing
    // in the child via _exit() below: when stdout isn't a terminal (piped,
    // redirected -- exactly how `ctest` and this very command run it), the
    // parent's test output sits fully buffered, unflushed, for a while.
    // Observed empirically under TSan: its runtime wraps _exit() and
    // flushes whatever stdio buffer it inherited anyway, so each of this
    // function's (multiple, per test) forked children re-dumped the
    // parent's ENTIRE accumulated output so far. Flushing here leaves
    // nothing buffered for a child to inherit and re-flush, regardless of
    // which runtime's _exit() wrapper does what.
    std::fflush(nullptr);
    const pid_t pid = fork();
    if (pid == 0) {
        close(fds[0]);
        work();
        const long kb = peak_rss_kb();
        char buf[32];
        const int n = std::snprintf(buf, sizeof buf, "%ld\n", kb);
        if (write(fds[1], buf, static_cast<std::size_t>(n)) < 0) { /* nothing to do */
        }
        close(fds[1]);
        _exit(0);  // not exit()/return: skip static destructors racing the parent's stdio
    }
    close(fds[1]);
    char buf[32] = {};
    const ssize_t r = read(fds[0], buf, sizeof(buf) - 1);
    close(fds[0]);
    int status = 0;
    waitpid(pid, &status, 0);
    return r > 0 ? std::atol(buf) : -1;
}

// Structurally identical types, differing in exactly one declaration --
// the same isolation cached_reference_bench.cpp's UncachedItem/CachedItem
// pair uses, so any memory difference is attributable to that one line.
class MemUncached final : public model::Object<MemUncached> {
public:
    model::Ref<Account> bucket;
    template <class Self, class V>
    static void define_references(Self& s, V&& v) {
        v(model::field_tag<&MemUncached::bucket>(), "bucket", s.bucket);
    }
};
class MemCached final : public model::Object<MemCached> {
public:
    model::Ref<Account> bucket;
    template <class Self, class V>
    static void define_references(Self& s, V&& v) {
        v(model::field_tag<&MemCached::bucket>(), "bucket", s.bucket);
    }
    template <class Self>
    static void define_cached_references(Self& s, const model::RefIndexReader& v) {
        (void)s;
        v.index<&MemCached::bucket>();
    }
};

// Spreads across n/kItemsPerBucket buckets, same as bucket_count_for() in
// examples/cached_reference_bench.cpp: a persistent map's memory cost is
// sensitive to how deep any ONE inner map gets, so dumping all n objects
// onto a single bucket (a degenerate, hub-like shape) would measure that
// worst case, not the typical per-object index-entry cost this test is
// actually after.
constexpr int kItemsPerBucket = 250;

template <class T>
void seed_one_type_spread_across_buckets(Model& m, int n) {
    const int n_buckets = std::max(50, n / kItemsPerBucket);
    std::vector<Ref<Account>> buckets;
    seed_accounts(m, n_buckets, buckets);
    Transaction txn = m.begin();
    for (int i = 0; i < n; ++i) {
        auto o = std::make_unique<T>();
        o->bucket = buckets[static_cast<std::size_t>(i) % buckets.size()];
        txn.create(std::move(o));
    }
    (void)m.try_commit(txn);
}

}  // namespace

PERF_TEST(cached_reference_index_memory_overhead_is_present_but_bounded) {
    const int kN = scaled(200000);
    const long baseline_kb = measure_child_peak_kb([] { /* just process startup cost */ });

    const long uncached_kb = measure_child_peak_kb([kN] {
                                 Model m;
                                 seed_one_type_spread_across_buckets<MemUncached>(m, kN);
                             }) - baseline_kb;

    const long cached_kb = measure_child_peak_kb([kN] {
                               Model m;
                               seed_one_type_spread_across_buckets<MemCached>(m, kN);
                           }) - baseline_kb;

    const double overhead = uncached_kb > 0 ? static_cast<double>(cached_kb) / uncached_kb : 0.0;
    std::printf("  n=%d   uncached=%ld KB   cached=%ld KB   overhead=%.2fx\n", kN, uncached_kb,
                cached_kb, overhead);

    CHECK(uncached_kb > 0);
    CHECK(cached_kb > uncached_kb);  // the claim: caching costs SOME extra memory
    // A persistent map's FIXED per-tree overhead dominates at small n and
    // amortizes away as n grows (confirmed empirically: ~7x at n=50,000,
    // ~2x at n=200,000, matching examples/cached_reference_bench.cpp's own
    // 1.44x-at-50k/1.16x-at-500k sweep) -- kN shrinks under a sanitizer
    // (see scaled()), landing back in the higher-ratio small-n regime. This
    // bound is deliberately loose enough to hold at EITHER end of that
    // curve; it exists to catch a runaway/leak-like blowup, not to pin an
    // exact ratio.
    CHECK(cached_kb < uncached_kb * 15 + 10000);
}

namespace {

// Same shape as seed_one_type_spread_across_buckets, but built on ONE
// BulkTransaction instead of one huge Transaction: the buckets themselves
// are part of the same batch (cross-object local refs, exactly like any
// other BulkTransaction), and commit_bulk_without_undo() installs everything with no
// per-object undo logging at all. This is the direct memory-cost comparison
// that motivated Model::begin_bulk()/commit_bulk_without_undo() in the first place (see
// their doc comments): a single Transaction with n creates retains n
// undo-log closures, each capturing a whole PersistentMap root, simultaneously,
// until the WHOLE transaction resolves -- measured at ~13-17x the
// steady-state per-object cost. A bulk load has nothing to roll back to
// (see commit_bulk_without_undo()'s own doc comment), so it never pays that cost.
template <class T>
void seed_one_type_via_bulk_load(Model& m, int n) {
    const int n_buckets = std::max(50, n / kItemsPerBucket);
    BulkTransaction t = m.begin_bulk();
    std::vector<Ref<Account>> buckets;
    buckets.reserve(static_cast<std::size_t>(n_buckets));
    for (int i = 0; i < n_buckets; ++i) {
        auto a = std::make_unique<Account>();
        a->name = "B" + std::to_string(i);
        buckets.push_back(t.create(std::move(a)));
    }
    for (int i = 0; i < n; ++i) {
        auto o = std::make_unique<T>();
        o->bucket = buckets[static_cast<std::size_t>(i) % buckets.size()];
        t.create(std::move(o));
    }
    (void)m.commit_bulk_without_undo(t);
}

}  // namespace

PERF_TEST(bulk_load_avoids_the_single_transaction_undo_log_memory_blowup) {
    const int kN = scaled(200000);
    const long baseline_kb = measure_child_peak_kb([] { /* just process startup cost */ });

    const long single_txn_kb = measure_child_peak_kb([kN] {
                                   Model m;
                                   seed_one_type_spread_across_buckets<MemUncached>(m, kN);
                               }) - baseline_kb;

    const long bulk_kb = measure_child_peak_kb([kN] {
                             Model m;
                             seed_one_type_via_bulk_load<MemUncached>(m, kN);
                         }) - baseline_kb;

    const double single_bpi = kN > 0 ? single_txn_kb * 1024.0 / kN : 0.0;
    const double bulk_bpi = kN > 0 ? bulk_kb * 1024.0 / kN : 0.0;
    const double ratio = bulk_kb > 0 ? static_cast<double>(single_txn_kb) / bulk_kb : 0.0;
    std::printf(
        "  n=%d   single-transaction=%ld KB (%.0f B/item)   bulk-load=%ld KB (%.0f B/item)   "
        "ratio=%.2fx\n",
        kN, single_txn_kb, single_bpi, bulk_kb, bulk_bpi, ratio);

    CHECK(single_txn_kb > 0);
    CHECK(bulk_kb > 0);
    // The whole point of commit_bulk_without_undo(): no per-object undo-log retention, so
    // its per-item cost lands near the steady-state batched-commit figure,
    // not the single-huge-transaction figure. Measured ratio, repeatedly:
    // ~11-12x at n=200,000 in the default build, ~12-15x under TSan.
    //
    // The ratio is asserted ONLY where kAssertTimings is (the default
    // build), for the same reason the timing bounds are: under ASan,
    // redzone overhead is charged per ALLOCATION, not per requested byte,
    // and every allocation-count optimization in the persistent indexes
    // (PersistentSet, the inline-entry Leaf, its unique_ptr chain link --
    // see persistent_map.h) shrank ASan's measured floor for BOTH paths
    // together, compressing the measured ratio at ASan's reduced n=8,000
    // from ~2.2x through ~1.8x down to ~1.26x across this file's history
    // while the default build's ratio stayed ~11x throughout. A floor
    // asserted on that number would be asserting ASan's allocator profile,
    // not the model's memory behavior. 5x leaves ~2x margin under the
    // weakest default-build measurement.
    if (kAssertTimings) CHECK(bulk_kb * 5 < single_txn_kb);
}

#endif  // PERF_HAS_MEMORY_SECTION

// ---------------------------------------------------------------------------

int main() {
    int total = 0;
    for (auto& t : registry()) {
        std::printf("[ RUN  ] %s\n", t.name);
        const int failed_before = g_failures;
        t.fn();
        std::printf("[ %s ] %s\n\n", g_failures == failed_before ? " OK " : "FAIL", t.name);
        ++total;
    }
    std::printf("%d test(s) run, %d check(s) failed\n", total, g_failures);
    return g_failures == 0 ? 0 : 1;
}
