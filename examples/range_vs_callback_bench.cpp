// Timing comparison for find_by_field/find_referrers's cache-hit branch
// (backed by the two bucket-backed indexes, by_cached_field and
// by_cached_reference) across every access form that exists for it:
//
//   range_by_field / range_referrers   -- range-based for
//   for_each_by_field / for_each_referrers -- callback, never stops early
//   all_of_by_field / all_of_referrers, STOP AFTER THE FIRST MATCH -- pred
//     returns false immediately
//   all_of_by_field / all_of_referrers, VISIT EVERY MATCH -- pred always
//     returns true, so it never actually short-circuits
//
// Two distinct object types exercise the two distinct lookup families:
// Widget (a plain cached VALUE field, Root::by_cached_field) and Gadget (a
// cached REFERENCE field, Root::by_cached_reference) -- the same two-level
// bucket shape, just keyed by a value string vs. a target Id (see model.h's
// Root and src/model.cpp's cached_field_short_circuit_raw/cached_referrer_
// short_circuit_raw). Both range_* forms were added in the same session that
// measured a bare PersistentSet range-for at ~1.1x a callback's cost for a
// full O(n) scan (persistent_map_tests.cpp's speed_callback_vs_range_for_
// full_scan) -- this program answers the same question at the Snapshot
// level, where the range also has to resolve each Id through find_raw() and
// downcast via cast<ClassT>, not just walk a bare trie.
//
// A FIXED bucket count (kNumBuckets), not one that scales with population
// (contrast cached_reference_bench.cpp's bucket_count_for, which deliberately
// keeps matches/bucket constant to isolate a different variable): here
// matches-per-query grows linearly with n, since that -- not the population
// size itself -- is what the per-element iteration cost documented above
// should scale with. Run under asan/tsan too, same as every other example:
// this is ordinary application code, not a privileged fast path.

#include "model/model.h"

#include <cassert>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <string>
#include <vector>

using namespace model;

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

class Bucket final : public Object<Bucket> {
public:
    std::string name;
};

// Exercises range_by_field / for_each_by_field / all_of_by_field's cache-hit
// branch: category is a low-cardinality VALUE field, indexed via
// define_fields() tagged LookupType::Cache.
class Widget final : public Object<Widget> {
public:
    std::int64_t category = 0;
    std::int64_t payload = 0;

    template <class Self>
    static void define_fields(Self& s, const LookupFieldReader& v) {
        v.key<&Widget::category>(s.category, LookupType::Cache, "category");
    }
};

// Exercises range_referrers / for_each_referrers / all_of_referrers's
// cache-hit branch: bucket is a Ref<Bucket>, indexed via define_references()
// tagged LookupType::Cache.
class Gadget final : public Object<Gadget> {
public:
    Ref<Bucket> bucket;
    std::int64_t payload = 0;

    template <class Self, class V>
    static void define_references(Self& s, V&& v) {
        v(field_tag<&Gadget::bucket>(), s.bucket, LookupType::Cache, "bucket");
    }
};

constexpr int kNumBuckets = 8;  // fixed -- matches/query grows with n, see file header

// ---------------------------------------------------------------------------
// Timing helpers
// ---------------------------------------------------------------------------

// Minimum across trials: scheduling jitter/cache effects can only push a
// single reading UP, never down -- same rationale as tests/
// performance_tests.cpp's best_of / tests/persistent_map_tests.cpp's
// best_of, duplicated here rather than shared since examples/ has no
// dependency on tests/.
template <class F>
double best_of_us(int trials, F&& f) {
    double best = -1.0;
    for (int t = 0; t < trials; ++t) {
        const auto t0 = std::chrono::steady_clock::now();
        f();
        const auto t1 = std::chrono::steady_clock::now();
        const double us = std::chrono::duration<double, std::micro>(t1 - t0).count();
        if (best < 0.0 || us < best) best = us;
    }
    return best;
}

constexpr int kTrials = 9;

// ---------------------------------------------------------------------------
// Widget population (by_cached_field)
// ---------------------------------------------------------------------------

void seed_widgets(Model& m, int n) {
    for (int i = 0; i < n; i += 1000) {
        Transaction txn = m.begin();
        for (int j = 0; j < 1000 && i + j < n; ++j) {
            auto w = std::make_unique<Widget>();
            w->category = (i + j) % kNumBuckets;
            w->payload = i + j;
            txn.create(std::move(w));
        }
        const CommitResult res = m.try_commit(txn);
        assert(res.status == CommitStatus::Committed);
    }
}

struct FieldResult {
    int n;
    std::size_t matches;
    double range_us;
    double range_stop_us;
    double for_each_us;
    double all_of_stop_us;
    double all_of_all_us;
};

FieldResult bench_by_cached_field(int n) {
    Model m;
    seed_widgets(m, n);
    Snapshot s = m.snapshot();
    constexpr std::int64_t target = 0;

    std::size_t matches = 0;
    s.for_each_by_field<&Widget::category>(target, [&](const Widget&) { ++matches; });

    volatile std::int64_t sink = 0;

    const double range_us = best_of_us(kTrials, [&] {
        std::int64_t total = 0;
        for (const Widget& w : s.range_by_field<&Widget::category>(target)) total += w.payload;
        sink = total;
    });
    // `break` on the very first entry: a range-for's own way of stopping the
    // walk early, same intent as all_of's pred returning false immediately.
    // Unlike the all_of/for_each family, nothing needs a special "stopping"
    // variant here -- the caller just breaks, so this measures the ordinary
    // range-for loop, not a different API entry point.
    const double range_stop_us = best_of_us(kTrials, [&] {
        std::int64_t total = 0;
        for (const Widget& w : s.range_by_field<&Widget::category>(target)) {
            total += w.payload;
            break;
        }
        sink = total;
    });
    const double for_each_us = best_of_us(kTrials, [&] {
        std::int64_t total = 0;
        s.for_each_by_field<&Widget::category>(target,
                                                [&](const Widget& w) { total += w.payload; });
        sink = total;
    });
    const double all_of_stop_us = best_of_us(kTrials, [&] {
        std::int64_t total = 0;
        s.all_of_by_field<&Widget::category>(target, [&](const Widget& w) {
            total += w.payload;
            return false;  // stop after the first match
        });
        sink = total;
    });
    const double all_of_all_us = best_of_us(kTrials, [&] {
        std::int64_t total = 0;
        s.all_of_by_field<&Widget::category>(target, [&](const Widget& w) {
            total += w.payload;
            return true;  // never stops -- visits every match, same as for_each/range
        });
        sink = total;
    });
    (void)sink;

    return {n, matches, range_us, range_stop_us, for_each_us, all_of_stop_us, all_of_all_us};
}

// ---------------------------------------------------------------------------
// Gadget population (by_cached_reference)
// ---------------------------------------------------------------------------

struct GadgetFixture {
    Model m;
    std::vector<Ref<Bucket>> buckets;
};

void seed_gadgets(GadgetFixture& fx, int n) {
    Transaction btxn = fx.m.begin();
    for (int i = 0; i < kNumBuckets; ++i) {
        auto b = std::make_unique<Bucket>();
        b->name = "B" + std::to_string(i);
        fx.buckets.push_back(btxn.create(std::move(b)));
    }
    const CommitResult bres = fx.m.try_commit(btxn);
    assert(bres.status == CommitStatus::Committed);
    for (auto& r : fx.buckets) r = bres.to_real(r);

    for (int i = 0; i < n; i += 1000) {
        Transaction txn = fx.m.begin();
        for (int j = 0; j < 1000 && i + j < n; ++j) {
            auto g = std::make_unique<Gadget>();
            g->bucket = fx.buckets[static_cast<std::size_t>(i + j) % fx.buckets.size()];
            g->payload = i + j;
            txn.create(std::move(g));
        }
        const CommitResult res = fx.m.try_commit(txn);
        assert(res.status == CommitStatus::Committed);
    }
}

FieldResult bench_cached_referrers(int n) {
    GadgetFixture fx;
    seed_gadgets(fx, n);
    Snapshot s = fx.m.snapshot();
    const Ref<Bucket> target = fx.buckets[0];

    std::size_t matches = 0;
    s.for_each_referrers<&Gadget::bucket>(target, [&](const Gadget&) { ++matches; });

    volatile std::int64_t sink = 0;

    const double range_us = best_of_us(kTrials, [&] {
        std::int64_t total = 0;
        for (const Gadget& g : s.range_referrers<&Gadget::bucket>(target)) total += g.payload;
        sink = total;
    });
    const double range_stop_us = best_of_us(kTrials, [&] {
        std::int64_t total = 0;
        for (const Gadget& g : s.range_referrers<&Gadget::bucket>(target)) {
            total += g.payload;
            break;
        }
        sink = total;
    });
    const double for_each_us = best_of_us(kTrials, [&] {
        std::int64_t total = 0;
        s.for_each_referrers<&Gadget::bucket>(target,
                                              [&](const Gadget& g) { total += g.payload; });
        sink = total;
    });
    const double all_of_stop_us = best_of_us(kTrials, [&] {
        std::int64_t total = 0;
        s.all_of_referrers<&Gadget::bucket>(target, [&](const Gadget& g) {
            total += g.payload;
            return false;  // stop after the first match
        });
        sink = total;
    });
    const double all_of_all_us = best_of_us(kTrials, [&] {
        std::int64_t total = 0;
        s.all_of_referrers<&Gadget::bucket>(target, [&](const Gadget& g) {
            total += g.payload;
            return true;  // never stops -- visits every match, same as for_each/range
        });
        sink = total;
    });
    (void)sink;

    return {n, matches, range_us, range_stop_us, for_each_us, all_of_stop_us, all_of_all_us};
}

// ---------------------------------------------------------------------------

void print_table(const char* title, const char* type_label, const std::vector<FieldResult>& results) {
    std::printf("\n=== %s ===\n", title);
    std::printf("%10s  %10s  %12s  %14s  %12s  %14s  %12s\n", "items", "matches", "range/all (us)",
               "range/1st (us)", "for_each (us)", "all_of/1st (us)", "all_of/all (us)");
    for (const FieldResult& r : results) {
        std::printf("%10d  %10zu  %12.2f  %14.3f  %12.2f  %14.3f  %12.2f\n", r.n, r.matches, r.range_us,
                    r.range_stop_us, r.for_each_us, r.all_of_stop_us, r.all_of_all_us);
    }
    std::printf(
        "\n%s: range/all, for_each, and all_of/all all visit every one of `matches` entries --\n"
        "expect them within a small constant factor of each other (see this file's header for\n"
        "what that factor was measured at for a bare PersistentSet). range/1st (break on the\n"
        "first loop iteration) and all_of/1st (pred returns false immediately) both visit\n"
        "exactly ONE entry regardless of `matches` -- expect both roughly FLAT across the sweep,\n"
        "and far cheaper than the three full-scan forms once `matches` is large: that gap is what\n"
        "a genuine short-circuit buys, whether it's spelled `break` or `return false`, over an\n"
        "API (for_each_*) that can't stop the underlying walk early at all.\n",
        type_label);
}

int main() {
    // Up to 200,000 objects per type, 8 fixed buckets -> up to 25,000
    // matches for the one bucket every query below targets.
    const std::vector<int> sizes = {2000, 20000, 200000};

    std::vector<FieldResult> field_results;
    for (int n : sizes) field_results.push_back(bench_by_cached_field(n));

    std::vector<FieldResult> referrer_results;
    for (int n : sizes) referrer_results.push_back(bench_cached_referrers(n));

    print_table("by_cached_field (Widget::category)", "by_cached_field", field_results);
    print_table("cached_referrers (Gadget::bucket)", "cached_referrers", referrer_results);

    std::printf(
        "\nNot asserted: this is a measurement to inform which for_each_*/all_of_* call sites are\n"
        "worth converting to range-for, not a regression gate -- there is no \"correct\" ratio to\n"
        "pin here.\n");
    return 0;
}
