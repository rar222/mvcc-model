// Measures the actual CPU cost native narrow-int keys/fields save over the
// ordinary string path, for the SAME field, same data, only the routing
// differing (force_string=true vs the default) -- see project memory
// (project_native_narrow_key_proposal) for why this is a CPU/structural
// claim, not a memory one: string-key-dedup already claimed the memory win
// this proposal originally chased, before this measurement was written.
//
// One type, two key fields and two cached fields, identical values:
//   narrow_key    -- by_key_narrow_,   KeyLookupType::Exact (default routing)
//   string_key    -- by_key_,          KeyLookupType::Exact, force_string=true
//   narrow_field  -- by_cached_field_narrow_, LookupType::Exact (default)
//   string_field  -- by_cached_field_,        LookupType::Exact, force_string=true
//
// Run:  ./build/default/narrow_key_tuning_demo [N]

#include "model/model.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <memory>
#include <string>

using namespace model;

namespace {

class NarrowTuningThing final : public Object<NarrowTuningThing> {
public:
    std::int64_t narrow_key = 0;
    std::int64_t string_key = 0;
    std::int64_t narrow_field = 0;
    std::int64_t string_field = 0;

    template <class Self>
    static void define_keys(Self& s, const FieldKeyReader& v) {
        v.key<&NarrowTuningThing::narrow_key>(s.narrow_key, KeyLookupType::Exact, "narrow_key");
        v.key<&NarrowTuningThing::string_key>(s.string_key, KeyLookupType::Exact, "string_key",
                                              /*force_string=*/true);
    }

    template <class Self>
    static void define_fields(Self& s, const LookupFieldReader& v) {
        v.field<&NarrowTuningThing::narrow_field>(s.narrow_field, LookupType::Exact, "narrow_field");
        v.field<&NarrowTuningThing::string_field>(s.string_field, LookupType::Exact, "string_field",
                                                  /*force_string=*/true);
    }
};

template <class F>
double time_us(F&& f) {
    const auto t0 = std::chrono::steady_clock::now();
    f();
    return std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count();
}

// Minimum across trials -- see lookup_type_tuning_demo.cpp's best_of for
// the rationale (jitter only pushes a reading up, never down).
template <class F>
double best_of(int trials, F&& f) {
    double best = f();
    for (int t = 1; t < trials; ++t) {
        const double v = f();
        if (v < best) best = v;
    }
    return best;
}

}  // namespace

int main(int argc, char** argv) {
    const int n = argc > 1 ? std::atoi(argv[1]) : 100'000;
    if (n <= 0) {
        std::fprintf(stderr, "error: N must be a positive integer (got %d)\n", n);
        return 1;
    }

    // Insert cost: build the whole population under best_of, comparing a
    // Model whose fields are all narrow-routed against one whose fields are
    // all force_string'd -- separate Models, since force_string is a
    // per-declaration property, not a per-value one.
    auto build = [&](bool force_string) {
        return best_of(5, [&] {
            return time_us([&] {
                       Model m;
                       m.set_max_undo_list_size(0);
                       Transaction txn = m.begin();
                       for (int i = 0; i < n; ++i) {
                           auto o = std::make_unique<NarrowTuningThing>();
                           if (force_string) {
                               o->string_key = i;
                               o->string_field = i;
                           } else {
                               o->narrow_key = i;
                               o->narrow_field = i;
                           }
                           txn.create(std::move(o));
                       }
                       m.try_commit(txn);
                   }) /
                   n;
        });
    };
    const double narrow_insert_ns = build(false) * 1000.0;
    const double string_insert_ns = build(true) * 1000.0;

    // Lookup cost: one shared Model with both forms populated, so the
    // comparison isn't sensitive to anything but the routing itself.
    Model m;
    m.set_max_undo_list_size(0);
    Transaction txn = m.begin();
    for (int i = 0; i < n; ++i) {
        auto o = std::make_unique<NarrowTuningThing>();
        o->narrow_key = i;
        o->string_key = i;
        o->narrow_field = i;
        o->string_field = i;
        txn.create(std::move(o));
    }
    m.try_commit(txn);
    Snapshot s = m.snapshot();

    constexpr int kReps = 5000;
    auto lookup_key_ns = [&](bool narrow) {
        return best_of(15, [&] {
                   return time_us([&] {
                              for (int i = 0; i < kReps; ++i) {
                                  if (narrow)
                                      (void)s.find_by_key<&NarrowTuningThing::narrow_key>(i % n);
                                  else
                                      (void)s.find_by_key<&NarrowTuningThing::string_key>(i % n);
                              }
                          }) /
                          kReps;
               }) *
               1000.0;
    };
    auto lookup_field_ns = [&](bool narrow) {
        return best_of(15, [&] {
                   return time_us([&] {
                              for (int i = 0; i < kReps; ++i) {
                                  if (narrow)
                                      (void)s.find_by_field<&NarrowTuningThing::narrow_field>(i % n);
                                  else
                                      (void)s.find_by_field<&NarrowTuningThing::string_field>(i % n);
                              }
                          }) /
                          kReps;
               }) *
               1000.0;
    };

    std::printf("N=%d\n\n", n);
    std::printf("  %-28s %12s %12s %8s\n", "", "narrow", "string", "ratio");
    std::printf("  %-28s %9.1f ns %9.1f ns %7.2fx\n", "insert (per object)", narrow_insert_ns,
               string_insert_ns, string_insert_ns / narrow_insert_ns);
    std::printf("  %-28s %9.1f ns %9.1f ns %7.2fx\n", "find_by_key (per call)", lookup_key_ns(true),
               lookup_key_ns(false), lookup_key_ns(false) / lookup_key_ns(true));
    std::printf("  %-28s %9.1f ns %9.1f ns %7.2fx\n", "find_by_field (per call)",
               lookup_field_ns(true), lookup_field_ns(false),
               lookup_field_ns(false) / lookup_field_ns(true));
    std::printf(
        "\n\"ratio\" is string/narrow -- >1x means narrow storage is faster. This is the CPU/\n"
        "structural case for native narrow keys (see this file's own top comment): dedup already\n"
        "claimed the memory win these fields would otherwise have projected.\n");
    return 0;
}
