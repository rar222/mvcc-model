// Demonstrates the workflow this project's own docs point at but never show
// end to end: given a define_fields() entry, how do you actually decide
// between LookupType::Exact/Coarse/Scan for it, and -- just as important --
// how do you check, on a REAL running Model, whether that choice is still
// earning its keep? model.h's LookupType comment states the shape of the
// tradeoff ("memory cost and lookup speed both decrease monotonically
// Exact -> Coarse -> Scan") and says Coarse's routing depth is a "starting
// point to tune against slot_stats_diagnostics()'s chain-length fields, not
// fixed forever" -- this example is that tuning pass, made concrete and
// reusable. (VeryCoarse existed alongside Coarse until it was removed:
// measured to have no defensible use case at this project's target scale --
// see CLAUDE.md's "things that look like improvements but are not".)
//
// THE RULE OF THUMB (read this if you only read one part of this comment):
// two numbers are ALWAYS derivable from ONE Model::slot_stats_diagnostics()
// call plus one number you almost certainly already know -- your own live
// object count for the type that owns the field (n, below; see
// field_stats' own comment for why slot_stats_diagnostics() doesn't hand
// that to you for free, and get it from one Snapshot::for_each<T> count if
// you truly don't already track it). For a Coarse field:
//     field_leaves  = diag.by_cached_field_detail[{&typeid(YourType),
//                                                   field_tag<&YourType::f>()}].leaf_count
//     outer_leaves  = field_leaves - n            // every object contributes exactly one INNER leaf
//     candidates_per_query = n / outer_leaves     // the number find_by_field has to resolve+filter
//     saturation    = outer_leaves / bucket_ceiling   // 32768.0 for Coarse
// candidates_per_query is meaningful on its own, with no twin field and no
// benchmark: it's the number a real find_by_field call resolves and filters
// -- decide for yourself whether that's acceptable for how often you query
// it. saturation tells you how much to trust reading MORE into it: below
// ~50% of the bucket ceiling, outer_leaves is a good stand-in for the
// field's true cardinality K, which means candidates_per_query is ALREADY
// close to what Exact would give too -- there's essentially no crowding to
// worry about, whatever Coarse is saving is close to free.
//
// A THIRD number -- how many times worse than Exact's own unavoidable floor
// (crowding_factor, printed below) -- answers "is the crowding tax actually
// the dominant cost here," but it's NOT twin-free the way the first two
// are: it needs n / (an Exact field's own outer_leaves) as the comparison
// point, either a real Exact-tagged twin (what this demo has) or, below
// saturation only, outer_leaves standing in for K well enough to compute it
// anyway. Once saturated and without a twin, K is not recoverable from a
// lone Coarse field's own stats at all -- crowding_factor and
// est_bytes_if_exact both become genuinely unanswerable, not just
// inconvenient, and this program says so rather than printing a fabricated
// number.
//
// THE OTHER DIRECTION -- starting from Exact, predicting Coarse -- has no
// such gap: an Exact field's own outer_leaves IS K, exactly, always (Exact
// never merges, so there's no saturation to worry about). predict_merged
// (below) turns that K into a real statistical estimate of what a Coarse
// version of the SAME field would look like, using the "balls into bins"
// occupancy formula for candidates_per_query (measured accurate to within
// ~1% against this program's own build-and-measure runs) and a calibrated
// struct-size model for est_bytes, also measured accurate to within ~1%
// against this program's own four scenarios once calibrated against Coarse
// alone (see kNodeCountCorrectionFactor's own comment). Printed right after
// each scenario's exact_field row, against that SAME scenario's actually-measured
// coarse_field row just below it, so the prediction's accuracy is visible
// directly rather than just claimed.
//
// **Measured below**: there are two ADDITIVE read costs Coarse pays that
// Exact never does, and structural memory savings are NOT unconditional the
// way the read-cost tax is.
//   1. A fixed per-real-match resolve-and-verify tax, paid on every Coarse
//      call regardless of cardinality, because a merged bucket's members
//      are never trusted without verification, whether or not any OTHER
//      value happens to land in that same bucket too. This tax is
//      invisible to candidates_per_query/crowding_factor -- it's paid on
//      every real match too, not just the extra ones.
//   2. A crowding tax on top of that -- the part candidates_per_query DOES
//      capture -- from genuinely different values sharing a bucket and
//      having to be filtered out as false positives. Once merging is
//      saturated (K at or above the bucket count), this converges to
//      roughly n / bucket_count and stops depending on K at all: a bucket
//      collects ~(K/buckets) distinct values, each bringing ~(n/K) objects
//      along, and the K cancels. So past saturation, more objects (bigger
//      n) means more crowding even if K never changes -- crowding tax is
//      only proportional to K/buckets BELOW saturation; the HIGH and
//      EXTREME scenarios below (same n, very different K, similar Coarse
//      candidates/query once both are saturated) show this directly.
// Structural memory (fewer/smaller index entries, see
// Model::slot_stats_diagnostics()) tracks cost 2's condition, not cost 1's:
// there is NOTHING to merge, hence nothing to save, until cardinality gets
// close to the bucket count -- at K=100 the node/leaf counts below are
// IDENTICAL across Exact and Coarse. So a field with genuinely low
// cardinality relative to Coarse's own 32,768 buckets has no reason to ever
// be anything but Exact: Coarse would cost more to read and save nothing.
//
// This runs the identical workload through three parallel fields (Exact,
// Coarse, Scan -- all three holding the SAME value per object, so a query
// against any of them returns the identical real answer set) at four
// cardinalities straddling Coarse's own bucket ceiling, and prints the
// memory-side signal and the speed-side signal together per scenario, plus
// (for Coarse) the crowding diagnostic and memory-savings estimate the rule
// of thumb above describes:
//   - "leaves" / "est. bytes" -- structural leaf count and a byte estimate
//     from struct-size constants (kNodeFixedBytes & co. below) -- an
//     approximation, not a measured RSS number (see
//     examples/dedup_bucket_merge_memory_probe.cpp for that, which measures
//     the whole process rather than one field in isolation), but a
//     per-field one, which real RSS measurement can't easily give you.
//   - "read (best-of-5)" -- measured find_by_field time.
//   - "candidates/query" / "saturation" / "est. bytes if this were Exact" --
//     the stats-only crowding and savings estimate above, computed and
//     printed for real so you can see it does (or, once saturated, honestly
//     doesn't) predict the measured read-time gap.
//
// How to read the output:
//   - LOW cardinality (well under the bucket ceiling): no structural
//     savings at all (identical node/leaf counts across Exact and Coarse),
//     crowding_factor ~1.0x (nothing extra to filter), plus cost 1's fixed
//     read-time tax with nothing to show for it either way. Exact wins
//     outright here -- there's no case for Coarse.
//   - MEDIUM cardinality (still well under the ~32,768 ceiling, a realistic
//     category-style field): Coarse stays unsaturated -- a real, if modest,
//     memory saving shows up, at a crowding_factor close to 1x.
//   - HIGH cardinality (above the bucket ceiling, but objects/value still >
//     1, e.g. K=50,000 against n=100,000 -- two objects/value on average):
//     Coarse is saturated -- real crowding, and the savings estimate
//     correctly declines to guess rather than invent a number. Its absolute
//     candidates/query is already close to n/32,768 (~3) regardless of its
//     own floor being 2 rather than 1.
//   - EXTREME cardinality (fully unique per object, e.g. a serial number or
//     computed key): Coarse's absolute candidates/query barely moves from
//     the HIGH scenario (~3 either way -- the n/bucket_count convergence
//     regardless of K, once saturated, described above) even though its
//     crowding_factor reads higher (Exact's own floor dropped from ~2 to 1,
//     shrinking the denominator, not because Coarse got meaningfully more
//     crowded).
//
// Run:  ./build/default/lookup_type_tuning_demo [N] [low_K] [medium_K] [high_K] [extreme_K]

#include "model/model.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <memory>
#include <string>
#include <typeinfo>
#include <vector>

using namespace model;

namespace {

// One field per LookupType, all three holding the SAME value per object --
// so find_by_field<&TuningThing::X>(v) returns the identical real answer
// set regardless of which X you ask, and any difference in call time is
// attributable entirely to routing precision, not to the data itself.
class TuningThing final : public Object<TuningThing> {
public:
    std::string exact_field;
    std::string coarse_field;
    std::string scan_field;

    template <class Self>
    static void define_fields(Self& s, const LookupFieldReader& v) {
        v.field<&TuningThing::exact_field>(s.exact_field, LookupType::Exact, "exact_field");
        v.field<&TuningThing::coarse_field>(s.coarse_field, LookupType::Coarse, "coarse_field");
        v.field<&TuningThing::scan_field>(s.scan_field, LookupType::Scan, "scan_field");
    }
};

// n objects, k distinct values (n/k objects per value on average) -- the
// SAME value assigned to all three fields on a given object, so every
// field's index sees the identical value distribution.
void build(Model& m, int n, int k) {
    m.set_max_undo_list_size(0);
    Transaction txn = m.begin();
    for (int i = 0; i < n; ++i) {
        auto o = std::make_unique<TuningThing>();
        const std::string value = "v" + std::to_string(i % k);
        o->exact_field = value;
        o->coarse_field = value;
        o->scan_field = value;
        txn.create(std::move(o));
    }
    m.try_commit(txn);
}

template <class F>
double time_us(F&& f) {
    const auto t0 = std::chrono::steady_clock::now();
    f();
    return std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count();
}

// Minimum across trials: scheduling jitter can only push one reading UP,
// never down, so the minimum is the least-corrupted estimate of the real
// per-call cost -- same rationale persistent_map_tests.cpp's best_of and
// performance_tests.cpp's own timing helpers already use.
template <class F>
double best_of(int trials, F&& f) {
    double best = f();
    for (int t = 1; t < trials; ++t) {
        const double v = f();
        if (v < best) best = v;
    }
    return best;
}

template <auto Field>
double avg_find_time_us(const Snapshot& s, int k) {
    // A BOUNDED sample of distinct values, not all k -- scan_field's own
    // cost is O(N objects) per call regardless of k, so sweeping all of a
    // large k (e.g. k == N, the "unique per object" scenario) would cost
    // O(sample * N) per trial: fine at sample=50, not at sample=100,000.
    // Sampled evenly across the value range rather than just the first few,
    // so a query that happens to land in a coarser/denser part of the
    // routing space isn't systematically over- or under-represented.
    const int sample = std::min(k, 50);
    const int stride = std::max(1, k / sample);
    return best_of(5, [&] {
        return time_us([&] {
                   for (int i = 0; i < sample; ++i)
                       (void)s.find_by_field<Field>("v" + std::to_string((i * stride) % k));
               }) /
               sample;
    });
}

// Struct-size constants for by_cached_field_'s two leaf shapes (see
// persistent_map.h's Leaf/Node/RcBase and model.h's DedupBucketKeyOf/
// by_cached_field_merged_ -- these are the same numbers this session's
// implementation work derived and measured elsewhere, not new arithmetic):
//   Node       = RcBase(16B) + bitmap(4B) + is_leaf(4B) + slots ptr(8B) = 32B
//                fixed, PLUS 8B (one RcHandle) per occupied slot -- the
//                latter is exactly total_capacity in pmap::SlotStats.
//   Outer leaf, Exact/Cache field (pmap::DedupMap<std::string,...>, StringHash
//                is NOT declared perfect): RcBase(16B) + PersistentSet
//                handle(24B) + a real ChainLink(8B) = 48B.
//   Outer leaf, Coarse field (by_cached_field_merged_,
//                pmap::IdentityHash64 IS declared perfect): RcBase(16B) +
//                PersistentSet handle(24B) + NoChain(0B) = 40B.
//   Inner leaf (every field alike -- PersistentSet<Id,IdHash>, IdHash is
//                perfect): RcBase(16B) + Id(8B) + NoChain(0B) = 24B.
constexpr double kNodeFixedBytes = 32.0;
constexpr double kSlotBytes = 8.0;
constexpr double kExactOuterLeafBytes = 48.0;
constexpr double kMergedOuterLeafBytes = 40.0;
constexpr double kInnerLeafBytes = 24.0;

// Coarse's routing-precision ceiling (model.cpp's max_shift_for: 15 bits)
// expressed as a bucket count.
constexpr double kCoarseBucketCeiling = 32768.0;

// Below this fraction of the bucket ceiling, outer_leaves (below) is a
// trustworthy stand-in for the field's real cardinality K -- few enough
// distinct values are colliding that merging hasn't meaningfully happened
// yet, so "what this field would cost as Exact" is a real estimate, not a
// guess. At or above it, K is no longer recoverable from this field's own
// stats at all (merging has already collapsed distinct values together
// indistinguishably) -- reporting an Exact-equivalent cost past this point
// would be fabricating a number, not estimating one.
constexpr double kSaturationTrustThreshold = 0.5;

struct FieldStats {
    bool indexed = false;
    std::size_t node_count = 0;
    std::size_t leaf_count = 0;
    double outer_leaves = 0.0;
    double est_bytes = 0.0;
    double avg_candidates_per_query = 0.0;  // object_count / outer_leaves -- the read-cost predictor
    double saturation = -1.0;               // outer_leaves / bucket_ceiling; -1 for Exact (no ceiling)
    double est_bytes_if_exact = -1.0;       // -1 == not reliably estimable (see kSaturationTrustThreshold)
};

// leaf_count (pmap::SlotStats) folds outer and inner leaves into one number
// (see by_cached_field_'s/by_cached_field_merged_'s own comments on why --
// DedupMap/the merged map both fold their bucket's structure in via
// accumulate_structure_only). Splitting them back out doesn't need any new
// API: every object contributes exactly one inner leaf per indexed field
// (one membership Id per field, always), so inner_leaves == n and
// outer_leaves == leaf_count - n follow directly from n. In this demo n is
// already known; a real caller who doesn't already know their own object
// count can get it from one Snapshot::for_each<T> count (see the top-of-
// file comment for why by_type_detail alone doesn't give it to you for
// free).
FieldStats field_stats(const Model& m, const void* field, bool merged, int n, double bucket_ceiling) {
    FieldStats fs;
    const auto diag = m.slot_stats_diagnostics();
    const FieldLookupKey key{&typeid(TuningThing), field};
    const auto it = diag.by_cached_field_detail.find(key);
    if (it == diag.by_cached_field_detail.end()) return fs;  // Scan: not indexed at all
    fs.indexed = true;
    fs.node_count = it->second.node_count;
    fs.leaf_count = it->second.leaf_count;
    const double inner_leaves = static_cast<double>(n);
    fs.outer_leaves = static_cast<double>(fs.leaf_count) - inner_leaves;
    fs.avg_candidates_per_query = fs.outer_leaves > 0 ? inner_leaves / fs.outer_leaves : 0.0;
    const double outer_bytes = merged ? kMergedOuterLeafBytes : kExactOuterLeafBytes;
    fs.est_bytes = static_cast<double>(it->second.node_count) * kNodeFixedBytes +
                   static_cast<double>(it->second.total_capacity) * kSlotBytes +
                   fs.outer_leaves * outer_bytes + inner_leaves * kInnerLeafBytes;
    if (merged && bucket_ceiling > 0.0) {
        fs.saturation = fs.outer_leaves / bucket_ceiling;
        if (fs.saturation < kSaturationTrustThreshold) {
            // outer_leaves stands in for K here: re-run the SAME formula
            // with Exact's outer-leaf size instead of the merged one, same
            // node/slot counts. Reusing this field's own (shallower, merged)
            // routing structure rather than recomputing what an Exact trie
            // over the same K would need slightly UNDERSTATES Exact's true
            // node cost, so this is a conservative (savings-understating,
            // never overstating) estimate, not an exact one either way.
            fs.est_bytes_if_exact = static_cast<double>(it->second.node_count) * kNodeFixedBytes +
                                    static_cast<double>(it->second.total_capacity) * kSlotBytes +
                                    fs.outer_leaves * kExactOuterLeafBytes + inner_leaves * kInnerLeafBytes;
        }
    }
    return fs;
}

struct MergedPrediction {
    double predicted_outer_leaves = 0.0;
    double predicted_candidates_per_query = 0.0;
    double predicted_est_bytes = 0.0;
};

// The FORWARD direction: given an EXACT field's own true cardinality k
// (always exact -- Exact never merges, so its outer_leaves IS k, precisely,
// with none of est_bytes_if_exact's saturation caveat), predict what a
// Coarse version of the SAME field would look like, without
// having to build it and measure.
//
// The leaf-count half is real probability theory: throwing k distinct
// values uniformly at random into `bucket_ceiling` coarse-hash buckets is
// the classic "balls into bins" occupancy problem -- the EXPECTED number of
// non-empty buckets is bucket_ceiling * (1 - (1 - 1/bucket_ceiling)^k),
// well-approximated here by bucket_ceiling * (1 - exp(-k/bucket_ceiling)).
// Still an expectation, not a guarantee (StringHash's real output for your
// actual values could distribute somewhat unevenly), but unlike
// est_bytes_if_exact above, there's no "am I past the trustworthy point"
// regime to worry about -- k itself is never in question.
//
// The node-count half is a much rougher model, in two parts: a 32-ary trie
// (TrieCore's own branching factor, persistent_map.h) needs roughly
// leaves/(32-1) internal nodes to route `leaves` entries in a reasonably
// balanced tree, each assumed close to full 32-way fanout -- applied once
// to the OUTER (routing) structure, and again to the INNER structure (every
// merged bucket's own PersistentSet<Id,IdHash>, which needs real internal
// nodes once it holds more than one member -- a singleton bucket is a
// leaf-root, zero Nodes, exactly like the outer trie's own leaf-root case).
// Total inner leaves summed across every bucket is always n regardless of
// how they're distributed, so total inner nodes needed (same leaves/31
// heuristic, this time never going below zero) is (n - predicted_outer_
// leaves) / 31.
//
// Even with both halves, this idealized "perfectly packed 32-ary tree"
// model under-predicts real bytes -- NOT because of any std::vector padding
// (Node::slots is a bare RcHandle*, exactly allocated every time --
// persistent_map.h: "grow_insert always reallocates to the exact new count
// -- there is no spare capacity to grow into, ever, by design" -- an
// earlier version of this comment claimed vector padding was the cause,
// which was simply wrong, not just imprecise). The real cause: "leaves/31"
// assumes every internal node ends up with the full 32-way fanout
// TrieCore's branching factor allows, but a real trie built from actual
// (effectively random) hash-bit distributions fills GRADUALLY as more
// entries compete for the same prefix -- most nodes hold well under 32
// children until the population routed through them is large. This is the
// exact phenomenon examples/slot_stats_probe.cpp's own top comment already
// documents for by_type_ ("saturates by roughly N=100,000 and then just
// fills toward 32/32 per node as N climbs") -- below that population, real
// average fanout is meaningfully under 32, so real node count (and the
// fixed per-node overhead that comes with each one) is meaningfully ABOVE
// leaves/31. kNodeCountCorrectionFactor corrects the NODE COUNT for that
// (not the slots array -- see the exact identity below), calibrated
// empirically against this program's own LOW/MEDIUM/HIGH/EXTREME output,
// not derived from first principles.
//
// ideal_nodes (outer_ideal + inner_ideal, before this factor is applied)
// telescopes to a constant, n/31, regardless of outer_leaves -- so at fixed
// n, every one of this program's four Coarse scenarios needs almost exactly
// the same correction factor to match its own real measured bytes (checked
// directly: 7.7-8.2 across LOW/MEDIUM/HIGH/EXTREME, a ~7% spread). 8.0 lands
// every scenario within ~1% of the real number. (An earlier version of this
// constant, back when it also had to fit LookupType::VeryCoarse's very
// differently-shaped bucket population -- a handful of buckets holding
// hundreds of members each, vs Coarse's many buckets holding a handful each
// -- could only manage an 80-90% compromise across both; removing
// VeryCoarse removed that heterogeneity, not just the need to predict it.)
constexpr double kNodeCountCorrectionFactor = 8.0;

MergedPrediction predict_merged(double k, double n, double bucket_ceiling) {
    MergedPrediction p;
    p.predicted_outer_leaves = bucket_ceiling * (1.0 - std::exp(-k / bucket_ceiling));
    p.predicted_candidates_per_query = n / p.predicted_outer_leaves;
    const double outer_nodes = (p.predicted_outer_leaves / 31.0) * kNodeCountCorrectionFactor;
    const double inner_nodes =
        (std::max(0.0, n - p.predicted_outer_leaves) / 31.0) * kNodeCountCorrectionFactor;
    const double total_nodes = outer_nodes + inner_nodes;
    // Node::slots being exactly-allocated (see above) makes total slot
    // COUNT across every node an exact structural identity, not an
    // estimate: every entry in the trie (leaf or sub-node) occupies exactly
    // one slot in its parent, so total slots == total_nodes + total_leaves
    // (every node except the root is referenced by exactly one parent
    // slot) -- no separate fanout assumption needed for this part at all.
    const double outer_slots = outer_nodes + p.predicted_outer_leaves;
    const double inner_slots = inner_nodes + n;
    p.predicted_est_bytes = total_nodes * kNodeFixedBytes + (outer_slots + inner_slots) * kSlotBytes +
                            p.predicted_outer_leaves * kMergedOuterLeafBytes + n * kInnerLeafBytes;
    return p;
}

// exact_baseline_candidates: the EXACT field's own avg_candidates_per_query
// (real matches per query -- N/K exactly, never approximate, since Exact is
// never merged). This is the unavoidable floor every LookupType pays at
// least -- comparing a merged field's candidate count against a flat 1
// would be wrong whenever a value is shared by more than one object (N/K >
// 1): Exact itself already resolves N/K matches per query, and a merged
// field's crowding tax is only the EXTRA candidates beyond that floor, not
// its whole candidate count.
void print_row(const char* name, const FieldStats& fs, double read_us, double exact_baseline_candidates,
              const char* note = "") {
    const std::string suffix = note[0] ? std::string("  ") + note : std::string();
    if (!fs.indexed) {
        std::printf("  %-16s %10s  %12s  %10.3f us%s\n", name, "n/a", "n/a", read_us, suffix.c_str());
        return;
    }
    std::printf("  %-16s %10zu  %9.0f KB  %10.3f us%s\n", name, fs.leaf_count, fs.est_bytes / 1024.0,
               read_us, suffix.c_str());
    if (fs.saturation < 0.0) return;  // Exact/Scan: no crowding concept applies
    const double crowding_factor =
        exact_baseline_candidates > 0.0 ? fs.avg_candidates_per_query / exact_baseline_candidates : 0.0;
    std::printf(
        "      candidates/query (est.): %6.1f  (exact's own floor: %5.1f, so %.1fx)   saturation: "
        "%5.1f%% of bucket ceiling\n",
        fs.avg_candidates_per_query, exact_baseline_candidates, crowding_factor, fs.saturation * 100.0);
    if (fs.est_bytes_if_exact >= 0.0) {
        const double saved_kb = (fs.est_bytes_if_exact - fs.est_bytes) / 1024.0;
        std::printf(
            "      est. bytes if this were Exact instead: %.0f KB  ->  this LookupType is saving "
            "~%.0f KB (%.0f%%) for %.1fx the per-query candidates Exact itself would already have\n",
            fs.est_bytes_if_exact / 1024.0, saved_kb, 100.0 * saved_kb * 1024.0 / fs.est_bytes_if_exact,
            crowding_factor);
    } else {
        std::printf(
            "      est. bytes if this were Exact instead: N/A -- saturated (>= %.0f%% of the bucket "
            "ceiling), real cardinality isn't recoverable from this field's own stats anymore; try "
            "Exact directly for a real number\n",
            kSaturationTrustThreshold * 100.0);
    }
}

void run_scenario(const char* label, int n, int k) {
    std::printf("\n=== %s: N=%d objects, K=%d distinct values (~%d objects/value) ===\n", label, n, k,
               n / k);
    Model m;
    build(m, n, k);
    Snapshot s = m.snapshot();

    const FieldStats exact = field_stats(m, field_tag<&TuningThing::exact_field>(), false, n, 0.0);
    const FieldStats coarse =
        field_stats(m, field_tag<&TuningThing::coarse_field>(), true, n, kCoarseBucketCeiling);
    const FieldStats scan = field_stats(m, field_tag<&TuningThing::scan_field>(), false, n, 0.0);

    const double exact_us = avg_find_time_us<&TuningThing::exact_field>(s, k);
    const double coarse_us = avg_find_time_us<&TuningThing::coarse_field>(s, k);
    const double scan_us = avg_find_time_us<&TuningThing::scan_field>(s, k);

    // Memory vs speed, side by side -- est. bytes is the whole index for
    // this ONE field (structural node/slot memory + every outer and inner
    // leaf, per the constants above); read cost is the same best-of-5
    // average find_by_field measurement as before. Reading down a column
    // shows the trade in isolation (bytes: Exact >= Coarse, per LookupType's
    // own doc comment); reading across a row shows what that trade actually
    // costs or buys for THIS field's real cardinality.
    std::printf("  %-16s %10s  %12s  %13s\n", "field", "leaves", "est. bytes", "read (best-of-5)");
    print_row("exact_field", exact, exact_us, 0.0);
    // Forward prediction, FROM exact_field's own always-precise cardinality
    // -- see predict_merged's own comment. Printed against the SAME
    // scenario's actually-measured coarse_field row right below, so the
    // prediction's accuracy is visible directly, not just asserted.
    if (exact.outer_leaves > 0.0) {
        const auto pred_coarse =
            predict_merged(exact.outer_leaves, static_cast<double>(n), kCoarseBucketCeiling);
        std::printf(
            "      if Coarse instead (predicted from exact_field's own K=%.0f): ~%.1f "
            "candidates/query, ~%.0f KB\n",
            exact.outer_leaves, pred_coarse.predicted_candidates_per_query,
            pred_coarse.predicted_est_bytes / 1024.0);
    }
    print_row("coarse_field", coarse, coarse_us, exact.avg_candidates_per_query);
    print_row("scan_field", scan, scan_us, 0.0, "<- O(N objects)/call, no index memory at all");
}

}  // namespace

int main(int argc, char** argv) {
    const int n = argc > 1 ? std::atoi(argv[1]) : 100'000;
    // Straddling Coarse's own ~32,768-bucket ceiling (model.cpp's
    // max_shift_for) on purpose: LOW sits far under it (nothing merges),
    // MEDIUM sits well under it too but high enough to be a realistic
    // category-style field, HIGH sits above it (real crowding, still >1
    // objects/value), EXTREME is fully unique per object (objects/value ==
    // 1) -- HIGH and EXTREME exist side by side specifically to show that
    // once Coarse is saturated, crowding_factor keeps climbing with n even
    // though K alone doesn't obviously predict how much (see the
    // top-of-file comment's correction of its own earlier K/buckets claim).
    const int low_k = argc > 2 ? std::atoi(argv[2]) : 100;
    const int medium_k = argc > 3 ? std::atoi(argv[3]) : 5'000;
    const int high_k = argc > 4 ? std::atoi(argv[4]) : 50'000;
    const int extreme_k = argc > 5 ? std::atoi(argv[5]) : n;  // unique per object

    // atoi() returns 0 for anything non-numeric (e.g. a blank/whitespace
    // arg), and every K below is a divisor (candidates_per_query, the
    // "objects/value" header) -- rejecting non-positive values here up
    // front turns a would-be SIGFPE into a clear message.
    for (const int v : {n, low_k, medium_k, high_k, extreme_k}) {
        if (v <= 0) {
            std::fprintf(stderr,
                        "error: N and every K must be positive integers (got %d) -- usage: %s [N] "
                        "[low_K] [medium_K] [high_K] [extreme_K]\n",
                        v, argv[0]);
            return 1;
        }
    }

    run_scenario("LOW cardinality (e.g. a status/category field)", n, low_k);
    run_scenario("MEDIUM cardinality (well under the bucket ceiling)", n, medium_k);
    run_scenario("HIGH cardinality (above the bucket ceiling, still several objects/value)", n, high_k);
    run_scenario("EXTREME cardinality (e.g. a unique serial number / computed key)", n, extreme_k);

    std::printf(
        "\nRule of thumb -- two numbers need only ONE Model::slot_stats_diagnostics() call plus\n"
        "your own object count (n), no benchmark and no Exact-tagged twin field required:\n"
        "  outer_leaves = field_leaf_count - n\n"
        "  candidates_per_query = n / outer_leaves   <- what a real find_by_field call resolves;\n"
        "                                                judge this alone against your own latency budget\n"
        "  saturation = outer_leaves / bucket_ceiling (32,768 for Coarse)\n"
        "Below ~50%% saturation, candidates_per_query is ALREADY close to what Exact would give --\n"
        "little to no crowding, so whatever memory this LookupType is saving (est. bytes above) is\n"
        "close to free (see LOW: identical est. bytes across all three -- nothing merged, nothing\n"
        "saved either). A THIRD number, crowding_factor (candidates_per_query vs an Exact field's\n"
        "own floor), tells you how much of candidates_per_query is genuine crowding tax rather than\n"
        "unavoidable real matches -- but it needs a real (or, below saturation, an implied) Exact\n"
        "comparison point, and once saturation passes ~50%% without one, it -- and the memory-savings\n"
        "estimate -- become genuinely unanswerable from this field's own stats, not just\n"
        "inconvenient: this program says so rather than fabricating a number past that line. Read a\n"
        "scenario's whole block as one trade: a LookupType only earns its keep by shrinking est.\n"
        "bytes noticeably more than candidates_per_query costs you, for how often YOU actually\n"
        "query it -- not by either number in isolation.\n");
    return 0;
}
