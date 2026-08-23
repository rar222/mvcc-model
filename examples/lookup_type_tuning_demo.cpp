// Demonstrates the workflow this project's own docs point at but never show
// end to end: given a define_fields() entry, how do you actually decide
// between LookupType::Exact/Coarse/VeryCoarse/Scan for it, and -- just as
// important -- how do you check, on a REAL running Model, whether that
// choice is still earning its keep? model.h's LookupType comment states the
// shape of the tradeoff ("memory cost and lookup speed both decrease
// monotonically Exact -> Coarse -> VeryCoarse -> Scan") and says
// Coarse/VeryCoarse's routing depths are "starting points to tune against
// slot_stats_diagnostics()'s chain-length fields, not fixed forever" -- this
// example is that tuning pass, made concrete and reusable.
//
// THE RULE OF THUMB (read this if you only read one part of this comment):
// two numbers are ALWAYS derivable from ONE Model::slot_stats_diagnostics()
// call plus one number you almost certainly already know -- your own live
// object count for the type that owns the field (n, below; see
// field_stats' own comment for why slot_stats_diagnostics() doesn't hand
// that to you for free, and get it from one Snapshot::for_each<T> count if
// you truly don't already track it). For a Coarse/VeryCoarse field:
//     field_leaves  = diag.by_cached_field_detail[{&typeid(YourType),
//                                                   field_tag<&YourType::f>()}].leaf_count
//     outer_leaves  = field_leaves - n            // every object contributes exactly one INNER leaf
//     candidates_per_query = n / outer_leaves     // the number find_by_field has to resolve+filter
//     saturation    = outer_leaves / bucket_ceiling   // 32768.0 for Coarse, 1024.0 for VeryCoarse
// candidates_per_query is meaningful on its own, with no twin field and no
// benchmark: it's the number a real find_by_field call resolves and filters
// -- decide for yourself whether that's acceptable for how often you query
// it. saturation tells you how much to trust reading MORE into it: below
// ~50% of the bucket ceiling, outer_leaves is a good stand-in for the
// field's true cardinality K, which means candidates_per_query is ALREADY
// close to what Exact would give too -- there's essentially no crowding to
// worry about, whatever this LookupType is saving is close to free.
//
// A THIRD number -- how many times worse than Exact's own unavoidable floor
// (crowding_factor, printed below) -- answers "is the crowding tax actually
// the dominant cost here," but it's NOT twin-free the way the first two
// are: it needs n / (an Exact field's own outer_leaves) as the comparison
// point, either a real Exact-tagged twin (what this demo has) or, below
// saturation only, outer_leaves standing in for K well enough to compute it
// anyway. Once saturated and without a twin, K is not recoverable from a
// lone Coarse/VeryCoarse field's own stats at all -- crowding_factor and
// est_bytes_if_exact both become genuinely unanswerable, not just
// inconvenient, and this program says so rather than printing a fabricated
// number.
//
// THE OTHER DIRECTION -- starting from Exact, predicting Coarse/VeryCoarse
// -- has no such gap: an Exact field's own outer_leaves IS K, exactly,
// always (Exact never merges, so there's no saturation to worry about).
// predict_merged (below) turns that K into a real statistical estimate of
// what a Coarse/VeryCoarse version of the SAME field would look like,
// using the "balls into bins" occupancy formula for candidates_per_query
// (measured accurate to within ~1% against this program's own build-and-
// measure runs) and a calibrated struct-size model for est_bytes -- the
// noisier of the two, and NOT close to ~1%: measured against this program's
// own four scenarios, predicted bytes land at roughly 80-90% of the real
// number every time, never over. See kNodeCountCorrectionFactor's own
// comment for why a single constant can't do better than that band. Printed
// right after each scenario's exact_field row, against that SAME scenario's
// actually-measured coarse_field/very_coarse_field rows just below it, so
// the prediction's accuracy is visible directly rather than just claimed.
//
// **Measured below, and worth stating plainly since it corrected two naive
// predictions this comment originally made**: there are two ADDITIVE read
// costs Coarse/VeryCoarse pay that Exact never does, and structural memory
// savings are NOT unconditional the way the read-cost tax is.
//   1. A fixed per-real-match resolve-and-verify tax, paid on every
//      Coarse/VeryCoarse call regardless of cardinality -- even at K=100
//      against VeryCoarse's 1,024 buckets (few enough distinct values that
//      almost none of them actually share a bucket, i.e. crowding_factor
//      measures ~1.0x below), VeryCoarse still measured ~3x Exact's
//      per-call time in the run this comment was written against, because
//      a merged bucket's members are never trusted without verification,
//      whether or not any OTHER value happens to land in that same bucket
//      too. This tax is invisible to candidates_per_query/crowding_factor
//      -- it's paid on every real match too, not just the extra ones.
//   2. A crowding tax on top of that -- the part candidates_per_query DOES
//      capture -- from genuinely different values sharing a bucket and
//      having to be filtered out as false positives. Once merging is
//      saturated (K at or above the bucket count), this converges to
//      roughly n / bucket_count and stops depending on K at all: a bucket
//      collects ~(K/buckets) distinct values, each bringing ~(n/K) objects
//      along, and the K cancels. So past saturation, more objects (bigger
//      n) means more crowding even if K never changes -- the earlier
//      version of this comment claimed the crowding tax was "proportional
//      to K/buckets," which is only true BELOW saturation; the measured
//      MEDIUM and EXTREME scenarios below (same n, very different K, similar
//      VeryCoarse crowding_factor once both are saturated) are what caught
//      the error.
// Structural memory (fewer/smaller index entries, see
// Model::slot_stats_diagnostics()) tracks cost 2's condition, not cost 1's:
// there is NOTHING to merge, hence nothing to save, until cardinality gets
// close to the bucket count -- at K=100 the node/leaf counts below are
// IDENTICAL across Exact/Coarse/VeryCoarse. So a field with genuinely low
// cardinality relative to even VeryCoarse's 1,024 buckets has no reason to
// ever be anything but Exact: Coarse/VeryCoarse would cost more to read and
// save nothing.
//
// This runs the identical workload through four parallel fields (Exact,
// Coarse, VeryCoarse, Scan -- all four holding the SAME value per object, so
// a query against any of them returns the identical real answer set) at
// four cardinalities straddling the two bucket counts, and prints the
// memory-side signal and the speed-side signal together per scenario, plus
// (for Coarse/VeryCoarse) the crowding diagnostic and memory-savings
// estimate the rule of thumb above describes:
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
//   - LOW cardinality (well under both bucket counts): no structural
//     savings at all (identical node/leaf counts across all three),
//     crowding_factor ~1.0x (nothing extra to filter), plus cost 1's fixed
//     read-time tax with nothing to show for it either way. Exact wins
//     outright here -- there's no case for Coarse or VeryCoarse.
//   - MEDIUM cardinality (above VeryCoarse's ~1,024, still under Coarse's
//     ~32,768): Coarse stays unsaturated -- a real, if modest, memory
//     saving shows up, at a crowding_factor close to 1x. VeryCoarse is
//     already saturated at this K -- real crowding, and the savings
//     estimate correctly declines to guess rather than invent a number.
//   - HIGH cardinality (above both bucket counts, but objects/value still >
//     1, e.g. K=50,000 against n=100,000 -- two objects/value on average):
//     both Coarse and VeryCoarse are saturated, but Coarse's absolute
//     candidates/query is already close to n/32,768 (~3) regardless of its
//     own floor being 2 rather than 1 -- Coarse stays a real memory/latency
//     middle ground here. VeryCoarse's crowding_factor is already severe
//     (measured ~49x in the run this comment was written against) --
//     already closer to Scan's own cost/precision trade than to Coarse's.
//   - EXTREME cardinality (fully unique per object, e.g. a serial number or
//     computed key): Coarse's absolute candidates/query barely moves from
//     the HIGH scenario (~3 either way -- the n/bucket_count convergence
//     regardless of K, once saturated, described above) even though its
//     crowding_factor reads higher (Exact's own floor dropped from ~2 to 1,
//     shrinking the denominator, not because Coarse got meaningfully more
//     crowded). VeryCoarse's crowding_factor lands close to MEDIUM's
//     (~98x vs ~5x there, but both converge on the SAME absolute
//     candidates/query, ~98, once saturated) -- the clearest evidence in
//     this program's own output that "n/bucket_count once saturated,
//     independent of K" is the real relationship, not "proportional to
//     K/buckets".
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

// One field per LookupType, all four holding the SAME value per object --
// so find_by_field<&TuningThing::X>(v) returns the identical real answer
// set regardless of which X you ask, and any difference in call time is
// attributable entirely to routing precision, not to the data itself.
class TuningThing final : public Object<TuningThing> {
public:
    std::string exact_field;
    std::string coarse_field;
    std::string very_coarse_field;
    std::string scan_field;

    template <class Self>
    static void define_fields(Self& s, const LookupFieldReader& v) {
        v.field<&TuningThing::exact_field>(s.exact_field, LookupType::Exact, "exact_field");
        v.field<&TuningThing::coarse_field>(s.coarse_field, LookupType::Coarse, "coarse_field");
        v.field<&TuningThing::very_coarse_field>(s.very_coarse_field, LookupType::VeryCoarse,
                                                 "very_coarse_field");
        v.field<&TuningThing::scan_field>(s.scan_field, LookupType::Scan, "scan_field");
    }
};

// n objects, k distinct values (n/k objects per value on average) -- the
// SAME value assigned to all four fields on a given object, so every
// field's index sees the identical value distribution.
void build(Model& m, int n, int k) {
    m.set_max_undo_list_size(0);
    Transaction txn = m.begin();
    for (int i = 0; i < n; ++i) {
        auto o = std::make_unique<TuningThing>();
        const std::string value = "v" + std::to_string(i % k);
        o->exact_field = value;
        o->coarse_field = value;
        o->very_coarse_field = value;
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
//   Outer leaf, Coarse/VeryCoarse field (by_cached_field_merged_,
//                pmap::IdentityHash64 IS declared perfect): RcBase(16B) +
//                PersistentSet handle(24B) + NoChain(0B) = 40B.
//   Inner leaf (every field alike -- PersistentSet<Id,IdHash>, IdHash is
//                perfect): RcBase(16B) + Id(8B) + NoChain(0B) = 24B.
constexpr double kNodeFixedBytes = 32.0;
constexpr double kSlotBytes = 8.0;
constexpr double kExactOuterLeafBytes = 48.0;
constexpr double kMergedOuterLeafBytes = 40.0;
constexpr double kInnerLeafBytes = 24.0;

// The two Coarse/VeryCoarse routing-precision ceilings (model.cpp's
// max_shift_for: 15 bits, 10 bits) expressed as bucket counts.
constexpr double kCoarseBucketCeiling = 32768.0;
constexpr double kVeryCoarseBucketCeiling = 1024.0;

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
// Coarse/VeryCoarse version of the SAME field would look like, without
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
// A single constant is a genuine ceiling on this model's accuracy, not just
// an under-tuned one -- confirmed by directly inspecting real node_count
// against predicted inputs while calibrating this value. very_coarse_field
// at MEDIUM (1,020 merged buckets, ~98 objects/bucket on average) and at
// HIGH/EXTREME (1,024 buckets, ~98 objects/bucket, an almost identical
// input to this formula) measure real node counts 57% apart (21,073 vs.
// ~33,000) -- because "inner nodes = (n - outer_leaves) / 31" only ever
// sees the pooled totals, but real node count depends on HOW population
// splits across the `outer_leaves` separate, independently-filling inner
// tries, not just their sum: MEDIUM's fewer, more crowded buckets (K/buckets
// ~4.9 average) fill less evenly than HIGH/EXTREME's more numerous, more
// saturated ones. No single scalar can correct for a per-scenario
// distribution shape it never sees. Given that ceiling, this constant is
// fit as a compromise across all four scenarios (minimizing the worst gap,
// not zeroing any one of them): predictions land consistently 80-90% of the
// real measured bytes, never over -- a stable, conservative band, not
// something further tuning will close.
constexpr double kNodeCountCorrectionFactor = 3.2;

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
    const FieldStats very_coarse = field_stats(m, field_tag<&TuningThing::very_coarse_field>(), true, n,
                                               kVeryCoarseBucketCeiling);
    const FieldStats scan = field_stats(m, field_tag<&TuningThing::scan_field>(), false, n, 0.0);

    const double exact_us = avg_find_time_us<&TuningThing::exact_field>(s, k);
    const double coarse_us = avg_find_time_us<&TuningThing::coarse_field>(s, k);
    const double very_coarse_us = avg_find_time_us<&TuningThing::very_coarse_field>(s, k);
    const double scan_us = avg_find_time_us<&TuningThing::scan_field>(s, k);

    // Memory vs speed, side by side -- est. bytes is the whole index for
    // this ONE field (structural node/slot memory + every outer and inner
    // leaf, per the constants above); read cost is the same best-of-5
    // average find_by_field measurement as before. Reading down a column
    // shows the trade in isolation (bytes: Exact >= Coarse >= VeryCoarse,
    // per LookupType's own doc comment); reading across a row shows what
    // that trade actually costs or buys for THIS field's real cardinality.
    std::printf("  %-16s %10s  %12s  %13s\n", "field", "leaves", "est. bytes", "read (best-of-5)");
    print_row("exact_field", exact, exact_us, 0.0);
    // Forward prediction, FROM exact_field's own always-precise cardinality
    // -- see predict_merged's own comment. Printed against the SAME
    // scenario's actually-measured coarse_field/very_coarse_field rows
    // right below, so the prediction's accuracy is visible directly, not
    // just asserted.
    if (exact.outer_leaves > 0.0) {
        const auto pred_coarse =
            predict_merged(exact.outer_leaves, static_cast<double>(n), kCoarseBucketCeiling);
        const auto pred_very_coarse =
            predict_merged(exact.outer_leaves, static_cast<double>(n), kVeryCoarseBucketCeiling);
        std::printf(
            "      if Coarse instead (predicted from exact_field's own K=%.0f): ~%.1f "
            "candidates/query, ~%.0f KB\n",
            exact.outer_leaves, pred_coarse.predicted_candidates_per_query,
            pred_coarse.predicted_est_bytes / 1024.0);
        std::printf(
            "      if VeryCoarse instead (predicted): ~%.1f candidates/query, ~%.0f KB\n",
            pred_very_coarse.predicted_candidates_per_query, pred_very_coarse.predicted_est_bytes / 1024.0);
    }
    print_row("coarse_field", coarse, coarse_us, exact.avg_candidates_per_query);
    print_row("very_coarse_field", very_coarse, very_coarse_us, exact.avg_candidates_per_query);
    print_row("scan_field", scan, scan_us, 0.0, "<- O(N objects)/call, no index memory at all");
}

}  // namespace

int main(int argc, char** argv) {
    const int n = argc > 1 ? std::atoi(argv[1]) : 100'000;
    // Straddling the two bucket counts (~1,024 for VeryCoarse, ~32,768 for
    // Coarse, see model.cpp's max_shift_for) on purpose: LOW sits under
    // both (nothing merges anywhere), MEDIUM sits between them (VeryCoarse
    // merges for real, Coarse barely does), HIGH sits above both (both
    // merge, but objects/value is still > 1), EXTREME is fully unique per
    // object (objects/value == 1) -- the two saturated scenarios (HIGH,
    // EXTREME) exist side by side specifically to show that once Coarse/
    // VeryCoarse are both saturated, crowding_factor keeps climbing with n
    // even though K alone doesn't obviously predict how much (see the
    // top-of-file comment's correction of its own earlier K/buckets claim).
    const int low_k = argc > 2 ? std::atoi(argv[2]) : 100;
    const int medium_k = argc > 3 ? std::atoi(argv[3]) : 5'000;
    const int high_k = argc > 4 ? std::atoi(argv[4]) : 50'000;
    const int extreme_k = argc > 5 ? std::atoi(argv[5]) : n;  // unique per object

    run_scenario("LOW cardinality (e.g. a status/category field)", n, low_k);
    run_scenario("MEDIUM cardinality (between the two bucket counts)", n, medium_k);
    run_scenario("HIGH cardinality (above both bucket counts, still several objects/value)", n, high_k);
    run_scenario("EXTREME cardinality (e.g. a unique serial number / computed key)", n, extreme_k);

    std::printf(
        "\nRule of thumb -- two numbers need only ONE Model::slot_stats_diagnostics() call plus\n"
        "your own object count (n), no benchmark and no Exact-tagged twin field required:\n"
        "  outer_leaves = field_leaf_count - n\n"
        "  candidates_per_query = n / outer_leaves   <- what a real find_by_field call resolves;\n"
        "                                                judge this alone against your own latency budget\n"
        "  saturation = outer_leaves / bucket_ceiling (32,768 Coarse, 1,024 VeryCoarse)\n"
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
