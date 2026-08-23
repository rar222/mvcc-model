// Measures whether KeyLookupType::Coarse (define_keys()/by_key_) is worth
// using over Exact, the key-field counterpart to
// examples/lookup_type_tuning_demo.cpp's field measurement -- but not the
// same question asked twice. by_key_'s Coarse was NEVER given a
// bucket-merge tier (see project memory on
// project_cached_field_bucket_merge_proposal): a key field enforces
// uniqueness (Model::validate_field_key_uniqueness, model.cpp), so a
// coarsened key's collision chain still holds one leaf PER DISTINCT KEY,
// reached by walking Leaf::next doing a resolve (Id -> object -> field
// string) + compare per link (persistent_map.h's leaf_get) -- there is no
// shared Set to short-circuit into, unlike by_cached_field_merged_'s
// bucket. Two consequences that make this a genuinely different
// measurement, not a re-run:
//   1. K == N, always, by construction (every live object's key is
//      distinct) -- there is no separate cardinality knob to sweep the way
//      lookup_type_tuning_demo.cpp swept K at fixed N. The only knob here
//      is N itself.
//   2. Coarsening buys ONLY fewer routing Node levels (each key still costs
//      its own 32B leaf -- pmap::DedupMap<string,Id,...>'s leaf, RcBase(16B)
//      + Id(8B) + ChainLink(8B), "Leaf sizes landed exactly as projected"
//      per project memory) -- there is no leaf-count reduction to earn back
//      the chain-walk cost the way bucket-merge did for by_cached_field_.
//
// Measured finding (KeyLookupType::VeryCoarse existed alongside Coarse
// until it was removed): unlike fields, where VeryCoarse's extra savings
// over Coarse were merely small, for keys they were close to nothing --
// once both routing trees saturate their own bucket ceiling, the FIXED
// per-key leaf cost (32B regardless of routing precision) dominates total
// bytes, so Coarse and VeryCoarse converged on nearly identical memory
// while VeryCoarse's chain length (and thus its read cost) kept growing
// far worse. Coarse itself remains a real, defensible win over Exact: see
// the output below, and CLAUDE.md's "things that look like improvements
// but are not" for the removal rationale.
//
// Run:  ./build/default/key_lookup_type_tuning_demo [n1] [n2] [n3] [n4]

#include "model/model.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <memory>
#include <string>
#include <typeinfo>

using namespace model;

namespace {

// One key field per KeyLookupType, both holding the SAME unique value per
// object -- so find_by_key<&KeyTuningThing::X>(v) returns the identical
// object regardless of which X you ask, and any timing difference is
// attributable entirely to routing precision.
class KeyTuningThing final : public Object<KeyTuningThing> {
public:
    std::string key_exact;
    std::string key_coarse;

    template <class Self>
    static void define_keys(Self& s, const FieldKeyReader& v) {
        v.key<&KeyTuningThing::key_exact>(s.key_exact, KeyLookupType::Exact, "key_exact");
        v.key<&KeyTuningThing::key_coarse>(s.key_coarse, KeyLookupType::Coarse, "key_coarse");
    }
};

void build(Model& m, int n) {
    m.set_max_undo_list_size(0);
    Transaction txn = m.begin();
    for (int i = 0; i < n; ++i) {
        auto o = std::make_unique<KeyTuningThing>();
        const std::string value = "k" + std::to_string(i);
        o->key_exact = value;
        o->key_coarse = value;
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

template <auto Field>
double avg_find_time_us(const Snapshot& s, int n) {
    // A bounded, evenly-spread sample of the n distinct keys -- same
    // rationale as lookup_type_tuning_demo.cpp's avg_find_time_us: a sample
    // that happened to land only in one part of the routing space could
    // over/under-represent the real average chain-walk cost.
    const int sample = std::min(n, 200);
    const int stride = std::max(1, n / sample);
    return best_of(5, [&] {
        return time_us([&] {
                   for (int i = 0; i < sample; ++i) (void)s.find_by_key<Field>("k" + std::to_string((i * stride) % n));
               }) /
               sample;
    });
}

// Leaf = pmap::DedupMap<std::string, Id, StringHash, DedupIdKeyOf>'s own
// entry: RcBase(16B) + Id(8B) + ChainLink(8B, StringHash is NOT declared
// perfect, so every leaf is chain-capable) = 32B. Matches the measured
// "by_key_ 64B->32B" result from the string-key-dedup work (project
// memory) -- this is the SAME 32B for every KeyLookupType, since coarsening
// never changes the leaf shape, only how many Node levels route to it.
constexpr double kNodeFixedBytes = 32.0;
constexpr double kSlotBytes = 8.0;
constexpr double kKeyLeafBytes = 32.0;

struct KeyFieldStats {
    std::size_t node_count = 0;
    std::size_t leaf_count = 0;
    std::size_t chain_slot_count = 0;
    std::size_t max_chain_len = 0;
    double avg_chain_len = 0.0;
    double est_bytes = 0.0;
};

KeyFieldStats key_field_stats(const Model& m, const void* field) {
    KeyFieldStats fs;
    const auto diag = m.slot_stats_diagnostics();
    const FieldLookupKey key{&typeid(KeyTuningThing), field};
    const auto it = diag.by_key_detail.find(key);
    if (it == diag.by_key_detail.end()) return fs;
    fs.node_count = it->second.node_count;
    fs.leaf_count = it->second.leaf_count;
    fs.chain_slot_count = it->second.chain_slot_count;
    fs.max_chain_len = it->second.max_chain_len;
    fs.avg_chain_len =
        fs.chain_slot_count > 0 ? static_cast<double>(fs.leaf_count) / static_cast<double>(fs.chain_slot_count)
                                 : 0.0;
    fs.est_bytes = static_cast<double>(fs.node_count) * kNodeFixedBytes +
                   static_cast<double>(it->second.total_capacity) * kSlotBytes +
                   static_cast<double>(fs.leaf_count) * kKeyLeafBytes;
    return fs;
}

void print_row(const char* name, const KeyFieldStats& fs, double read_us) {
    std::printf("  %-16s %9zu nodes  %9.0f KB  avg chain %5.2f  max chain %4zu  %10.3f us/call\n", name,
               fs.node_count, fs.est_bytes / 1024.0, fs.avg_chain_len, fs.max_chain_len, read_us);
}

void run_scenario(int n) {
    std::printf("\n=== N=%d objects (K == N always -- every key is unique) ===\n", n);
    Model m;
    build(m, n);
    Snapshot s = m.snapshot();

    const auto exact = key_field_stats(m, field_tag<&KeyTuningThing::key_exact>());
    const auto coarse = key_field_stats(m, field_tag<&KeyTuningThing::key_coarse>());

    const double exact_us = avg_find_time_us<&KeyTuningThing::key_exact>(s, n);
    const double coarse_us = avg_find_time_us<&KeyTuningThing::key_coarse>(s, n);

    std::printf("  %-16s %14s  %12s  %14s  %13s  %s\n", "field", "", "est. bytes", "", "", "find_by_key");
    print_row("key_exact", exact, exact_us);
    print_row("key_coarse", coarse, coarse_us);

    const double coarse_saved_kb = exact.est_bytes / 1024.0 - coarse.est_bytes / 1024.0;
    std::printf("  vs. Exact: Coarse saves %.0f KB (%.1fx read time)\n", coarse_saved_kb,
               exact_us > 0.0 ? coarse_us / exact_us : 0.0);
}

}  // namespace

int main(int argc, char** argv) {
    const int n1 = argc > 1 ? std::atoi(argv[1]) : 2'000;
    const int n2 = argc > 2 ? std::atoi(argv[2]) : 20'000;
    const int n3 = argc > 3 ? std::atoi(argv[3]) : 100'000;
    const int n4 = argc > 4 ? std::atoi(argv[4]) : 500'000;

    // atoi() returns 0 for anything non-numeric (e.g. a blank/whitespace
    // arg); avg_find_time_us's stride computation (n / sample) divides by
    // an N-derived value, so a non-positive N turns into a SIGFPE instead
    // of a clear message without this check.
    for (const int v : {n1, n2, n3, n4}) {
        if (v <= 0) {
            std::fprintf(stderr,
                        "error: every N must be a positive integer (got %d) -- usage: %s [n1] [n2] "
                        "[n3] [n4]\n",
                        v, argv[0]);
            return 1;
        }
    }

    run_scenario(n1);
    run_scenario(n2);
    run_scenario(n3);
    run_scenario(n4);

    std::printf(
        "\nEvery key is unique (K == N), so there is no crowding tax the way a merged field pays "
        "one --\n"
        "the only cost is the collision-CHAIN walk (resolve + string-compare per link) that Coarse\n"
        "forces once trie depth bottoms out, and the only saving is fewer routing Node levels -- "
        "leaf count\n"
        "(and its 32B/leaf cost) never changes, unlike by_cached_field_'s bucket-merge. Read this "
        "as: does\n"
        "the KB saved at a given N justify that N's read-time multiplier, for how often you "
        "actually call\n"
        "find_by_key on this field?\n");
    return 0;
}
