// Cached vs. uncached reference lookup: read speed, and the write-side price
// that pays for it.
//
// Two types, structurally identical except for ONE declaration:
//   UncachedItem::bucket -- listed in define_references() only. "Who points
//     at this Bucket?" (Snapshot::find_referrers) is an O(#items) linear
//     scan: nothing indexes it.
//   CachedItem::bucket -- ALSO listed in define_cached_references().
//     find_cached_referrers answers the same question in O(log n +
//     #matches), backed by Root::by_cached_reference -- see DESIGN.md's
//     "Lookup families" section.
//
// That isolates the cache to exactly one axis: everything else (field
// shapes, cascade rules, transaction sizes) is identical between the two
// types, so any timing difference is attributable to the one declaration.
//
// Run under asan/tsan too, same as every other example -- this is ordinary
// application code, not a privileged fast path, so it gets no exemption
// from the correctness bar.

#include "model/model.h"

#include <cassert>
#include <chrono>
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

class UncachedItem final : public Object<UncachedItem> {
public:
    Ref<Bucket> bucket;
    std::int64_t payload = 0;

    template <class Self, class V>
    static void define_references(Self& s, V&& v) {
        v(field_tag<&UncachedItem::bucket>(), "bucket", s.bucket);
    }
};

class CachedItem final : public Object<CachedItem> {
public:
    Ref<Bucket> bucket;
    std::int64_t payload = 0;

    template <class Self, class V>
    static void define_references(Self& s, V&& v) {
        v(field_tag<&CachedItem::bucket>(), "bucket", s.bucket);
    }

    // The one line UncachedItem doesn't have.
    template <class Self>
    static void define_cached_references(Self& s, const RefIndexReader& v) {
        (void)s;
        v.index<&CachedItem::bucket>();
    }
};

// Items per bucket, held roughly CONSTANT across the sweep by scaling bucket
// count with n_per_type (see bucket_count_for below). This matters for the
// WRITE benchmark specifically: reconcile_referrer_edges (the writer-private
// referrers_ index, maintained for EVERY reference field regardless of
// caching -- see CLAUDE.md's documented scope boundary) scans its target's
// referrer list linearly, so if bucket count stayed fixed while n_per_type
// grew, that PRE-EXISTING cost would grow with population and swamp the
// one thing this benchmark is trying to isolate: the FIXED per-write tax of
// the NEW cached-reference index. Scaling buckets with n cancels that out,
// so what's left is (close to) just the cached-reference index's own cost.
constexpr int kItemsPerBucket = 250;
int bucket_count_for(int n_per_type) { return std::max(50, n_per_type / kItemsPerBucket); }

template <class F>
double time_us(int reps, F&& f) {
    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < reps; ++i) f();
    auto t1 = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::micro>(t1 - t0).count() / reps;
}

// Populates BOTH item types identically -- same buckets, same round-robin
// distribution -- so a read query for one bucket returns the same match
// count on either type, and a write batch touches the same shape of data.
// This is the only fair way to isolate "cached vs. not": same population,
// same access pattern, one declaration different.
struct Fixture {
    Model m;
    std::vector<Ref<Bucket>> buckets;
};

void seed(Fixture& fx, int n_per_type) {
    const int n_buckets = bucket_count_for(n_per_type);
    Transaction seed_txn = fx.m.begin();
    for (int i = 0; i < n_buckets; ++i) {
        auto b = std::make_unique<Bucket>();
        b->name = "B" + std::to_string(i);
        fx.buckets.push_back(seed_txn.create(std::move(b)));
    }
    const CommitResult bres = fx.m.try_commit(seed_txn);
    assert(bres.status == CommitStatus::Committed);
    for (auto& r : fx.buckets) r = bres.resolve(r);

    for (int i = 0; i < n_per_type; i += 1000) {
        Transaction txn = fx.m.begin();
        for (int j = 0; j < 1000 && i + j < n_per_type; ++j) {
            const Ref<Bucket> b = fx.buckets[static_cast<std::size_t>((i + j) % fx.buckets.size())];
            auto u = std::make_unique<UncachedItem>();
            u->bucket = b;
            u->payload = i + j;
            txn.create(std::move(u));
            auto c = std::make_unique<CachedItem>();
            c->bucket = b;
            c->payload = i + j;
            txn.create(std::move(c));
        }
        const CommitResult res = fx.m.try_commit(txn);
        assert(res.status == CommitStatus::Committed);
    }
}

// ---------------------------------------------------------------------------
// One seeded Fixture answers BOTH the read and the write question -- half
// the object-creation cost of seeding twice per size, for the same result.
// ---------------------------------------------------------------------------

struct Result {
    int n_per_type;
    std::size_t matches;
    double scan_us;      // find_referrers: O(#items) linear scan
    double indexed_us;   // find_cached_referrers: O(log n + matches)
    double uncached_write_us_per_op;  // reassigning UncachedItem::bucket
    double cached_write_us_per_op;    // reassigning CachedItem::bucket (also touches the index)
};

template <class T>
double bench_reassign(Model& m, const std::vector<Ref<T>>& items, const std::vector<Ref<Bucket>>& buckets,
                      int batch, int reps) {
    // A DIFFERENT target bucket every rep (k+rep, not just k), so every
    // reassignment is a real value change -- old != new -- and actually
    // exercises reconciliation instead of hitting the "unchanged, skip" path
    // reconcile_referrer_edges/reconcile_cached_references both take.
    int rep = 0;
    const double total_us = time_us(reps, [&] {
        Transaction txn = m.begin();
        for (int k = 0; k < batch; ++k) {
            const Ref<Bucket> nb = buckets[static_cast<std::size_t>((k + rep) % buckets.size())];
            if (T* p = txn.update(items[static_cast<std::size_t>(k)])) p->bucket = nb;
        }
        const CommitResult res = m.try_commit(txn);
        assert(res.status == CommitStatus::Committed);
        ++rep;
    });
    return total_us / batch;
}

Result bench(int n_per_type) {
    Fixture fx;
    seed(fx, n_per_type);
    Snapshot s = fx.m.snapshot();
    const Ref<Bucket> target = fx.buckets[0];

    // ---- reads: measured first, against the freshly seeded state --------
    const auto slow = s.find_referrers<&UncachedItem::bucket>(target);
    const auto fast = s.find_cached_referrers<&CachedItem::bucket>(target);
    assert(slow.size() == fast.size());  // correctness rides along, not just speed
    assert(!slow.empty());

    const double scan_us = time_us(12, [&] {
        auto v = s.find_referrers<&UncachedItem::bucket>(target);
        assert(!v.empty());
    });
    const double indexed_us = time_us(3000, [&] {
        auto v = s.find_cached_referrers<&CachedItem::bucket>(target);
        assert(!v.empty());
    });

    // ---- writes: reassign 500 already-seeded items' bucket field ---------
    // `s` (above) is still held, so nothing retired below is freed yet --
    // deliberately: destroying s here would let the reaper start recycling
    // mid-measurement, and this benchmark cares about try_commit() latency,
    // not reclamation.
    constexpr int kBatch = 500, kReps = 20;
    std::vector<Ref<UncachedItem>> uitems;
    std::vector<Ref<CachedItem>> citems;
    s.for_each<UncachedItem>([&](const UncachedItem& o) {
        if (static_cast<int>(uitems.size()) < kBatch) uitems.push_back(Ref<UncachedItem>(o.id));
    });
    s.for_each<CachedItem>([&](const CachedItem& o) {
        if (static_cast<int>(citems.size()) < kBatch) citems.push_back(Ref<CachedItem>(o.id));
    });
    assert(uitems.size() == kBatch && citems.size() == kBatch);

    const double u_us = bench_reassign(fx.m, uitems, fx.buckets, kBatch, kReps);
    const double c_us = bench_reassign(fx.m, citems, fx.buckets, kBatch, kReps);

    return {n_per_type, slow.size(), scan_us, indexed_us, u_us, c_us};
}

int main() {
    // "Big enough to show differences": up to 500,000 items PER TYPE --
    // 1,000,000+ objects total, at the top of this project's documented
    // target scale (see CLAUDE.md). Bucket count scales with n_per_type
    // (bucket_count_for) so the match count -- and the writer-private
    // referrers_ index's own per-bucket cost -- stay comparable across the
    // sweep; what's left to grow is what this benchmark is actually about.
    const std::vector<int> sizes = {50000, 500000};
    std::vector<Result> results;
    for (int n : sizes) results.push_back(bench(n));

    std::printf(
        "=== READS: \"every item pointing at this Bucket?\" ===\n"
        "for_each_referrer's O(#items) scan vs. find_cached_referrers' O(log n + matches).\n\n");
    std::printf("%12s  %10s  %14s  %14s  %10s\n", "items/type", "matches", "scan (us)", "indexed (us)",
               "speedup");
    for (const Result& r : results) {
        std::printf("%12d  %10zu  %14.1f  %14.3f  %9.0fx\n", r.n_per_type, r.matches, r.scan_us,
                    r.indexed_us, r.scan_us / r.indexed_us);
    }
    std::printf(
        "\nThe scan pays for the WHOLE population on every query, no matter how few items "
        "actually match -- its cost grows with items/type. The indexed lookup's grows far "
        "slower (O(log n) in the index depth, not O(n)): that widening gap is the entire "
        "reason Root::by_cached_reference exists.\n");

    std::printf(
        "\n=== WRITES: reassigning 500 already-existing items' reference field ===\n"
        "Uncached touches only referrers_ (writer-private, unconditionally maintained for\n"
        "EVERY reference field, cached or not). Cached ALSO touches Root::by_cached_reference\n"
        "-- one more persistent-map index entry moved per changed reference.\n\n");
    std::printf("%12s  %16s  %16s  %10s\n", "items/type", "uncached (us/op)", "cached (us/op)",
               "overhead");
    for (const Result& r : results) {
        std::printf("%12d  %16.2f  %16.2f  %9.2fx\n", r.n_per_type, r.uncached_write_us_per_op,
                    r.cached_write_us_per_op,
                    r.cached_write_us_per_op / r.uncached_write_us_per_op);
    }
    std::printf(
        "\nNeither column is perfectly flat: try_commit() copies the whole spine vector per\n"
        "commit (see DESIGN.md's Structure diagram -- O(#chunks), i.e. O(model size / 256),\n"
        "shared_ptr bumps, not a deep copy), so BOTH grow slowly with items/type regardless of\n"
        "caching. What matters is that neither grows anywhere near as fast as the read-side\n"
        "SCAN above -- the gap between the two write columns stays a small, roughly-constant\n"
        "multiple, not a widening one: a fixed per-write tax for the field that opted into the\n"
        "index, paid on every commit that touches it, in exchange for the read-side speedup.\n"
        "\nThis is the whole tradeoff in one run: index only the reverse lookups you actually\n"
        "run often (define_cached_references) -- and leave the rest on the always-correct,\n"
        "zero-write-cost scan (plain find_referrers).\n");
    return 0;
}
