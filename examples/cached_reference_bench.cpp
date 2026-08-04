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
//
// The memory section forks a child per measurement and reads that child's
// OWN peak RSS from /proc/self/status (Linux-only -- see the __linux__ guard
// below). fork() WITHOUT exec() is enough for isolation here: each child's
// heap is copy-on-write from the parent, so mallocs in one measurement can't
// leave arena pages behind to pollute the next one. getrusage(RUSAGE_CHILDREN)
// was deliberately NOT used instead -- it reports the max across ALL
// terminated children of this process, not the most recent one, which would
// silently corrupt the second of two sequential measurements.

#include "model/model.h"

#include <cassert>
#include <chrono>
#include <cstdio>
#include <memory>
#include <string>
#include <vector>

#if defined(__linux__)
#include <cstdlib>
#include <sys/wait.h>
#include <unistd.h>
#define CACHED_REF_BENCH_HAS_MEMORY_SECTION 1
#else
#define CACHED_REF_BENCH_HAS_MEMORY_SECTION 0
#endif

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
// WRITE benchmark specifically: reconcile_out_refs (the writer-private
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
    for (auto& r : fx.buckets) r = bres.to_real(r);

    for (int i = 0; i < n_per_type; i += 1000) {
        Transaction txn = fx.m.begin();
        for (int j = 0; j < 1000 && i + j < n_per_type; ++j) {
            const Ref<Bucket> b = fx.buckets[(i + j) % fx.buckets.size()];
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

// Single-type counterpart of seed(), for the memory benchmark: it needs each
// type's population built ALONE, in its own process (see below), so the
// measured RSS delta is attributable to that one type's index, not diluted
// by (or double-counting) the other type's objects sharing the run.
template <class T>
void seed_one_type(Fixture& fx, int n_per_type) {
    const int n_buckets = bucket_count_for(n_per_type);
    Transaction seed_txn = fx.m.begin();
    for (int i = 0; i < n_buckets; ++i) {
        auto b = std::make_unique<Bucket>();
        b->name = "B" + std::to_string(i);
        fx.buckets.push_back(seed_txn.create(std::move(b)));
    }
    const CommitResult bres = fx.m.try_commit(seed_txn);
    assert(bres.status == CommitStatus::Committed);
    for (auto& r : fx.buckets) r = bres.to_real(r);

    for (int i = 0; i < n_per_type; i += 1000) {
        Transaction txn = fx.m.begin();
        for (int j = 0; j < 1000 && i + j < n_per_type; ++j) {
            const Ref<Bucket> b = fx.buckets[(i + j) % fx.buckets.size()];
            auto o = std::make_unique<T>();
            o->bucket = b;
            o->payload = i + j;
            txn.create(std::move(o));
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
    // reconcile_out_refs/reconcile_cached_references both take.
    int rep = 0;
    const double total_us = time_us(reps, [&] {
        Transaction txn = m.begin();
        for (int k = 0; k < batch; ++k) {
            const Ref<Bucket> nb = buckets[(k + rep) % buckets.size()];
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

// ---------------------------------------------------------------------------
// Memory: what Root::by_cached_reference actually costs at rest.
// ---------------------------------------------------------------------------

#if CACHED_REF_BENCH_HAS_MEMORY_SECTION

// This process's own peak resident set size (VmHWM), in KB. High-water mark,
// not current usage -- exactly what we want here: the worst the allocator
// made the OS commit while building the population, not whatever happens to
// still be resident after (glibc may or may not have returned pages to the
// OS by then).
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

// Runs `work` in a forked child and returns THAT CHILD's own peak RSS, read
// from its own /proc/self/status and sent back through a pipe. Each call
// forks fresh off the (single-threaded, at this point in main()) parent, so
// every measurement gets an isolated heap -- no allocator arena carried over
// from a previous measurement can inflate or deflate this one.
template <class F>
long measure_child_peak_kb(F&& work) {
    int fds[2];
    if (pipe(fds) != 0) return -1;
    const pid_t pid = fork();
    if (pid == 0) {
        close(fds[0]);
        work();
        const long kb = peak_rss_kb();
        char buf[32];
        const int n = std::snprintf(buf, sizeof buf, "%ld\n", kb);
        // Fire-and-forget: this process exits immediately after, and there is
        // nothing useful to do about a partial write to our own pipe here.
        if (write(fds[1], buf, static_cast<std::size_t>(n)) < 0) { /* nothing to do */
        }
        close(fds[1]);
        // NOT exit()/return: skip C++ static destructors and atexit hooks
        // racing the parent's already-buffered stdio. Side effect under
        // ASan/LSan: LeakSanitizer's own finalizer is an atexit hook too, so
        // it never runs for this child -- a leak introduced inside `work()`
        // would go unreported HERE. It would still be caught by every other
        // Model this program constructs directly (in bench(), outside any
        // fork), which is exactly where ASan runs on this codebase's actual
        // model code, not on this measurement harness.
        _exit(0);
    }
    close(fds[1]);
    char buf[32] = {};
    const ssize_t r = read(fds[0], buf, sizeof(buf) - 1);
    close(fds[0]);
    int status = 0;
    waitpid(pid, &status, 0);
    return r > 0 ? std::atol(buf) : -1;
}

struct MemResult {
    int n_per_type;
    long uncached_extra_kb;  // over an empty-process baseline
    long cached_extra_kb;    // ditto
};

MemResult bench_memory(int n_per_type, long baseline_kb) {
    const long uncached_kb = measure_child_peak_kb([n_per_type] {
        Fixture fx;
        seed_one_type<UncachedItem>(fx, n_per_type);
    });
    const long cached_kb = measure_child_peak_kb([n_per_type] {
        Fixture fx;
        seed_one_type<CachedItem>(fx, n_per_type);
    });
    return {n_per_type, uncached_kb - baseline_kb, cached_kb - baseline_kb};
}

#endif  // CACHED_REF_BENCH_HAS_MEMORY_SECTION

int main() {
    // "Big enough to show differences": up to 500,000 items PER TYPE --
    // 1,000,000+ objects total, at the top of this project's documented
    // target scale (see CLAUDE.md). Bucket count scales with n_per_type
    // (bucket_count_for) so the match count -- and the writer-private
    // referrers_ index's own per-bucket cost -- stay comparable across the
    // sweep; what's left to grow is what this benchmark is actually about.
    const std::vector<int> sizes = {50000, 500000};

#if CACHED_REF_BENCH_HAS_MEMORY_SECTION
    // Measured FIRST, before this process itself does any heavy allocation:
    // fork() gives each child a copy-on-write snapshot of the PARENT's
    // current memory, arena and all. Fork late (after the read/write
    // benchmarks below have already built and torn down several 500k-object
    // Models) and every child inherits that huge, already-resident-but-freed
    // glibc arena -- its own mallocs then get satisfied from memory that was
    // already resident before it ever touched anything, and the "extra RSS"
    // delta this section exists to measure comes out as noise near zero.
    // Forking while the parent is still small avoids that entirely.
    const long baseline_kb = measure_child_peak_kb([] { /* nothing: just process startup cost */ });
    std::vector<MemResult> mem_results;
    for (int n : sizes) mem_results.push_back(bench_memory(n, baseline_kb));
#endif

    std::vector<Result> results;
    for (int n : sizes) results.push_back(bench(n));

    std::printf(
        "=== READS: \"every item pointing at this Bucket?\" ===\n"
        "for_each_referrers's O(#items) scan vs. find_cached_referrers' O(log n + matches).\n\n");
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
        "index, paid on every commit that touches it, in exchange for the read-side speedup.\n");

#if CACHED_REF_BENCH_HAS_MEMORY_SECTION
    std::printf(
        "\n=== MEMORY: resident set size for a population of ONE type, alone ===\n"
        "Each figure is a FRESH forked child's own peak RSS (VmHWM), minus an empty-process\n"
        "baseline -- isolated per measurement, so no allocator arena from an earlier run can\n"
        "pollute this one. Cached pays for Root::by_cached_reference: one more persistent-map\n"
        "index entry held PER OBJECT for the field that opted in.\n\n");
    std::printf("%12s  %16s  %16s  %12s  %14s\n", "items/type", "uncached (MB)", "cached (MB)",
               "overhead", "bytes/item");
    for (const MemResult& r : mem_results) {
        const double u_mb = static_cast<double>(r.uncached_extra_kb) / 1024.0;
        const double c_mb = static_cast<double>(r.cached_extra_kb) / 1024.0;
        const double bytes_per_item =
            static_cast<double>(r.cached_extra_kb - r.uncached_extra_kb) * 1024.0 / r.n_per_type;
        std::printf("%12d  %16.1f  %16.1f  %11.2fx  %14.1f\n", r.n_per_type, u_mb, c_mb,
                    c_mb / u_mb, bytes_per_item);
    }
    std::printf(
        "\nExpect \"bytes/item\" to stay roughly constant across items/type -- that per-object\n"
        "figure IS the fixed cost of one Root::by_cached_reference entry (an outer-map lookup\n"
        "plus one leaf in the target's inner bucket), not something that grows with population.\n"
        "It's the standing rent for the read-side speedup above: pay it for every object of a\n"
        "type that declares the field cached, whether or not that particular object's bucket\n"
        "ever gets queried.\n");
#else
    std::printf(
        "\n=== MEMORY ===\nSkipped: this section reads /proc/self/status and uses fork(),\n"
        "both Linux-only. Build and run on Linux to see the memory comparison.\n");
#endif

    std::printf(
        "\nThis is the whole tradeoff in one run: index only the reverse lookups you actually\n"
        "run often (define_cached_references) -- and leave the rest on the always-correct,\n"
        "zero-write-cost, zero-standing-memory scan (plain find_referrers).\n");
    return 0;
}
