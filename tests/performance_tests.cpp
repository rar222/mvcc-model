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
//   1. The claims themselves, swept across several sizes (1k/10k/100k),
//      each printed as a number AND checked against a generous bound.
//   2. Worst-case / degenerate data: the two specific costs CLAUDE.md's
//      own "Known scope boundaries" section already documents as
//      unbounded (a hub object with many referrers; an unbounded cascade)
//      -- demonstrated and bounded, not silently accepted as "fine."
//   3. Object PAYLOAD size as its own axis, independent of object COUNT:
//      write cost (clone/copy) should scale with an object's own byte
//      size; read cost (find_by_key) should not.
//
// Dependency-free, same as tests/tests.cpp (see CLAUDE.md: "Don't add
// gtest/Catch2") -- a small self-contained harness, adapted here to print
// a timing line per measurement instead of just OK/FAIL.
//
// Thresholds are DELIBERATELY loose (generous multiples, not tight
// bounds): this runs on unknown hardware, under default/asan/tsan alike,
// so the goal is catching a catastrophic (10x-1000x) regression, not
// flagging ordinary machine-to-machine variance.
//
// Run: ctest --preset default -R performance_tests --output-on-failure
//   or: ./build/default/performance_tests

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "demo/types.h"
#include "model/model.h"

#if defined(__linux__)
#include <cstdlib>
#include <sys/wait.h>
#include <unistd.h>
#define PERF_HAS_MEMORY_SECTION 1
#else
#define PERF_HAS_MEMORY_SECTION 0
#endif

using namespace model;
using namespace demo;

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
        if (!(cond)) {                                                   \
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

int scaled(int n) { return kSlowSanitizedBuild ? std::max(20, n / 25) : n; }

template <class F>
double time_ms(F&& f) {
    const auto t0 = std::chrono::steady_clock::now();
    f();
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
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
// scan) -- see demo/types.h. Same field, same data, same query: any timing
// difference is attributable entirely to the index, not to anything else.
PERF_TEST(find_by_key_is_flat_while_find_by_scan_field_grows_with_population) {
    const std::vector<int> sizes = {scaled(1000), scaled(10000), scaled(100000)};
    std::vector<double> key_us, scan_us;
    for (int n : sizes) {
        Model m;
        std::vector<Ref<Account>> accounts;
        seed_accounts(m, n, accounts);
        const std::string probe = "A" + std::to_string(n / 2);
        Snapshot s = m.snapshot();

        constexpr int kKeyReps = 5000;
        const double t_key = time_ms([&] {
            for (int i = 0; i < kKeyReps; ++i) (void)s.find_by_key<&Account::name>(probe);
        });
        key_us.push_back(t_key * 1000.0 / kKeyReps);

        // Fewer repetitions as n grows -- each individual scan call is
        // O(n), so this keeps total wall time roughly bounded while still
        // reporting a stable per-call average.
        const int scan_reps = std::max(5, 200000 / n);
        const double t_scan = time_ms([&] {
            for (int i = 0; i < scan_reps; ++i) (void)s.find_by_scan_field<&Account::name>(probe);
        });
        scan_us.push_back(t_scan * 1000.0 / scan_reps);

        std::printf("  n=%7d  find_by_key=%9.4f us/call   find_by_scan_field=%9.2f us/call\n", n,
                    key_us.back(), scan_us.back());
    }
    // O(1): a 100x population increase must not blow up the per-call cost.
    CHECK(key_us.back() < key_us.front() * 10.0 + 5.0);
    // The tradeoff the two families exist to document (see demo/types.h):
    // at the largest size, the unindexed scan must be markedly slower than
    // the indexed lookup on the SAME field.
    CHECK(scan_us.back() > key_us.back() * 5.0);
}

// Same comparison, one level up: Order::qty is in BOTH define_cached_fields
// (indexed, O(log n + matches)) and define_scan_fields (O(#orders) scan).
PERF_TEST(find_by_cached_field_stays_near_flat_while_scan_grows_on_the_same_field) {
    const std::vector<int> sizes = {scaled(1000), scaled(10000), scaled(100000)};
    std::vector<double> cached_us, scan_us;
    for (int n : sizes) {
        Model m;
        std::vector<Ref<Account>> accounts;
        seed_accounts(m, 4, accounts);
        std::vector<Ref<Order>> orders;
        // qty_mod scales with n so match count for `probe_qty` stays ~50
        // regardless of n -- isolates the index's O(log n) term from its
        // O(matches) term (see seed_orders' own comment).
        const int qty_mod = std::max(8, n / 50);
        seed_orders(m, accounts, n, orders, qty_mod);
        const std::int64_t probe_qty = 7;  // always < qty_mod (>= 8), always ~50 matches
        Snapshot s = m.snapshot();

        constexpr int kCachedReps = 2000;
        const double t_cached = time_ms([&] {
            for (int i = 0; i < kCachedReps; ++i) (void)s.find_by_cached_field<&Order::qty>(probe_qty);
        });
        cached_us.push_back(t_cached * 1000.0 / kCachedReps);

        const int scan_reps = std::max(5, 200000 / n);
        const double t_scan = time_ms([&] {
            for (int i = 0; i < scan_reps; ++i) (void)s.find_by_scan_field<&Order::qty>(probe_qty);
        });
        scan_us.push_back(t_scan * 1000.0 / scan_reps);

        std::printf(
            "  n=%7d  find_by_cached_field=%9.4f us/call   find_by_scan_field=%9.2f us/call\n", n,
            cached_us.back(), scan_us.back());
    }
    CHECK(cached_us.back() < cached_us.front() * 20.0 + 20.0);
    CHECK(scan_us.back() > cached_us.back() * 5.0);
}

// The reverse-lookup counterpart: Order::account is in define_references()
// (always -- required for cascade) AND define_cached_references() (opt-in
// index). find_referrers is the always-available O(#orders) scan;
// find_cached_referrers is the O(log n + matches) indexed alternative --
// same comparison examples/cached_reference_bench.cpp benchmarks in more
// depth, here as an asserted claim at smaller, CI-friendly sizes.
PERF_TEST(find_cached_referrers_beats_the_scan_and_the_gap_widens_with_population) {
    const std::vector<int> sizes = {scaled(1000), scaled(10000), scaled(100000)};
    std::vector<double> scan_us, indexed_us;
    for (int n : sizes) {
        Model m;
        // Account count scales with n (same reason as bucket_count_for in
        // examples/cached_reference_bench.cpp): match count for `target`
        // stays ~250 regardless of n, isolating the index's O(log n) term
        // from its O(matches) term.
        const int n_accounts = std::max(4, n / 250);
        std::vector<Ref<Account>> accounts;
        seed_accounts(m, n_accounts, accounts);
        std::vector<Ref<Order>> orders;
        seed_orders(m, accounts, n, orders);
        Snapshot s = m.snapshot();
        const Ref<Account> target = accounts[0];

        const int reps = std::max(5, 200000 / n);
        const double t_scan = time_ms([&] {
            for (int i = 0; i < reps; ++i) (void)s.find_referrers<&Order::account>(target);
        });
        scan_us.push_back(t_scan * 1000.0 / reps);

        constexpr int kIndexedReps = 2000;
        const double t_idx = time_ms([&] {
            for (int i = 0; i < kIndexedReps; ++i) (void)s.find_cached_referrers<&Order::account>(target);
        });
        indexed_us.push_back(t_idx * 1000.0 / kIndexedReps);

        std::printf("  n=%7d  scan=%10.2f us/call   indexed=%10.4f us/call   speedup=%8.0fx\n", n,
                    scan_us.back(), indexed_us.back(), scan_us.back() / indexed_us.back());
    }
    CHECK(indexed_us.back() < indexed_us.front() * 20.0 + 20.0);
    CHECK(scan_us.back() > indexed_us.back() * 5.0);
}

// ---------------------------------------------------------------------------
// Write-side complexity claims
// ---------------------------------------------------------------------------

// try_commit()'s publish step copies spine_ (O(#chunks) shared_ptr copies,
// not a deep copy -- see DESIGN.md's Structure diagram) and every
// per-type/per-field index (O(#types)/O(#indexed fields), independent of
// model size). Net effect: a single-field update's commit latency is NOT
// expected to stay flat as the model grows -- it grows with #chunks -- but
// it must stay BOUNDED (a small multiple, not proportional to a deep copy
// of the whole model). That's the claim under test here.
PERF_TEST(single_field_commit_latency_stays_bounded_as_total_model_size_grows) {
    const std::vector<int> sizes = {scaled(1000), scaled(10000), scaled(100000)};
    std::vector<double> avg_ms;
    for (int n : sizes) {
        Model m;
        std::vector<Ref<Account>> accounts;
        seed_accounts(m, n, accounts);

        constexpr int kReps = 20;
        double total = 0;
        for (int i = 0; i < kReps; ++i) {
            Transaction txn = m.begin();
            txn.update(accounts[static_cast<std::size_t>(i) % accounts.size()])->balance = i;
            total += time_ms([&] { (void)m.try_commit(txn); });
        }
        avg_ms.push_back(total / kReps);
        std::printf("  n=%7d  single-field commit = %8.4f ms/commit (avg of %d)\n", n, avg_ms.back(),
                    kReps);
    }
    // Generous: catches "someone turned the chunk copy into a deep
    // per-object copy" (which would blow well past this even at 100k), not
    // ordinary linear growth in chunk count.
    CHECK(avg_ms.back() < 200.0);
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

// Removing ONE of a hub's many referrers forces drop_out_refs to scan the
// hub's ENTIRE referrer list (find_if) to find and erase that one edge --
// this is the O(#referrers) cost CLAUDE.md's scope-boundaries section
// documents. Demonstrated by holding the REST of the model's size fixed
// and varying only the hub's fanout.
PERF_TEST(worst_case_removing_one_referrer_of_a_hub_scales_with_hub_fanout) {
    const std::vector<int> fanouts = {scaled(100), scaled(1000), scaled(10000)};
    std::vector<double> times_ms;
    for (int fanout : fanouts) {
        Model m;
        Ref<Account> hub;
        std::vector<Ref<Order>> hub_orders;
        seed_hub_with_fanout(m, fanout, /*background=*/1000, hub, hub_orders);

        const double t = time_ms([&] {
            Transaction txn = m.begin();
            txn.remove(hub_orders.back());
            (void)m.try_commit(txn);
        });
        times_ms.push_back(t);
        std::printf("  hub fanout=%6d   remove one referrer = %8.4f ms\n", fanout, t);
    }
    // Bounded, not flat -- this IS documented, expected growth (a known
    // tradeoff, not a bug). Catch only a further, unexpected blowup (e.g.
    // an accidental O(fanout^2)) on top of the already-linear cost.
    CHECK(times_ms.back() < 1000.0);
}

namespace {

// Account/Order have no non-nullable SELF-reference (Order.account always
// points at an Account, never another Order; Order.parent is nullable, so
// cascade NULLS it instead of propagating -- see demo/types.h). This type
// exists purely to build a genuinely deep, non-nullable reference chain.
class ChainNode final : public model::Object<ChainNode> {
public:
    std::string label;
    model::Ref<ChainNode> prev;  // non-nullable: this node dies when `prev` does

    template <class Self, class V>
    static void define_references(Self& s, V&& v) {
        v(model::field_tag<&ChainNode::prev>(), "prev", s.prev);
    }
};

// Builds head->...->tail (`depth` nodes), each node's `prev` pointing at
// the one before it. The head points at ITSELF -- a self-loop is how a
// non-nullable field survives having no real predecessor; remove_raw()'s
// cycle handling (see cascade_terminates_on_cycles in tests/tests.cpp)
// makes this a safe, already-tested shape, not a hack. Removing the head
// cascades through the WHOLE chain, one hop at a time.
Ref<ChainNode> build_chain(Model& m, int depth) {
    Transaction txn = m.begin();
    auto h = std::make_unique<ChainNode>();
    h->label = "N0";
    const Ref<ChainNode> head = txn.create(std::move(h));
    txn.update(head)->prev = head;  // self-loop; update() on a local id mutates in place

    Ref<ChainNode> prev = head;
    for (int i = 1; i < depth; ++i) {
        auto n = std::make_unique<ChainNode>();
        n->label = "N" + std::to_string(i);
        n->prev = prev;
        prev = txn.create(std::move(n));
    }
    CommitResult res = m.try_commit(txn);
    return res.to_real(head);
}

}  // namespace

// Removing the head of a long non-nullable reference chain cascades
// through every node, one hop at a time -- unbounded cascade fan-out from
// a single remove() intent, the OTHER cost CLAUDE.md's scope-boundaries
// section documents (distinct from the hub-fanout test above: this is
// cascade DEPTH, not breadth).
PERF_TEST(worst_case_removing_the_head_of_a_long_cascade_chain_scales_with_depth) {
    const std::vector<int> depths = {scaled(100), scaled(1000), scaled(10000)};
    std::vector<double> times_ms;
    for (int depth : depths) {
        Model m;
        const Ref<ChainNode> head = build_chain(m, depth);

        const double t = time_ms([&] {
            Transaction txn = m.begin();
            txn.remove(head);
            (void)m.try_commit(txn);
        });
        times_ms.push_back(t);
        std::printf("  chain depth=%6d   remove head (full cascade) = %8.4f ms\n", depth, t);

        CHECK(m.snapshot().size() == std::size_t{0});  // the whole chain died, not just the head
    }
    CHECK(times_ms.back() < 1000.0);
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
    CHECK(create_ms[1] < create_ms[0] * 80.0 + 200.0);
    CHECK(update_ms[1] < update_ms[0] * 80.0 + 5.0);
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
    CHECK(lookup_us[1] < lookup_us[0] * 5.0 + 5.0);
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
