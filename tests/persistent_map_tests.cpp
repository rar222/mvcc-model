// Dependency-free (of model.h) exerciser for the HAMT in persistent_map.h --
// see that file's own top comment for why it exists. Uses the SAME
// TEST()/CHECK()/g_failures harness as tests/test_*.cpp (test_harness.h,
// linked in via test_harness.cpp -- see CMakeLists.txt), so every test here
// is individually named and individually reported ([ RUN ]/[ OK ]/[ FAIL ]),
// exactly like model_tests. test_harness.h itself has no model.h dependency
// (just <cstdio>/<functional>/<string>/<vector>), so linking it in doesn't
// compromise this file's independence from the rest of the model.
//
// Run:  ctest --preset default -R persistent_map_tests --output-on-failure
//   or: ./build/default/persistent_map_tests
//
// Most checks here go through a `require(cond, what)`-style local lambda
// that prints a specific, labeled diagnostic (which KEY mismatched, which
// container, which Hash) and bumps g_failures directly, rather than a bare
// CHECK(cond) -- deliberately: a differential fuzz run against
// std::unordered_map/std::unordered_set is only useful for debugging if a
// failure says WHICH key diverged, not just that some CHECK on some line
// failed. This is still "using test_harness": g_failures is the exact same
// global CHECK()/CHECK_EQ() themselves increment, so TEST()'s pass/fail
// detection (diffing g_failures across a test body, in test_harness.cpp's
// own main()) works identically either way.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <map>
#include <memory_resource>
#include <random>
#include <string>
#include <unordered_map>
#include <unordered_set>

#include "model/persistent_map.h"
#include "test_harness.h"
using namespace model::pmap;

namespace {

// A key type distinct from std::string, to prove PersistentMap's key isn't
// hardcoded -- mirrors how model::IdHash hashes model::Id: a stateless
// functor, no scrambling, just enough to exercise the K/Hash template
// parameters end to end. Not model::IdHash itself: this file stays
// dependency-free of model.h, matching persistent_map.h's own independence
// from model::Id.
struct U64Hash {
    std::uint64_t operator()(std::uint64_t k) const noexcept { return k; }
};

// Degenerate on purpose: collapses every key onto one of three full 64-bit
// hash values, forcing long collision chains -- the Leaf::next path that a
// real hash essentially never exercises (two keys must share all 64 hash
// bits). Differential churn under this hash covers chain insert / replace /
// erase-at-head / erase-in-middle / iterate, which would otherwise be dead
// code as far as the test suite could tell.
struct ClashHash {
    std::uint64_t operator()(std::uint64_t k) const noexcept { return k % 3; }
};

// Identical body to U64Hash above, but ALSO declares is_perfect = true --
// genuinely so, since identity is bijective on uint64_t. Nothing in this
// file previously instantiated TrieCore with a Hash that opts into this:
// only model::IdHash (declared perfect in model.h) ever took that path, and
// only indirectly, through the full Model in tests/test_*.cpp, never in
// this dedicated low-level HAMT suite. Every churn/persistence/edge-case
// block below that uses this Hash exercises TrieCore's collision-chain-free
// Leaf shape (NoChain) and the "always replace, never append" branches in
// chain_set/chain_erase (persistent_map.h) for the first time in this file.
struct PerfectU64Hash {
    std::uint64_t operator()(std::uint64_t k) const noexcept { return k; }
    static constexpr bool is_perfect = true;
};

// A LIE: declares is_perfect = true (same promise PerfectU64Hash makes) but
// its body is ClashHash's -- genuinely colliding. Exists to trigger
// chain_set's "Hash declared is_perfect but produced a real collision"
// assert (persistent_map.h) from a real set() call -- see
// chain_set_asserts_when_a_hash_declared_perfect_actually_collides below,
// and that test's own comment for why chain_erase's identical-looking
// assert ISN'T reachable the same way.
struct LyingPerfectHash {
    std::uint64_t operator()(std::uint64_t k) const noexcept { return k % 3; }
    static constexpr bool is_perfect = true;
};

// Declares is_perfect = false EXPLICITLY, as opposed to U64Hash/StringHash/
// ClashHash simply never mentioning it -- exercises the other half of
// hash_is_perfect_v's std::bool_constant<Hash::is_perfect> branch (not just
// its SFINAE default-to-false fallback). Never instantiates a trie -- the
// static_asserts below are the entire test.
struct ExplicitlyImperfectHash {
    std::uint64_t operator()(std::uint64_t k) const noexcept { return k; }
    static constexpr bool is_perfect = false;
};

// A small POD VALUE type, distinct from the `int` every PersistentMap test
// above stores -- mirrors the shape of model::Id (two uint32 fields, memberwise
// equality) without depending on model.h (see this file's own top comment and
// U64Hash's comment for why). Root::by_key (model.h) is exactly
// PersistentMap<std::string, Id, StringHash>: a struct value, not a
// primitive. Never used as a KEY/hashed here -- only as the V in a
// PersistentMap<std::string, FakeId, StringHash> below -- so it needs
// equality but no hash.
struct FakeId {
    std::uint32_t index = 0;
    std::uint32_t gen = 0;
    // Hand-written, not `= default`: defaulting an operator== is C++20-only, and C++20
    // also synthesizes operator!= from it (rewritten candidates) -- something C++17
    // doesn't do, so any `!=` use elsewhere would silently stop compiling under C++17.
    // See model::Id::operator== (model.h) for the same reasoning.
    friend bool operator==(FakeId a, FakeId b) noexcept { return a.index == b.index && a.gen == b.gen; }
    friend bool operator!=(FakeId a, FakeId b) noexcept { return !(a == b); }
};

// Compile-time-only coverage of the trait-detection machinery itself
// (detail::hash_is_perfect_v, persistent_map.h) -- a template option in its
// own right, independent of any trie behavior it later gates.
static_assert(!detail::hash_is_perfect_v<StringHash>, "never declares is_perfect -> defaults false");
static_assert(!detail::hash_is_perfect_v<U64Hash>, "never declares is_perfect -> defaults false");
static_assert(!detail::hash_is_perfect_v<ClashHash>, "never declares is_perfect -> defaults false");
static_assert(!detail::hash_is_perfect_v<ExplicitlyImperfectHash>, "explicit false stays false");
static_assert(detail::hash_is_perfect_v<PerfectU64Hash>, "explicit true is detected");

// Deterministic (non-random) edge cases that the random-churn differential
// tests exercise only probabilistically, if at all: the empty-container
// start state (root_ == nullptr -- get_in/erase_in/each_in's `if (!n)`
// guards), erasing a key that was never present (erase_in's `return owner`
// no-op path, unchanged and zero-allocation -- see
// erase_of_absent_key_allocates_nothing_and_shares_the_original_root below
// for the allocation-count proof), and re-writing a key that's
// already there (chain_set's/chain_erase's "replace" branch under BOTH the
// chained and the perfect-hash Leaf shape, plus IdentityKeyOf's own
// documented "replacing a matched entry with itself is a harmless no-op"
// for PersistentSet). Templated on <K, Hash> so the identical checks run,
// with no risk of drifting apart, against a string-keyed default Hash, a
// u64-keyed default Hash, a deliberately colliding Hash, and a genuinely
// perfect Hash -- see the TEST()s that call these, below.
template <class K, class Hash>
void check_map_edge_cases(const char* label, const K& k1, const K& k2, const K& k_absent) {
    auto require = [&](bool cond, const char* what) {
        if (!cond) {
            std::printf("EDGE FAIL (map/%s): %s\n", label, what);
            ++g_failures;
        }
    };

    PersistentMap<K, int, Hash> m;
    require(m.size() == 0, "fresh map size == 0");
    require(m.empty(), "fresh map empty()");
    require(m.get(k1) == nullptr, "fresh map get() is null for any key");
    bool visited = false;
    m.for_each([&](const K&, int) { visited = true; });
    require(!visited, "fresh map for_each never invokes f");

    // Erase of an absent key from an EMPTY map (root_ == nullptr): a no-op,
    // not a crash -- erase_in's `if (!n) return owner;` guard (owner is
    // already null here, so this returns null right back).
    m = m.erase(k_absent);
    require(m.size() == 0, "erase on empty map stays empty");

    m = m.set(k1, 100);
    m = m.set(k2, 200);
    require(m.size() == 2, "two distinct keys -> size 2");

    // Erase of an absent key from a NON-empty map: erase_in's `return owner`
    // no-op path -- unrelated keys must be untouched.
    m = m.erase(k_absent);
    require(m.size() == 2, "erase of absent key leaves size unchanged");
    require(m.get(k1) && *m.get(k1) == 100, "erase of absent key: k1 survives");
    require(m.get(k2) && *m.get(k2) == 200, "erase of absent key: k2 survives");

    // set() of an ALREADY-PRESENT key: value updates, size does not grow --
    // chain_set's "replace" branch (KeyOf{}(lf->entry) == key, or, under a
    // perfect Hash, the unconditional replace).
    m = m.set(k1, 101);
    require(m.size() == 2, "re-set of existing key doesn't grow size");
    require(m.get(k1) && *m.get(k1) == 101, "re-set of existing key updates value");
    require(m.get(k2) && *m.get(k2) == 200, "re-set of k1 leaves k2 untouched");

    // Draining to empty exercises erase_in's "chain emptied"/"subtree
    // emptied, drop this slot" branches from a TINY map, not only as an
    // incidental side effect of the large random churn elsewhere.
    m = m.erase(k1);
    m = m.erase(k2);
    require(m.size() == 0, "erasing every key empties the map");
    require(m.empty(), "erasing every key -> empty() true");
    require(m.get(k1) == nullptr, "erased key no longer found");
}

template <class K, class Hash>
void check_set_edge_cases(const char* label, const K& k1, const K& k2, const K& k_absent) {
    auto require = [&](bool cond, const char* what) {
        if (!cond) {
            std::printf("EDGE FAIL (set/%s): %s\n", label, what);
            ++g_failures;
        }
    };

    PersistentSet<K, Hash> s;
    require(s.size() == 0, "fresh set size == 0");
    require(s.empty(), "fresh set empty()");
    require(!s.contains(k1), "fresh set contains() is false for any key");
    bool visited = false;
    s.for_each([&](const K&) { visited = true; });
    require(!visited, "fresh set for_each never invokes f");

    s = s.erase(k_absent);
    require(s.size() == 0, "erase on empty set stays empty");

    s = s.insert(k1);
    s = s.insert(k2);
    require(s.size() == 2, "two distinct keys -> size 2");

    s = s.erase(k_absent);
    require(s.size() == 2, "erase of absent key leaves size unchanged");
    require(s.contains(k1) && s.contains(k2), "erase of absent key leaves both present");

    // insert() of an ALREADY-PRESENT key: IdentityKeyOf's documented
    // "replacing a matched entry with itself is a correct, harmless no-op"
    // (see its own comment in persistent_map.h) -- size must not grow.
    s = s.insert(k1);
    require(s.size() == 2, "re-insert of existing key doesn't grow size");
    require(s.contains(k1) && s.contains(k2), "re-insert of existing key leaves both present");

    s = s.erase(k1);
    s = s.erase(k2);
    require(s.size() == 0, "erasing every key empties the set");
    require(s.empty(), "erasing every key -> empty() true");
}

// Nested-container coverage: PersistentMap<K, PersistentSet<V,Hash2>, Hash>.
// Root::by_cached_field and Root::by_cached_reference (model.h) both store a
// PersistentSet as the VALUE of an outer PersistentMap -- an outer map from
// a key (a cached field's string value, or a cached reference's target Id)
// to an inner "bucket" set of every Id currently holding that value. Every
// map test above stores a primitive/POD value; nothing yet nests one
// persistent container inside another as a value. check_nested_bucket_map
// exercises exactly src/model.cpp's own bucket discipline (see
// add_cached_fields/drop_cached_fields, add_cached_references/
// drop_cached_references): get-or-default on insert, and drop the outer key
// entirely when its bucket empties on erase -- not a simplified stand-in for
// that pattern, the same one, against a
// std::unordered_map<K, std::unordered_set<uint64_t>> reference.
template <class K, class Hash, class MakeKey>
void check_nested_bucket_map(const char* label, int iters, MakeKey make_key) {
    using Bucket = PersistentSet<std::uint64_t, PerfectU64Hash>;
    std::mt19937 rng(4242);
    PersistentMap<K, Bucket, Hash> outer;
    std::unordered_map<K, std::unordered_set<std::uint64_t>> ref;

    auto check_equal = [&](const char* where) {
        if (outer.size() != ref.size()) {
            std::printf("NESTED(%s) SIZE MISMATCH at %s: outer=%zu ref=%zu\n", label, where,
                        outer.size(), ref.size());
            ++g_failures;
        }
        for (auto& [k, bucket] : ref) {
            const Bucket* b = outer.get(k);
            if (!b) {
                std::printf("NESTED(%s) MISSING BUCKET at %s\n", label, where);
                ++g_failures;
                continue;
            }
            if (b->size() != bucket.size()) {
                std::printf("NESTED(%s) BUCKET SIZE MISMATCH at %s\n", label, where);
                ++g_failures;
            }
            for (std::uint64_t v : bucket) {
                if (!b->contains(v)) {
                    std::printf("NESTED(%s) BUCKET CONTAINS MISMATCH at %s\n", label, where);
                    ++g_failures;
                }
            }
        }
        // The drop-when-empty discipline means a present bucket must never be
        // empty -- an emptied bucket should have taken its outer key with it.
        outer.for_each([&](const K&, const Bucket& b) {
            if (b.empty()) {
                std::printf("NESTED(%s) EMPTY BUCKET LEFT BEHIND at %s\n", label, where);
                ++g_failures;
            }
        });
    };

    const int base = g_failures;
    for (int i = 0; i < iters && g_failures == base; i++) {
        K key = make_key(rng() % 40);
        std::uint64_t id = rng() % 500;
        if (rng() % 3) {
            // Insert: get-or-default then insert -- model.cpp's own pattern
            // (e.g. add_cached_fields), not `outer[key].insert(id)`.
            const Bucket* b = outer.get(key);
            outer = outer.set(key, (b ? *b : Bucket{}).insert(id));
            ref[key].insert(id);
        } else {
            // Erase: drop the outer key entirely when the bucket empties
            // (e.g. drop_cached_fields), so no bucket is ever present-but-empty.
            const Bucket* b = outer.get(key);
            if (b) {
                auto nb = b->erase(id);
                outer = nb.empty() ? outer.erase(key) : outer.set(key, nb);
            }
            auto it = ref.find(key);
            if (it != ref.end()) {
                it->second.erase(id);
                if (it->second.empty()) ref.erase(it);
            }
        }
        if (i % 200 == 0) check_equal("churn");
    }
    check_equal("final");
}

// Structural analog of Root::by_cached_field: PersistentMap<std::string,
// PersistentSet<Id,IdHash>, StringHash>. K/Hash are the SAME types the model
// actually uses (pmap::StringHash) -- only the bucket's element type is the
// U64Hash-family Id stand-in this file already relies on elsewhere.
TEST(nested_bucket_map_string_keyed_matches_by_cached_field_pattern) {
    check_nested_bucket_map<std::string, StringHash>(
        "string-keyed", 6000, [](std::uint32_t n) { return "k" + std::to_string(n); });
}

// Structural analog of Root::by_cached_reference: PersistentMap<Id,
// PersistentSet<Id,IdHash>, IdHash> -- an outer map keyed by the SAME
// perfect-hash family as model::IdHash (see PerfectU64Hash's own comment),
// whose value is itself a PersistentSet.
TEST(nested_bucket_map_perfect_hash_keyed_matches_by_cached_reference_pattern) {
    check_nested_bucket_map<std::uint64_t, PerfectU64Hash>(
        "perfect-hash-keyed", 6000, [](std::uint32_t n) { return static_cast<std::uint64_t>(n); });
}

// Structural analog of Root::by_key: PersistentMap<std::string, Id,
// StringHash> -- SAME K and Hash as the model (pmap::StringHash), with
// FakeId (see its own comment above) standing in for Id as a non-primitive
// VALUE, exercising set()/get()/erase()/for_each() copying a struct instead
// of an int.
TEST(string_keyed_map_with_struct_value_matches_unordered_map) {
    std::mt19937 rng(777);
    PersistentMap<std::string, FakeId, StringHash> pm;
    std::unordered_map<std::string, FakeId> ref;
    std::uint32_t next_gen = 1;

    const int base = g_failures;
    auto check_equal = [&](const char* where) {
        if (pm.size() != ref.size()) {
            std::printf("STRUCT-VAL SIZE MISMATCH at %s: pm=%zu ref=%zu\n", where, pm.size(),
                        ref.size());
            ++g_failures;
        }
        for (auto& [k, v] : ref) {
            const FakeId* p = pm.get(k);
            if (!p || !(*p == v)) {
                std::printf("STRUCT-VAL GET MISMATCH at %s key=%s\n", where, k.c_str());
                ++g_failures;
                return;
            }
        }
        size_t seen = 0;
        pm.for_each([&](const std::string& k, const FakeId& v) {
            auto it = ref.find(k);
            if (it == ref.end() || !(it->second == v)) {
                std::printf("STRUCT-VAL ITER EXTRA at %s key=%s\n", where, k.c_str());
                ++g_failures;
            }
            ++seen;
        });
        if (seen != ref.size()) {
            std::printf("STRUCT-VAL ITER COUNT at %s: %zu vs %zu\n", where, seen, ref.size());
            ++g_failures;
        }
    };

    for (int i = 0; i < 8000 && g_failures == base; i++) {
        std::string k = "f" + std::to_string(rng() % 800);
        if (rng() % 3) {
            FakeId v{static_cast<std::uint32_t>(rng() % 1000), next_gen++};
            pm = pm.set(k, v);
            ref[k] = v;
        } else {
            pm = pm.erase(k);
            ref.erase(k);
        }
        if (i % 300 == 0) check_equal("churn");
    }
    check_equal("final");
}

// Speed: set()/contains() must grow like O(log32 n) as the population
// grows, never O(n) -- a regression that turned path-copying into a full
// deep copy (or a lookup into a linear scan) would still pass every
// correctness check above; only a timing-SHAPE check catches it. Same
// self-contained best-of/steady_clock discipline as
// tests/performance_tests.cpp's own check_scaling -- duplicated here rather
// than shared, since this file is deliberately dependency-free of the rest
// of tests/ (see its own top comment), matching persistent_map.h's own
// independence from the rest of the model.
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

int scaled_n(int n) {
    return kSlowSanitizedBuild ? std::max(50, n / 50) : n;
}

template <class F>
double time_ms(F&& f) {
    const auto t0 = std::chrono::steady_clock::now();
    f();
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

// Minimum across trials: scheduling jitter/cache misses can only push a
// single reading UP, never down, so the minimum is the least-corrupted
// estimate -- same rationale as tests/performance_tests.cpp's best_of.
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

// NOT tested anywhere in this file: set_in's "ran out of hash bits"
// merge-into-one-chain fallback (persistent_map.h, guarded by
// `if (shift + 5 >= 64)`). It is provably unreachable by any two DISTINCT
// 64-bit hash values: 13 levels of 5-bit slices (shift = 0, 5, ..., 60)
// partition all 64 bits with no gaps, so two hashes that still haven't
// diverged by shift=60 have agreed on every bit and are therefore equal --
// contradicting the "different hash sharing this slot" precondition that
// guards entry into that branch in the first place. Reaching it would
// require a Hash::operator() that returns different values for the same
// key across calls (undefined behavior for any Hash, per this file's own
// differential tests, which all assume determinism) -- not a "general
// usage pattern," a broken Hash contract. Left as the one deliberately
// uncovered line, same as CLAUDE.md's own "Known scope boundaries"
// convention: documented, not silently skipped.

TEST(string_keyed_map_random_churn_matches_unordered_map) {
    std::mt19937 rng(99);
    PersistentMap<std::string, int, StringHash> pm;
    std::unordered_map<std::string, int> ref;

    const int base = g_failures;
    auto check_equal = [&](const char* where) {
        if (pm.size() != ref.size()) {
            std::printf("SIZE MISMATCH at %s: pm=%zu ref=%zu\n", where, pm.size(), ref.size());
            ++g_failures;
        }
        for (auto& [k, v] : ref) {
            const int* p = pm.get(k);
            if (!p || *p != v) {
                std::printf("GET MISMATCH at %s key=%s\n", where, k.c_str());
                ++g_failures;
                return;
            }
        }
        size_t seen = 0;
        pm.for_each([&](const std::string& k, int v) {
            auto it = ref.find(k);
            if (it == ref.end() || it->second != v) {
                std::printf("ITER EXTRA at %s key=%s\n", where, k.c_str());
                ++g_failures;
            }
            ++seen;
        });
        if (seen != ref.size()) {
            std::printf("ITER COUNT at %s: %zu vs %zu\n", where, seen, ref.size());
            ++g_failures;
        }
    };

    // Heavy random churn, differential against unordered_map.
    for (int i = 0; i < 20000 && g_failures == base; i++) {
        std::string k = "k" + std::to_string(rng() % 2000);
        if (rng() % 3) {
            int v = rng();
            pm = pm.set(k, v);
            ref[k] = v;
        } else {
            pm = pm.erase(k);
            ref.erase(k);
        }
        if (i % 500 == 0) check_equal("churn");
    }
    check_equal("final");
}

// Structural sharing / persistence: an old version must be unaffected by
// later edits to a derived version.
TEST(string_keyed_map_persists_old_version_across_derived_edits) {
    PersistentMap<std::string, int, StringHash> a;
    for (int i = 0; i < 1000; i++) a = a.set("s" + std::to_string(i), i);
    PersistentMap<std::string, int, StringHash> b = a;
    for (int i = 0; i < 1000; i++) b = b.set("s" + std::to_string(i), i + 100000);
    b = b.set("newkey", 7);
    b = b.erase("s500");
    // a must be pristine
    for (int i = 0; i < 1000; i++) {
        const int* p = a.get("s" + std::to_string(i));
        if (!p || *p != i) {
            std::printf("PERSIST FAIL: a[s%d] changed\n", i);
            ++g_failures;
            break;
        }
    }
    if (a.get("newkey")) {
        std::printf("PERSIST FAIL: newkey leaked into a\n");
        ++g_failures;
    }
    if (!a.get("s500")) {
        std::printf("PERSIST FAIL: s500 erased from a\n");
        ++g_failures;
    }
    CHECK_EQ(a.size(), std::size_t{1000});
    CHECK_EQ(b.size(), std::size_t{1000});
}

// Same differential churn, but keyed on std::uint64_t via U64Hash -- the
// shape by_type_/by_cached_reference_ need (see model.h's Root), with no
// string round-trip anywhere.
TEST(uint64_keyed_map_random_churn_matches_unordered_map) {
    std::mt19937 rng(99);
    PersistentMap<std::uint64_t, int, U64Hash> pm2;
    std::unordered_map<std::uint64_t, int> ref2;
    const int base = g_failures;
    auto check_equal_u64 = [&](const char* where) {
        if (pm2.size() != ref2.size()) {
            std::printf("U64 SIZE MISMATCH at %s: pm=%zu ref=%zu\n", where, pm2.size(), ref2.size());
            ++g_failures;
        }
        for (auto& [k, v] : ref2) {
            const int* p = pm2.get(k);
            if (!p || *p != v) {
                std::printf("U64 GET MISMATCH at %s key=%llu\n", where, static_cast<unsigned long long>(k));
                ++g_failures;
                return;
            }
        }
        size_t seen = 0;
        pm2.for_each([&](std::uint64_t k, int v) {
            auto it = ref2.find(k);
            if (it == ref2.end() || it->second != v) {
                std::printf("U64 ITER EXTRA at %s key=%llu\n", where, static_cast<unsigned long long>(k));
                ++g_failures;
            }
            ++seen;
        });
        if (seen != ref2.size()) {
            std::printf("U64 ITER COUNT at %s: %zu vs %zu\n", where, seen, ref2.size());
            ++g_failures;
        }
    };
    for (int i = 0; i < 20000 && g_failures == base; i++) {
        std::uint64_t k = rng() % 2000;
        if (rng() % 3) {
            int v = rng();
            pm2 = pm2.set(k, v);
            ref2[k] = v;
        } else {
            pm2 = pm2.erase(k);
            ref2.erase(k);
        }
        if (i % 500 == 0) check_equal_u64("churn");
    }
    check_equal_u64("final");
}

// PersistentSet<uint64_t>: same differential-churn discipline, against
// std::unordered_set instead of std::unordered_map -- no value half to
// check, just membership.
TEST(uint64_keyed_set_random_churn_matches_unordered_set) {
    std::mt19937 rng(99);
    PersistentSet<std::uint64_t, U64Hash> ps;
    std::unordered_set<std::uint64_t> refs;
    const int base = g_failures;
    auto check_equal_set = [&](const char* where) {
        if (ps.size() != refs.size()) {
            std::printf("SET SIZE MISMATCH at %s: ps=%zu ref=%zu\n", where, ps.size(), refs.size());
            ++g_failures;
        }
        for (std::uint64_t k : refs) {
            if (!ps.contains(k)) {
                std::printf("SET CONTAINS MISMATCH at %s key=%llu\n", where, static_cast<unsigned long long>(k));
                ++g_failures;
                return;
            }
        }
        size_t seen = 0;
        ps.for_each([&](std::uint64_t k) {
            if (refs.find(k) == refs.end()) {
                std::printf("SET ITER EXTRA at %s key=%llu\n", where, static_cast<unsigned long long>(k));
                ++g_failures;
            }
            ++seen;
        });
        if (seen != refs.size()) {
            std::printf("SET ITER COUNT at %s: %zu vs %zu\n", where, seen, refs.size());
            ++g_failures;
        }
    };
    for (int i = 0; i < 20000 && g_failures == base; i++) {
        std::uint64_t k = rng() % 2000;
        if (rng() % 3) {
            ps = ps.insert(k);
            refs.insert(k);
        } else {
            ps = ps.erase(k);
            refs.erase(k);
        }
        if (i % 500 == 0) check_equal_set("churn");
    }
    check_equal_set("final");
}

// Structural sharing / persistence, same discipline as the map case above.
TEST(uint64_keyed_set_persists_old_version_across_derived_edits) {
    PersistentSet<std::uint64_t, U64Hash> sa;
    for (std::uint64_t i = 0; i < 1000; i++) sa = sa.insert(i);
    PersistentSet<std::uint64_t, U64Hash> sb = sa;
    for (std::uint64_t i = 0; i < 1000; i++) sb = sb.erase(i);  // empty sb entirely
    sb = sb.insert(9999);
    // sa must be pristine
    for (std::uint64_t i = 0; i < 1000; i++) {
        if (!sa.contains(i)) {
            std::printf("SET PERSIST FAIL: sa missing %llu\n", static_cast<unsigned long long>(i));
            ++g_failures;
            break;
        }
    }
    if (sa.contains(9999)) {
        std::printf("SET PERSIST FAIL: 9999 leaked into sa\n");
        ++g_failures;
    }
    CHECK_EQ(sa.size(), std::size_t{1000});
    CHECK_EQ(sb.size(), std::size_t{1});
}

// Collision-chain coverage: same map churn as above, but under ClashHash,
// so every key lands in one of three long chains.
TEST(collision_chain_map_random_churn_matches_unordered_map) {
    std::mt19937 rng(99);
    PersistentMap<std::uint64_t, int, ClashHash> cm;
    std::unordered_map<std::uint64_t, int> cref;
    const int base = g_failures;
    auto check_cm = [&](const char* where) {
        if (cm.size() != cref.size()) {
            std::printf("CLASH-MAP SIZE MISMATCH at %s: pm=%zu ref=%zu\n", where, cm.size(),
                        cref.size());
            ++g_failures;
        }
        for (auto& [k, v] : cref) {
            const int* p = cm.get(k);
            if (!p || *p != v) {
                std::printf("CLASH-MAP GET MISMATCH at %s key=%llu\n", where, static_cast<unsigned long long>(k));
                ++g_failures;
                return;
            }
        }
        size_t seen = 0;
        cm.for_each([&](std::uint64_t k, int v) {
            auto it = cref.find(k);
            if (it == cref.end() || it->second != v) {
                std::printf("CLASH-MAP ITER EXTRA at %s key=%llu\n", where, static_cast<unsigned long long>(k));
                ++g_failures;
            }
            ++seen;
        });
        if (seen != cref.size()) {
            std::printf("CLASH-MAP ITER COUNT at %s: %zu vs %zu\n", where, seen, cref.size());
            ++g_failures;
        }
    };
    for (int i = 0; i < 6000 && g_failures == base; i++) {
        std::uint64_t k = rng() % 150;  // ~50-long chains
        if (rng() % 3) {
            int v = rng();
            cm = cm.set(k, v);
            cref[k] = v;
        } else {
            cm = cm.erase(k);
            cref.erase(k);
        }
        if (i % 500 == 0) check_cm("churn");
    }
    check_cm("final");
}

TEST(collision_chain_set_random_churn_matches_unordered_set) {
    std::mt19937 rng(99);
    PersistentSet<std::uint64_t, ClashHash> cs;
    std::unordered_set<std::uint64_t> cref;
    const int base = g_failures;
    auto check_cs = [&](const char* where) {
        if (cs.size() != cref.size()) {
            std::printf("CLASH-SET SIZE MISMATCH at %s: ps=%zu ref=%zu\n", where, cs.size(),
                        cref.size());
            ++g_failures;
        }
        for (std::uint64_t k : cref) {
            if (!cs.contains(k)) {
                std::printf("CLASH-SET CONTAINS MISMATCH at %s key=%llu\n", where,
                            static_cast<unsigned long long>(k));
                ++g_failures;
                return;
            }
        }
        size_t seen = 0;
        cs.for_each([&](std::uint64_t k) {
            if (cref.find(k) == cref.end()) {
                std::printf("CLASH-SET ITER EXTRA at %s key=%llu\n", where, static_cast<unsigned long long>(k));
                ++g_failures;
            }
            ++seen;
        });
        if (seen != cref.size()) {
            std::printf("CLASH-SET ITER COUNT at %s: %zu vs %zu\n", where, seen, cref.size());
            ++g_failures;
        }
    };
    for (int i = 0; i < 6000 && g_failures == base; i++) {
        std::uint64_t k = rng() % 150;
        if (rng() % 3) {
            cs = cs.insert(k);
            cref.insert(k);
        } else {
            cs = cs.erase(k);
            cref.erase(k);
        }
        if (i % 500 == 0) check_cs("churn");
    }
    check_cs("final");
}

// Persistence across chain edits: a derived version's chain surgery must
// not disturb the original's chains.
TEST(collision_chain_set_persists_old_version_across_chain_edits) {
    PersistentSet<std::uint64_t, ClashHash> ca;
    for (std::uint64_t i = 0; i < 90; i++) ca = ca.insert(i);
    PersistentSet<std::uint64_t, ClashHash> cb = ca;
    for (std::uint64_t i = 0; i < 90; i += 2) cb = cb.erase(i);  // gut half of every chain
    for (std::uint64_t i = 0; i < 90; i++) {
        if (!ca.contains(i)) {
            std::printf("CLASH PERSIST FAIL: ca missing %llu\n", static_cast<unsigned long long>(i));
            ++g_failures;
            break;
        }
    }
    CHECK_EQ(ca.size(), std::size_t{90});
    CHECK_EQ(cb.size(), std::size_t{45});
}

// Perfect-hash coverage: same differential-churn discipline as the U64Hash
// tests above, but under PerfectU64Hash (is_perfect = true) -- the ONE
// template option nothing in this file exercised directly before now (see
// PerfectU64Hash's own comment). Every set()/erase() below runs through
// TrieCore's NoChain Leaf and chain_set/chain_erase's perfect-hash
// branches, not the general chained ones.
TEST(perfect_hash_map_random_churn_matches_unordered_map) {
    std::mt19937 rng(99);
    PersistentMap<std::uint64_t, int, PerfectU64Hash> pm3;
    std::unordered_map<std::uint64_t, int> ref3;
    const int base = g_failures;
    auto check_pm3 = [&](const char* where) {
        if (pm3.size() != ref3.size()) {
            std::printf("PERFECT-MAP SIZE MISMATCH at %s: pm=%zu ref=%zu\n", where, pm3.size(),
                        ref3.size());
            ++g_failures;
        }
        for (auto& [k, v] : ref3) {
            const int* p = pm3.get(k);
            if (!p || *p != v) {
                std::printf("PERFECT-MAP GET MISMATCH at %s key=%llu\n", where,
                            static_cast<unsigned long long>(k));
                ++g_failures;
                return;
            }
        }
        size_t seen = 0;
        pm3.for_each([&](std::uint64_t k, int v) {
            auto it = ref3.find(k);
            if (it == ref3.end() || it->second != v) {
                std::printf("PERFECT-MAP ITER EXTRA at %s key=%llu\n", where,
                            static_cast<unsigned long long>(k));
                ++g_failures;
            }
            ++seen;
        });
        if (seen != ref3.size()) {
            std::printf("PERFECT-MAP ITER COUNT at %s: %zu vs %zu\n", where, seen, ref3.size());
            ++g_failures;
        }
    };
    for (int i = 0; i < 20000 && g_failures == base; i++) {
        std::uint64_t k = rng() % 2000;
        if (rng() % 3) {
            int v = rng();
            pm3 = pm3.set(k, v);
            ref3[k] = v;
        } else {
            pm3 = pm3.erase(k);
            ref3.erase(k);
        }
        if (i % 500 == 0) check_pm3("churn");
    }
    check_pm3("final");
}

// Persistence under the perfect-hash Leaf shape specifically: a derived
// version's set()/erase() (chain_set's/chain_erase's "replace"/"always
// empties" branches) must not disturb an older version sharing the same
// nodes.
TEST(perfect_hash_map_persists_old_version_across_derived_edits) {
    PersistentMap<std::uint64_t, int, PerfectU64Hash> pa;
    for (std::uint64_t i = 0; i < 1000; i++) pa = pa.set(i, static_cast<int>(i));
    PersistentMap<std::uint64_t, int, PerfectU64Hash> pb = pa;
    for (std::uint64_t i = 0; i < 1000; i++) pb = pb.set(i, static_cast<int>(i) + 100000);  // re-set every key
    pb = pb.set(9999, 7);
    pb = pb.erase(500);
    for (std::uint64_t i = 0; i < 1000; i++) {
        const int* p = pa.get(i);
        if (!p || *p != static_cast<int>(i)) {
            std::printf("PERFECT-MAP PERSIST FAIL: pa[%llu] changed\n", static_cast<unsigned long long>(i));
            ++g_failures;
            break;
        }
    }
    if (pa.get(9999)) {
        std::printf("PERFECT-MAP PERSIST FAIL: 9999 leaked into pa\n");
        ++g_failures;
    }
    if (!pa.get(500)) {
        std::printf("PERFECT-MAP PERSIST FAIL: 500 erased from pa\n");
        ++g_failures;
    }
    CHECK_EQ(pa.size(), std::size_t{1000});
    CHECK_EQ(pb.size(), std::size_t{1000});
}

TEST(perfect_hash_set_random_churn_matches_unordered_set) {
    std::mt19937 rng(99);
    PersistentSet<std::uint64_t, PerfectU64Hash> ps2;
    std::unordered_set<std::uint64_t> refs2;
    const int base = g_failures;
    auto check_ps2 = [&](const char* where) {
        if (ps2.size() != refs2.size()) {
            std::printf("PERFECT-SET SIZE MISMATCH at %s: ps=%zu ref=%zu\n", where, ps2.size(),
                        refs2.size());
            ++g_failures;
        }
        for (std::uint64_t k : refs2) {
            if (!ps2.contains(k)) {
                std::printf("PERFECT-SET CONTAINS MISMATCH at %s key=%llu\n", where,
                            static_cast<unsigned long long>(k));
                ++g_failures;
                return;
            }
        }
        size_t seen = 0;
        ps2.for_each([&](std::uint64_t k) {
            if (refs2.find(k) == refs2.end()) {
                std::printf("PERFECT-SET ITER EXTRA at %s key=%llu\n", where,
                            static_cast<unsigned long long>(k));
                ++g_failures;
            }
            ++seen;
        });
        if (seen != refs2.size()) {
            std::printf("PERFECT-SET ITER COUNT at %s: %zu vs %zu\n", where, seen, refs2.size());
            ++g_failures;
        }
    };
    for (int i = 0; i < 20000 && g_failures == base; i++) {
        std::uint64_t k = rng() % 2000;
        if (rng() % 3) {
            ps2 = ps2.insert(k);
            refs2.insert(k);
        } else {
            ps2 = ps2.erase(k);
            refs2.erase(k);
        }
        if (i % 500 == 0) check_ps2("churn");
    }
    check_ps2("final");
}

// Draining an entire populated set back to empty, under the perfect-hash
// Leaf shape -- exercises chain_erase's "always empties the slot" branch,
// and erase_in's subtree-collapse (bitmap shrink) branch, repeatedly, from
// a real, non-trivial multi-level tree.
TEST(perfect_hash_set_persists_old_version_after_draining_to_empty) {
    PersistentSet<std::uint64_t, PerfectU64Hash> sa2;
    for (std::uint64_t i = 0; i < 1000; i++) sa2 = sa2.insert(i);
    PersistentSet<std::uint64_t, PerfectU64Hash> sb2 = sa2;
    for (std::uint64_t i = 0; i < 1000; i++) sb2 = sb2.erase(i);  // empty sb2 entirely
    sb2 = sb2.insert(9999);
    for (std::uint64_t i = 0; i < 1000; i++) {
        if (!sa2.contains(i)) {
            std::printf("PERFECT-SET PERSIST FAIL: sa2 missing %llu\n", static_cast<unsigned long long>(i));
            ++g_failures;
            break;
        }
    }
    if (sa2.contains(9999)) {
        std::printf("PERFECT-SET PERSIST FAIL: 9999 leaked into sa2\n");
        ++g_failures;
    }
    CHECK_EQ(sa2.size(), std::size_t{1000});
    CHECK_EQ(sb2.size(), std::size_t{1});
}

// Deterministic edge cases (see check_map_edge_cases/check_set_edge_cases'
// own comment), one TEST() per Hash flavor this file defines: a
// string-keyed default Hash, a u64-keyed default Hash, a deliberately
// colliding Hash, and a genuinely perfect Hash. Random churn above hits
// most of these paths eventually but never deterministically or in
// isolation.
TEST(map_edge_cases_string_hash) {
    check_map_edge_cases<std::string, StringHash>("StringHash", "alpha", "beta", "nonexistent");
}
TEST(map_edge_cases_u64_hash) {
    check_map_edge_cases<std::uint64_t, U64Hash>("U64Hash", 1, 2, 999);
}
TEST(map_edge_cases_clash_hash) {
    check_map_edge_cases<std::uint64_t, ClashHash>("ClashHash", 1, 2, 999);
}
TEST(map_edge_cases_perfect_u64_hash) {
    check_map_edge_cases<std::uint64_t, PerfectU64Hash>("PerfectU64Hash", 1, 2, 999);
}
TEST(set_edge_cases_string_hash) {
    check_set_edge_cases<std::string, StringHash>("StringHash", "alpha", "beta", "nonexistent");
}
TEST(set_edge_cases_u64_hash) {
    check_set_edge_cases<std::uint64_t, U64Hash>("U64Hash", 1, 2, 999);
}
TEST(set_edge_cases_clash_hash) {
    check_set_edge_cases<std::uint64_t, ClashHash>("ClashHash", 1, 2, 999);
}
TEST(set_edge_cases_perfect_u64_hash) {
    check_set_edge_cases<std::uint64_t, PerfectU64Hash>("PerfectU64Hash", 1, 2, 999);
}

// Counts allocations so erase_key()'s no-op-vs-real-work cost can be proven
// directly, not just inferred from behavior. erase_in used to unconditionally
// clone_node() every level on its way down BEFORE discovering, at the
// bottom, that the key wasn't there -- same allocation count as a real
// erase, wasted. The fix (erase_in's `return owner` short-circuit, see its
// own comment in persistent_map.h) makes a no-op erase cost exactly what
// get() costs: a trie walk with zero allocations, because it returns the
// original root unchanged instead of rebuilding an equal one.
struct CountingResource : std::pmr::memory_resource {
    std::size_t allocations = 0;
    void* do_allocate(std::size_t bytes, std::size_t align) override {
        ++allocations;
        return std::pmr::new_delete_resource()->allocate(bytes, align);
    }
    void do_deallocate(void* p, std::size_t bytes, std::size_t align) noexcept override {
        std::pmr::new_delete_resource()->deallocate(p, bytes, align);
    }
    bool do_is_equal(const std::pmr::memory_resource& other) const noexcept override {
        return this == &other;
    }
};

TEST(erase_of_absent_key_allocates_nothing_and_shares_the_original_root) {
    CountingResource mem;
    PersistentMap<std::uint64_t, int, U64Hash> m(&mem);
    for (std::uint64_t k = 0; k < 500; ++k) m = m.set(k, static_cast<int>(k));

    const std::size_t before = mem.allocations;
    PersistentMap<std::uint64_t, int, U64Hash> after_noop = m.erase(std::uint64_t{999999});
    CHECK_EQ(mem.allocations, before);  // zero allocations for a no-op erase
    CHECK_EQ(after_noop.size(), m.size());
    CHECK(after_noop.get(std::uint64_t{999999}) == nullptr);
    CHECK(after_noop.get(std::uint64_t{7}) && *after_noop.get(std::uint64_t{7}) == 7);

    // Contrast: an erase that actually removes a key DOES allocate (the
    // necessary O(log32 n) path-copy, same shape as set()) -- proving the
    // zero above is specific to the no-op case, not "this map never
    // allocates once built."
    PersistentMap<std::uint64_t, int, U64Hash> after_real = m.erase(std::uint64_t{7});
    CHECK(mem.allocations > before);
    CHECK_EQ(after_real.size(), m.size() - 1);
    CHECK(after_real.get(std::uint64_t{7}) == nullptr);
}

// chain_set's kPerfectHash branch trusts its caller's promise (Hash::
// is_perfect == true) unconditionally -- no per-entry key comparison, unlike
// erase_in, which only ever calls chain_erase() after leaf_get() has already
// confirmed presence via a real key comparison (see persistent_map.h's own
// comments on both). A Hash that lies about being perfect makes that trust
// observable: two DIFFERENT keys sharing a hash under LyingPerfectHash reach
// set_in's "same hash, replace-or-append" branch (lf_hash == hash), which
// calls chain_set() believing the existing leaf must already BE this key --
// and it isn't. (chain_erase's identical-looking assert has no equivalent
// path: leaf_get already ruled out anything but a genuine match before
// chain_erase is ever called, so it can't observe this class of lie.)
TEST(chain_set_asserts_when_a_hash_declared_perfect_actually_collides) {
    PersistentMap<std::uint64_t, int, LyingPerfectHash> m;
    m = m.set(std::uint64_t{1}, 100);  // fine: first (and, per the lie, only) key at this hash

    // 4 % 3 == 1 % 3 == 1: a real collision LyingPerfectHash claims can't happen.
    CHECK(dies_of_assert([&] {
        PersistentMap<std::uint64_t, int, LyingPerfectHash> m2 = m.set(std::uint64_t{4}, 200);
        (void)m2;
    }));
}

// for_each_short_circuit's whole reason to exist (see its own comment in
// persistent_map.h) is that it stops the underlying trie walk itself, not
// just further calls to `f` -- a distinction for_each's void contract can't
// express. Counting invocations is the only way to prove that from outside
// TrieCore: a fake short-circuit built by wrapping for_each in an "if (ok)"
// guard would still report exactly 1 CALL to the user's predicate too, but
// would have walked all `n` entries doing it. 500 entries makes that
// difference unmistakable if this regresses to the fake version.
template <class K, class Hash, class MakeKey>
void check_map_for_each_short_circuit(const char* label, int n, MakeKey make_key) {
    auto require = [&](bool cond, const char* what) {
        if (!cond) {
            std::printf("SHORT-CIRCUIT FAIL (map/%s): %s\n", label, what);
            ++g_failures;
        }
    };

    PersistentMap<K, int, Hash> m;
    for (int i = 0; i < n; ++i) m = m.set(make_key(i), i);

    // Empty map: vacuously true, f never called.
    PersistentMap<K, int, Hash> empty;
    bool empty_visited = false;
    const bool empty_completed =
        empty.for_each_short_circuit([&](const K&, int) {
            empty_visited = true;
            return true;
        });
    require(empty_completed, "empty map: walk reports completed");
    require(!empty_visited, "empty map: f never invoked");

    // Never stopping: completes, and visits exactly `n` entries -- same
    // coverage as for_each, just via the bool-returning callback.
    int full_count = 0;
    const bool full_completed = m.for_each_short_circuit([&](const K&, int) {
        ++full_count;
        return true;
    });
    require(full_completed, "never stopping -> walk reports completed");
    require(full_count == n, "never stopping -> visits every entry");

    // Stopping on the very first call: walk reports NOT completed, and --
    // the actual point of this primitive -- exactly ONE entry was visited,
    // not all n.
    int stop_count = 0;
    const bool stop_completed = m.for_each_short_circuit([&](const K&, int) {
        ++stop_count;
        return false;
    });
    require(!stop_completed, "stopping on first call -> walk reports NOT completed");
    require(stop_count == 1, "stopping on first call -> exactly one entry visited, not all n");
}

template <class K, class Hash, class MakeKey>
void check_set_for_each_short_circuit(const char* label, int n, MakeKey make_key) {
    auto require = [&](bool cond, const char* what) {
        if (!cond) {
            std::printf("SHORT-CIRCUIT FAIL (set/%s): %s\n", label, what);
            ++g_failures;
        }
    };

    PersistentSet<K, Hash> s;
    for (int i = 0; i < n; ++i) s = s.insert(make_key(i));

    PersistentSet<K, Hash> empty;
    bool empty_visited = false;
    const bool empty_completed = empty.for_each_short_circuit([&](const K&) {
        empty_visited = true;
        return true;
    });
    require(empty_completed, "empty set: walk reports completed");
    require(!empty_visited, "empty set: f never invoked");

    int full_count = 0;
    const bool full_completed = s.for_each_short_circuit([&](const K&) {
        ++full_count;
        return true;
    });
    require(full_completed, "never stopping -> walk reports completed");
    require(full_count == n, "never stopping -> visits every entry");

    int stop_count = 0;
    const bool stop_completed = s.for_each_short_circuit([&](const K&) {
        ++stop_count;
        return false;
    });
    require(!stop_completed, "stopping on first call -> walk reports NOT completed");
    require(stop_count == 1, "stopping on first call -> exactly one entry visited, not all n");
}

TEST(for_each_short_circuit_map_u64_hash) {
    check_map_for_each_short_circuit<std::uint64_t, U64Hash>(
        "U64Hash", 500, [](int i) { return static_cast<std::uint64_t>(i); });
}
TEST(for_each_short_circuit_map_perfect_u64_hash) {
    check_map_for_each_short_circuit<std::uint64_t, PerfectU64Hash>(
        "PerfectU64Hash", 500, [](int i) { return static_cast<std::uint64_t>(i); });
}
// ClashHash collapses every key onto one of three hash values, forcing long
// collision chains -- exercises each_in_short_circuit's inner chain loop
// (`for (l = ...; l; l = l->next.get()) if (!f(...)) return false;`), which
// U64Hash/PerfectU64Hash's short (depth <=1) chains never reach.
TEST(for_each_short_circuit_map_clash_hash) {
    check_map_for_each_short_circuit<std::uint64_t, ClashHash>(
        "ClashHash", 500, [](int i) { return static_cast<std::uint64_t>(i); });
}
TEST(for_each_short_circuit_set_u64_hash) {
    check_set_for_each_short_circuit<std::uint64_t, U64Hash>(
        "U64Hash", 500, [](int i) { return static_cast<std::uint64_t>(i); });
}
TEST(for_each_short_circuit_set_clash_hash) {
    check_set_for_each_short_circuit<std::uint64_t, ClashHash>(
        "ClashHash", 500, [](int i) { return static_cast<std::uint64_t>(i); });
}

// Range-based-for correctness: iterating `for (auto& [k, v] : m)` /
// `for (auto& k : s)` must visit exactly the same multiset of entries as
// for_each -- differential against for_each itself (not against
// std::unordered_map/set again) since for_each is already covered by every
// churn test above and is the thing the new external iterator must agree
// with, entry for entry, in count and content. Covers the three Leaf/chain
// shapes this file's Hash zoo exists to reach: ordinary (U64Hash), perfect
// (PerfectU64Hash, NoChain), and long collision chains (ClashHash) -- the
// iterator's leaf-chain-resume branch (advance()'s `if (chain_)`) is dead
// code without that last one.
template <class K, class Hash, class MakeKey>
void check_map_range_for_matches_for_each(const char* label, int n, MakeKey make_key) {
    auto require = [&](bool cond, const char* what) {
        if (!cond) {
            std::printf("RANGE-FOR FAIL (map/%s): %s\n", label, what);
            ++g_failures;
        }
    };

    PersistentMap<K, int, Hash> m;
    for (int i = 0; i < n; ++i) m = m.set(make_key(i), i);

    std::unordered_map<K, int> via_for_each;
    m.for_each([&](const K& k, int v) { via_for_each[k] = v; });

    std::unordered_map<K, int> via_range_for;
    std::size_t seen = 0;
    for (const auto& [k, v] : m) {
        via_range_for[k] = v;
        ++seen;
    }

    require(seen == via_for_each.size(), "range-for visits the same entry count as for_each");
    require(via_range_for.size() == via_for_each.size(), "range-for visits no duplicate keys");
    require(via_range_for == via_for_each, "range-for visits the same (key, value) pairs as for_each");

    // Empty map: begin() == end() immediately, no entries visited.
    PersistentMap<K, int, Hash> empty;
    int empty_seen = 0;
    for (const auto& kv : empty) {
        (void)kv;
        ++empty_seen;
    }
    require(empty_seen == 0, "empty map: range-for visits nothing");
    require(empty.begin() == empty.end(), "empty map: begin() == end()");
}

template <class K, class Hash, class MakeKey>
void check_set_range_for_matches_for_each(const char* label, int n, MakeKey make_key) {
    auto require = [&](bool cond, const char* what) {
        if (!cond) {
            std::printf("RANGE-FOR FAIL (set/%s): %s\n", label, what);
            ++g_failures;
        }
    };

    PersistentSet<K, Hash> s;
    for (int i = 0; i < n; ++i) s = s.insert(make_key(i));

    std::unordered_set<K> via_for_each;
    s.for_each([&](const K& k) { via_for_each.insert(k); });

    std::unordered_set<K> via_range_for;
    std::size_t seen = 0;
    for (const K& k : s) {
        via_range_for.insert(k);
        ++seen;
    }

    require(seen == via_for_each.size(), "range-for visits the same entry count as for_each");
    require(via_range_for == via_for_each, "range-for visits the same keys as for_each");

    PersistentSet<K, Hash> empty;
    int empty_seen = 0;
    for (const K& k : empty) {
        (void)k;
        ++empty_seen;
    }
    require(empty_seen == 0, "empty set: range-for visits nothing");
    require(empty.begin() == empty.end(), "empty set: begin() == end()");
}

TEST(range_for_map_u64_hash_matches_for_each) {
    check_map_range_for_matches_for_each<std::uint64_t, U64Hash>(
        "U64Hash", 500, [](int i) { return static_cast<std::uint64_t>(i); });
}
TEST(range_for_map_perfect_u64_hash_matches_for_each) {
    check_map_range_for_matches_for_each<std::uint64_t, PerfectU64Hash>(
        "PerfectU64Hash", 500, [](int i) { return static_cast<std::uint64_t>(i); });
}
TEST(range_for_map_clash_hash_matches_for_each) {
    // Long collision chains: exercises advance()'s leaf-chain-resume branch.
    check_map_range_for_matches_for_each<std::uint64_t, ClashHash>(
        "ClashHash", 500, [](int i) { return static_cast<std::uint64_t>(i); });
}
TEST(range_for_map_string_hash_matches_for_each) {
    check_map_range_for_matches_for_each<std::string, StringHash>(
        "StringHash", 500, [](int i) { return "k" + std::to_string(i); });
}
TEST(range_for_set_u64_hash_matches_for_each) {
    check_set_range_for_matches_for_each<std::uint64_t, U64Hash>(
        "U64Hash", 500, [](int i) { return static_cast<std::uint64_t>(i); });
}
TEST(range_for_set_perfect_u64_hash_matches_for_each) {
    check_set_range_for_matches_for_each<std::uint64_t, PerfectU64Hash>(
        "PerfectU64Hash", 500, [](int i) { return static_cast<std::uint64_t>(i); });
}
TEST(range_for_set_clash_hash_matches_for_each) {
    check_set_range_for_matches_for_each<std::uint64_t, ClashHash>(
        "ClashHash", 500, [](int i) { return static_cast<std::uint64_t>(i); });
}

// Speed: set()/contains() latency must grow like O(log32 n), not O(n), as
// the population grows -- see the timing helpers' own comment. Run against
// PersistentSet<uint64_t, PerfectU64Hash>: the newest code path (NoChain,
// merged children/leaves slots) and the one most representative of the
// model's actual hot instantiations (Root::by_type, by_cached_reference --
// both keyed by model::Id via model::IdHash, itself declared perfect).
TEST(speed_set_and_contains_grow_like_log_n_not_linear) {
    using PSet = PersistentSet<std::uint64_t, PerfectU64Hash>;
    const int n_small = scaled_n(2000);
    const int n_large = scaled_n(200000);

    PSet base_small;
    for (int i = 0; i < n_small; ++i) base_small = base_small.insert(static_cast<std::uint64_t>(i));
    PSet base_large;
    for (int i = 0; i < n_large; ++i) base_large = base_large.insert(static_cast<std::uint64_t>(i));

    constexpr int kOpsPerTrial = 200;
    // Fresh, never-before-seen keys each trial (well past either
    // population's range) -- an insert, not a replace, every time: the
    // worst case for path-copying, and the one that actually allocates.
    auto set_op_us = [&](const PSet& base) {
        return best_of(5,
                       [&] {
                           return time_ms([&] {
                               PSet t = base;
                               for (int i = 0; i < kOpsPerTrial; ++i)
                                   t = t.insert(10000000ull + static_cast<std::uint64_t>(i));
                           });
                       }) *
              1000.0 / kOpsPerTrial;
    };
    auto get_op_us = [&](const PSet& base, int n) {
        return best_of(5,
                       [&] {
                           return time_ms([&] {
                               for (int i = 0; i < kOpsPerTrial; ++i) {
                                   const bool c = base.contains(static_cast<std::uint64_t>(i % n));
                                   if (!c) std::printf("SPEED SETUP BUG: expected key missing\n");
                               }
                           });
                       }) *
              1000.0 / kOpsPerTrial;
    };

    const double set_small_us = set_op_us(base_small);
    const double set_large_us = set_op_us(base_large);
    const double get_small_us = get_op_us(base_small, n_small);
    const double get_large_us = get_op_us(base_large, n_large);

    const double n_ratio = static_cast<double>(n_large) / n_small;
    const double log_ratio = std::log(static_cast<double>(n_large)) / std::log(static_cast<double>(n_small));
    const double set_ratio = set_small_us > 0 ? set_large_us / set_small_us : 0.0;
    const double get_ratio = get_small_us > 0 ? get_large_us / get_small_us : 0.0;

    std::printf("\nspeed (PersistentSet<uint64_t, PerfectU64Hash>, n: %d -> %d, %.0fx):\n", n_small,
                n_large, n_ratio);
    std::printf(
        "  insert()   %8.4f us -> %8.4f us   (x%.2f; O(log n) predicts x%.2f, O(n) predicts "
        "x%.0f)\n",
        set_small_us, set_large_us, set_ratio, log_ratio, n_ratio);
    std::printf(
        "  contains() %8.4f us -> %8.4f us   (x%.2f; O(log n) predicts x%.2f, O(n) predicts "
        "x%.0f)\n",
        get_small_us, get_large_us, get_ratio, log_ratio, n_ratio);

    // Asserted only outside a sanitizer build: under asan/tsan the
    // population is 50x smaller (scaled_n) and per-operation
    // instrumentation overhead is not proportional to algorithmic work --
    // same rationale as tests/performance_tests.cpp's kAssertTimings. A
    // generous ceiling, not a tight band: real O(log32 n) growth here
    // predicts roughly x1.6 (log32 of a 100x population increase); this
    // only needs to catch a regression toward O(n) (which would show
    // ~100x) or O(n log n), not pin the exact constant.
    if (!kSlowSanitizedBuild) {
        if (set_ratio > 10.0) {
            std::printf(
                "SPEED FAIL: insert() grew x%.2f over a %.0fx population increase -- looks "
                "linear, not log n\n",
                set_ratio, n_ratio);
            ++g_failures;
        }
        if (get_ratio > 10.0) {
            std::printf(
                "SPEED FAIL: contains() grew x%.2f over a %.0fx population increase -- looks "
                "linear, not log n\n",
                get_ratio, n_ratio);
            ++g_failures;
        }
    } else {
        std::printf("speed check SKIPPED (sanitizer build, see kSlowSanitizedBuild)\n");
    }
}

// Speed: callback (for_each_short_circuit) vs. range-based for, over the
// exact PersistentSet<Id, IdHash> shape model.h's by_type_/by_cached_
// reference indices use (PerfectU64Hash stands in for model::IdHash here --
// same "declared perfect" contract, see PerfectU64Hash's own comment).
// for_each_short_circuit's walk is one recursive function with `f` as a
// template parameter, so at -O2 it and the lambda body typically inline into
// a single loop with no reified state. The range-for form instead resumes
// an explicit (stack_, chain_) state machine (TrieCore::iterator::advance)
// on every single ++ -- real per-step bookkeeping the recursive form never
// pays. Both sides run to completion, summing every entry -- the "never
// stop early" shape model.h's for_each_by_field/for_each_referrers both use
// (see their own doc comments), i.e. the case those call sites would
// actually pay if converted to range-for.
TEST(speed_callback_vs_range_for_full_scan) {
    using PSet = PersistentSet<std::uint64_t, PerfectU64Hash>;
    const int n = scaled_n(500000);

    PSet s;
    for (int i = 0; i < n; ++i) s = s.insert(static_cast<std::uint64_t>(i));

    volatile std::uint64_t sink = 0;
    const double via_callback_ms = best_of(5, [&] {
        return time_ms([&] {
            std::uint64_t total = 0;
            s.for_each_short_circuit([&](std::uint64_t k) {
                total += k;
                return true;
            });
            sink = total;
        });
    });
    const double via_range_for_ms = best_of(5, [&] {
        return time_ms([&] {
            std::uint64_t total = 0;
            for (std::uint64_t k : s) total += k;
            sink = total;
        });
    });
    (void)sink;

    std::printf(
        "\nspeed (PersistentSet<uint64_t, PerfectU64Hash>, n=%d, full scan summing every "
        "entry):\n"
        "  for_each_short_circuit  %8.3f ms\n"
        "  range-based for         %8.3f ms   (x%.2f vs. callback)\n",
        n, via_callback_ms, via_range_for_ms, via_range_for_ms / via_callback_ms);

    // Not asserted: this is a measurement to inform a design decision (which
    // for_each_*/all_of_* call sites, if any, are worth converting to range-
    // for), not a regression gate -- there is no "correct" ratio to pin, and
    // under a sanitizer build instrumentation overhead swamps the constant
    // factor being measured anyway (kSlowSanitizedBuild, same rationale as
    // the speed test above).
    if (kSlowSanitizedBuild) {
        std::printf("(sanitizer build -- ratio above is not representative, see kSlowSanitizedBuild)\n");
    }
}
