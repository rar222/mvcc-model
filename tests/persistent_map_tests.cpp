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

// Declares is_perfect = false EXPLICITLY, as opposed to U64Hash/StringHash/
// ClashHash simply never mentioning it -- exercises the other half of
// hash_is_perfect_v's std::bool_constant<Hash::is_perfect> branch (not just
// its SFINAE default-to-false fallback). Never instantiates a trie -- the
// static_asserts below are the entire test.
struct ExplicitlyImperfectHash {
    std::uint64_t operator()(std::uint64_t k) const noexcept { return k; }
    static constexpr bool is_perfect = false;
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
// guards), erasing a key that was never present (copy_shared's "not
// present, reuse unchanged" path in erase_in), and re-writing a key that's
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
    // not a crash -- erase_in's `if (!n) return nullptr;` guard.
    m = m.erase(k_absent);
    require(m.size() == 0, "erase on empty map stays empty");

    m = m.set(k1, 100);
    m = m.set(k2, 200);
    require(m.size() == 2, "two distinct keys -> size 2");

    // Erase of an absent key from a NON-empty map: copy_shared's "not
    // present" path -- unrelated keys must be untouched.
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
                std::printf("U64 GET MISMATCH at %s key=%llu\n", where, (unsigned long long)k);
                ++g_failures;
                return;
            }
        }
        size_t seen = 0;
        pm2.for_each([&](std::uint64_t k, int v) {
            auto it = ref2.find(k);
            if (it == ref2.end() || it->second != v) {
                std::printf("U64 ITER EXTRA at %s key=%llu\n", where, (unsigned long long)k);
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
        std::uint64_t k = (std::uint64_t)(rng() % 2000);
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
                std::printf("SET CONTAINS MISMATCH at %s key=%llu\n", where, (unsigned long long)k);
                ++g_failures;
                return;
            }
        }
        size_t seen = 0;
        ps.for_each([&](std::uint64_t k) {
            if (refs.find(k) == refs.end()) {
                std::printf("SET ITER EXTRA at %s key=%llu\n", where, (unsigned long long)k);
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
        std::uint64_t k = (std::uint64_t)(rng() % 2000);
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
            std::printf("SET PERSIST FAIL: sa missing %llu\n", (unsigned long long)i);
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
                std::printf("CLASH-MAP GET MISMATCH at %s key=%llu\n", where, (unsigned long long)k);
                ++g_failures;
                return;
            }
        }
        size_t seen = 0;
        cm.for_each([&](std::uint64_t k, int v) {
            auto it = cref.find(k);
            if (it == cref.end() || it->second != v) {
                std::printf("CLASH-MAP ITER EXTRA at %s key=%llu\n", where, (unsigned long long)k);
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
        std::uint64_t k = (std::uint64_t)(rng() % 150);  // ~50-long chains
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
                            (unsigned long long)k);
                ++g_failures;
                return;
            }
        }
        size_t seen = 0;
        cs.for_each([&](std::uint64_t k) {
            if (cref.find(k) == cref.end()) {
                std::printf("CLASH-SET ITER EXTRA at %s key=%llu\n", where, (unsigned long long)k);
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
        std::uint64_t k = (std::uint64_t)(rng() % 150);
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
            std::printf("CLASH PERSIST FAIL: ca missing %llu\n", (unsigned long long)i);
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
                            (unsigned long long)k);
                ++g_failures;
                return;
            }
        }
        size_t seen = 0;
        pm3.for_each([&](std::uint64_t k, int v) {
            auto it = ref3.find(k);
            if (it == ref3.end() || it->second != v) {
                std::printf("PERFECT-MAP ITER EXTRA at %s key=%llu\n", where,
                            (unsigned long long)k);
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
        std::uint64_t k = (std::uint64_t)(rng() % 2000);
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
    for (std::uint64_t i = 0; i < 1000; i++) pa = pa.set(i, (int)i);
    PersistentMap<std::uint64_t, int, PerfectU64Hash> pb = pa;
    for (std::uint64_t i = 0; i < 1000; i++) pb = pb.set(i, (int)i + 100000);  // re-set every key
    pb = pb.set(9999, 7);
    pb = pb.erase(500);
    for (std::uint64_t i = 0; i < 1000; i++) {
        const int* p = pa.get(i);
        if (!p || *p != (int)i) {
            std::printf("PERFECT-MAP PERSIST FAIL: pa[%llu] changed\n", (unsigned long long)i);
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
                            (unsigned long long)k);
                ++g_failures;
                return;
            }
        }
        size_t seen = 0;
        ps2.for_each([&](std::uint64_t k) {
            if (refs2.find(k) == refs2.end()) {
                std::printf("PERFECT-SET ITER EXTRA at %s key=%llu\n", where,
                            (unsigned long long)k);
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
        std::uint64_t k = (std::uint64_t)(rng() % 2000);
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
            std::printf("PERFECT-SET PERSIST FAIL: sa2 missing %llu\n", (unsigned long long)i);
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
    for (int i = 0; i < n_small; ++i) base_small = base_small.insert((std::uint64_t)i);
    PSet base_large;
    for (int i = 0; i < n_large; ++i) base_large = base_large.insert((std::uint64_t)i);

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
                                   t = t.insert(10000000ull + (std::uint64_t)i);
                           });
                       }) *
              1000.0 / kOpsPerTrial;
    };
    auto get_op_us = [&](const PSet& base, int n) {
        return best_of(5,
                       [&] {
                           return time_ms([&] {
                               for (int i = 0; i < kOpsPerTrial; ++i) {
                                   const bool c = base.contains((std::uint64_t)(i % n));
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

    const double n_ratio = (double)n_large / n_small;
    const double log_ratio = std::log((double)n_large) / std::log((double)n_small);
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
