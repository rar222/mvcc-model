// Dependency-free (of model.h) exerciser for the flat hash set in flat_set.h.
// Same TEST()/CHECK() harness as persistent_map_tests.cpp; every random run
// is differential against std::unordered_set and reports the diverging key.
//
// Run:  ctest --preset default -R flat_set_tests --output-on-failure
//   or: ./build/default/flat_set_tests

#include <cinttypes>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <random>
#include <unordered_set>
#include <utility>
#include <vector>

#include "model/flat_set.h"
#include "test_harness.h"
using namespace model::fset;

namespace {

// No scrambling, so sequential keys reach the set's own scrambler as-is.
struct IdentityHash {
    std::uint64_t operator()(std::uint64_t k) const noexcept { return k; }
};

// Three distinct hash values for any number of keys: long probe runs, and
// the backward-shift path on every erase.
struct ClashHash {
    std::uint64_t operator()(std::uint64_t k) const noexcept { return k % 3; }
};

// Shaped like model::Id: null is the value-initialized key, and two keys
// sharing an index but not a generation are different keys.
struct GenKey {
    std::uint32_t index = 0;
    std::uint32_t gen = 0;
    bool operator==(const GenKey&) const = default;
};
struct GenKeyHash {
    std::uint64_t operator()(const GenKey& k) const noexcept {
        return (static_cast<std::uint64_t>(k.gen) << 32) | k.index;
    }
};

using U64Set = FlatSet<std::uint64_t, IdentityHash>;

void fail(const char* what, const char* detail, std::uint64_t key) {
    std::printf("  FAIL %s: %s (key %" PRIu64 ")\n", what, detail, key);
    ++g_failures;
}

template <class Set>
bool same_as(const Set& s, const std::unordered_set<std::uint64_t>& ref) {
    if (s.size() != ref.size()) return false;
    // Every reference key must be found by contains().
    for (const std::uint64_t k : ref)
        if (!s.contains(k)) return false;
    std::size_t visited = 0;
    bool all_members = true;
    // for_each() must visit exactly the reference keys.
    s.for_each([&](const std::uint64_t& k) {
        ++visited;
        if (ref.count(k) == 0) all_members = false;
    });
    return all_members && visited == ref.size();
}

template <class Hash>
void differential_churn(const char* what, std::uint64_t seed, std::uint64_t domain,
                        std::size_t ops, unsigned insert_pct, std::size_t check_every) {
    std::mt19937_64 rng(seed);
    FlatSet<std::uint64_t, Hash> s;
    std::unordered_set<std::uint64_t> ref;
    // Applies one random insert or erase to both sets and compares the results.
    for (std::size_t op = 0; op < ops; ++op) {
        const std::uint64_t k = 1 + rng() % domain;
        // Insert with probability insert_pct, else erase.
        if (rng() % 100 < insert_pct) {
            if (s.insert(k) != ref.insert(k).second) return fail(what, "insert result differs", k);
        } else {
            if (s.erase(k) != (ref.erase(k) == 1)) return fail(what, "erase result differs", k);
        }
        if (s.contains(k) != (ref.count(k) == 1)) return fail(what, "contains differs after op", k);
        if (op % check_every == 0 && !same_as(s, ref)) return fail(what, "contents diverged", k);
    }
    if (!same_as(s, ref)) fail(what, "contents diverged at end", 0);
}

}  // namespace

TEST(empty_set_reports_empty_and_finds_nothing) {
    U64Set s;
    CHECK(s.empty());
    CHECK_EQ(s.size(), 0u);
    CHECK_EQ(s.cell_count(), 0u);
    CHECK(!s.contains(1));
    CHECK(!s.erase(1));
    std::size_t visited = 0;
    s.for_each([&](const std::uint64_t&) { ++visited; });
    CHECK_EQ(visited, 0u);
}

TEST(insert_reports_whether_the_key_was_new) {
    U64Set s;
    CHECK(s.insert(5));
    CHECK(!s.insert(5));
    CHECK_EQ(s.size(), 1u);
    CHECK(s.contains(5));
}

TEST(erase_reports_whether_the_key_was_present) {
    U64Set s;
    CHECK(s.insert(5));
    CHECK(s.erase(5));
    CHECK(!s.erase(5));
    CHECK(!s.contains(5));
    CHECK(s.empty());
}

TEST(erase_then_reinsert_the_same_key_works) {
    U64Set s;
    // Cycles one key in and out.
    for (int round = 0; round < 50; ++round) {
        CHECK(s.insert(9));
        CHECK(s.contains(9));
        CHECK(s.erase(9));
        CHECK(!s.contains(9));
    }
    CHECK(s.empty());
}

TEST(the_empty_key_is_never_a_member) {
    U64Set s;
    CHECK(s.insert(1));
    CHECK(!s.contains(0));
    CHECK(!s.erase(0));
    CHECK_EQ(s.size(), 1u);
    CHECK_ASSERT_FAILURE(s.insert(0));
}

TEST(table_grows_and_keeps_every_key_below_three_quarters_full) {
    U64Set s;
    const std::uint64_t n = 10000;
    // Checks the table shape after every insert.
    for (std::uint64_t k = 1; k <= n; ++k) {
        CHECK(s.insert(k));
        const std::size_t cells = s.cell_count();
        // Fails on a cell count that is not a power of two or a table over 3/4 full.
        if ((cells & (cells - 1)) != 0 || s.size() * 4 > cells * 3) {
            fail("table_grows", "cell count not a power of two or over 3/4 full", k);
            return;
        }
    }
    CHECK_EQ(s.size(), std::size_t{n});
    // Every key survives every rehash.
    for (std::uint64_t k = 1; k <= n; ++k)
        if (!s.contains(k)) return fail("table_grows", "key lost across growth", k);
    CHECK(!s.contains(n + 1));
}

TEST(erase_keeps_displaced_keys_reachable_under_total_collision) {
    FlatSet<std::uint64_t, ClashHash> s;
    const std::uint64_t n = 300;
    for (std::uint64_t k = 1; k <= n; ++k) CHECK(s.insert(k));

    for (std::uint64_t k = 3; k <= n; k += 3) CHECK(s.erase(k));
    // Exactly the multiples of 3 are gone; every other key is still reachable.
    for (std::uint64_t k = 1; k <= n; ++k)
        if (s.contains(k) != (k % 3 != 0))
            return fail("displaced_keys", "wrong membership after ascending erase", k);

    // Erases the survivors from the far end, shifting the runs the other way.
    for (std::uint64_t k = n; k >= 1; --k)
        if (k % 3 != 0 && !s.erase(k)) return fail("displaced_keys", "erase missed a live key", k);
    CHECK(s.empty());
}

TEST(random_churn_over_a_large_domain_matches_unordered_set) {
    // One run per seed.
    for (std::uint64_t seed = 1; seed <= 4; ++seed)
        differential_churn<IdentityHash>("large_domain", seed, 100000, 200000, 60, 997);
}

TEST(random_churn_over_a_tiny_domain_exercises_wraparound_and_small_tables) {
    // One run per domain size, from 3 keys to 24.
    for (std::uint64_t domain = 3; domain <= 24; ++domain)
        differential_churn<IdentityHash>("tiny_domain", domain, domain, 20000, 50, 1);
}

TEST(random_churn_with_colliding_hashes_matches_unordered_set) {
    // One run per seed.
    for (std::uint64_t seed = 1; seed <= 4; ++seed)
        differential_churn<ClashHash>("clash_hash", seed, 200, 50000, 50, 13);
}

TEST(sequential_key_churn_matches_unordered_set) {
    std::mt19937_64 rng(7);
    U64Set s;
    std::unordered_set<std::uint64_t> ref;
    std::vector<std::uint64_t> live;
    std::uint64_t next = 1;
    // Inserts fresh increasing keys and erases random live ones.
    for (int op = 0; op < 100000; ++op) {
        // Inserts the next key.
        if (live.empty() || rng() % 100 < 55) {
            live.push_back(next);
            if (!s.insert(next) || !ref.insert(next).second)
                return fail("sequential_churn", "fresh key rejected", next);
            ++next;
        } else {
            // Erases a random live key.
            const std::size_t at = rng() % live.size();
            const std::uint64_t k = live[at];
            live[at] = live.back();
            live.pop_back();
            if (!s.erase(k) || ref.erase(k) != 1) return fail("sequential_churn", "live key not erased", k);
        }
    }
    CHECK(same_as(s, ref));
}

TEST(for_each_visits_every_key_exactly_once) {
    U64Set s;
    for (std::uint64_t k = 1; k <= 500; ++k) s.insert(k);
    for (std::uint64_t k = 2; k <= 500; k += 2) s.erase(k);
    std::unordered_set<std::uint64_t> seen;
    std::size_t visits = 0;
    // Counts visits and collects the distinct keys seen.
    s.for_each([&](const std::uint64_t& k) {
        ++visits;
        seen.insert(k);
    });
    CHECK_EQ(visits, s.size());
    CHECK_EQ(seen.size(), s.size());
    for (const std::uint64_t k : seen) CHECK(k % 2 == 1);
}

TEST(for_each_short_circuit_stops_early_and_reports_it) {
    U64Set s;
    for (std::uint64_t k = 1; k <= 100; ++k) s.insert(k);

    std::size_t visits = 0;
    const bool completed = s.for_each_short_circuit([&](const std::uint64_t&) { return ++visits < 10; });
    CHECK(!completed);
    CHECK_EQ(visits, 10u);

    visits = 0;
    // A visitor that never stops sees every key.
    CHECK(s.for_each_short_circuit([&](const std::uint64_t&) {
        ++visits;
        return true;
    }));
    CHECK_EQ(visits, 100u);
}

TEST(copy_is_independent_of_the_original) {
    U64Set a;
    for (std::uint64_t k = 1; k <= 100; ++k) a.insert(k);
    U64Set b = a;
    a.erase(50);
    a.insert(1000);
    CHECK(b.contains(50));
    CHECK(!b.contains(1000));
    CHECK_EQ(b.size(), 100u);
    CHECK_EQ(a.size(), 100u);

    U64Set c;
    c.insert(5000);
    c = b;
    CHECK(!c.contains(5000));
    CHECK_EQ(c.size(), 100u);
}

TEST(moved_from_set_is_empty_and_reusable) {
    U64Set a;
    for (std::uint64_t k = 1; k <= 100; ++k) a.insert(k);

    U64Set b = std::move(a);
    CHECK_EQ(b.size(), 100u);
    CHECK(a.empty());
    CHECK(!a.contains(1));
    CHECK(!a.erase(1));
    CHECK(a.insert(1));
    CHECK(a.contains(1));

    U64Set c;
    c.insert(2000);
    c = std::move(b);
    CHECK_EQ(c.size(), 100u);
    CHECK(!c.contains(2000));
    CHECK(b.empty());
    CHECK(b.insert(3));
}

TEST(reserve_preallocates_and_never_shrinks) {
    U64Set s;
    CHECK(s.insert(7));
    s.reserve(1000);
    CHECK_EQ(s.cell_count(), 2048u);
    CHECK(s.contains(7));
    for (std::uint64_t k = 100; k < 1100; ++k) s.insert(k);
    CHECK_EQ(s.cell_count(), 2048u);
    s.reserve(10);
    CHECK_EQ(s.cell_count(), 2048u);
}

TEST(shrink_to_fit_returns_space_after_mass_erase) {
    U64Set s;
    for (std::uint64_t k = 1; k <= 10000; ++k) s.insert(k);
    for (std::uint64_t k = 11; k <= 10000; ++k) s.erase(k);
    CHECK(s.cell_count() >= 10000);

    s.shrink_to_fit();
    CHECK_EQ(s.cell_count(), 16u);
    for (std::uint64_t k = 1; k <= 10; ++k) CHECK(s.contains(k));
    CHECK_EQ(s.size(), 10u);

    for (std::uint64_t k = 1; k <= 10; ++k) s.erase(k);
    s.shrink_to_fit();
    CHECK_EQ(s.cell_count(), 0u);
    CHECK(s.insert(42));
    CHECK(s.contains(42));
}

TEST(struct_keys_distinguish_generations_of_the_same_index) {
    FlatSet<GenKey, GenKeyHash> s;
    CHECK(s.insert(GenKey{5, 1}));
    CHECK(s.contains(GenKey{5, 1}));
    CHECK(!s.contains(GenKey{5, 2}));
    CHECK(!s.contains(GenKey{6, 1}));
    CHECK(!s.contains(GenKey{}));

    CHECK(s.erase(GenKey{5, 1}));
    CHECK(s.insert(GenKey{5, 2}));
    CHECK(!s.contains(GenKey{5, 1}));
    CHECK(s.contains(GenKey{5, 2}));
    CHECK_ASSERT_FAILURE(s.insert(GenKey{}));
}
