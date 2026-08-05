#pragma once
//
// Shared test-registration harness for the split tests/test_*.cpp files.
//
// registry()/g_failures/g_current/Registrar are declared here and defined
// ONCE in test_harness.cpp -- not in an anonymous namespace -- so that a
// Registrar constructed by a TEST() in ANY test_*.cpp registers into the
// SAME vector that main() (also in test_harness.cpp) iterates. An
// anonymous-namespace registry(), the way this lived when everything was
// one TU, would give each TU its own private copy invisible to main(),
// and every test outside that one TU would silently never run.

#include <cstdio>
#include <functional>
#include <string>
#include <vector>

struct TestCase {
    const char* name;
    const char* filename;
    int linenum;
    std::function<void()> fn;
};

std::vector<TestCase>& registry();
extern int g_failures;
extern const char* g_current;

struct Registrar {
    Registrar(const char* name, const char* filename, int linenum, std::function<void()> fn);
};

#define TEST(name)                            \
    static void name();                       \
    static Registrar reg_##name(#name, __FILE__, __LINE__, name); \
    static void name()

#define CHECK(cond)                                                       \
    do {                                                                  \
        if (!(cond)) {                                                    \
            std::printf("  FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
            ++g_failures;                                                 \
        }                                                                 \
    } while (0)

#define CHECK_EQ(a, b)                                                           \
    do {                                                                         \
        auto lhs__ = (a);                                                        \
        auto rhs__ = (b);                                                        \
        if (!(lhs__ == rhs__)) {                                                 \
            std::printf("  FAIL %s:%d: %s == %s\n", __FILE__, __LINE__, #a, #b); \
            ++g_failures;                                                        \
        }                                                                        \
    } while (0)

/// Runs `fn` in a forked child (same fork() technique performance_tests.cpp
/// uses for out-of-process measurement) and reports whether the child died
/// from a failed `assert` -- for exercising an invariant this project
/// enforces via a runtime assert rather than a returned error code, without
/// aborting the actual test binary. True = the child was killed by SIGABRT
/// (the assert fired); false = anything else, including fn() returning
/// normally (the assert did NOT fire). Only meaningful in a build where
/// `assert` is live -- every preset in this project's CMakeLists keeps it on
/// (see CLAUDE.md), so that's every configuration these tests run under.
/// Defined once in test_harness.cpp; CHECK_ASSERT_FAILURE below is the
/// macro every test actually uses, matching CHECK/CHECK_EQ's shape.
bool dies_of_assert(const std::function<void()>& fn);

#define CHECK_ASSERT_FAILURE(expr)                                                        \
    do {                                                                                   \
        if (!dies_of_assert([&] { expr; })) {                                              \
            std::printf("  FAIL %s:%d: expected an assert failure from: %s\n", __FILE__,   \
                        __LINE__, #expr);                                                  \
            ++g_failures;                                                                  \
        }                                                                                   \
    } while (0)
