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
    std::function<void()> fn;
};

std::vector<TestCase>& registry();
extern int g_failures;
extern const char* g_current;

struct Registrar {
    Registrar(const char* name, std::function<void()> fn);
};

#define TEST(name)                            \
    static void name();                       \
    static Registrar reg_##name(#name, name); \
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
