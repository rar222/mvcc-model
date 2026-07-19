// Dependency-free test harness. No network in some environments, so no gtest
// or Catch2 -- just a macro and a registry, shared across tests/test_*.cpp.
//
// Run:  ctest --preset default --output-on-failure
//   or: ./build/default/model_tests

#include "test_harness.h"

std::vector<TestCase>& registry() {
    static std::vector<TestCase> r;
    return r;
}

int g_failures = 0;
const char* g_current = "";

Registrar::Registrar(const char* name, std::function<void()> fn) {
    registry().push_back({name, std::move(fn)});
}

int main(int argc, char** argv) {
    const char* filter = argc > 1 ? argv[1] : nullptr;
    int run = 0;

    for (const auto& t : registry()) {
        if (filter && std::string(t.name).find(filter) == std::string::npos) continue;
        g_current = t.name;
        const int before = g_failures;
        std::printf("[ RUN  ] %s\n", t.name);
        t.fn();
        ++run;
        std::printf("[%s] %s\n", g_failures == before ? "  OK  " : " FAIL ", t.name);
    }

    std::printf("\n%d test(s) run, %d check(s) failed\n", run, g_failures);
    return g_failures == 0 ? 0 : 1;
}
