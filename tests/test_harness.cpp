// Dependency-free test harness. No network in some environments, so no gtest
// or Catch2 -- just a macro and a registry, shared across tests/test_*.cpp.
//
// Run:  ctest --preset default --output-on-failure
//   or: ./build/default/model_tests

#include "test_harness.h"

#include <chrono>
#include <csignal>
#include <sys/wait.h>
#include <thread>
#include <unistd.h>

std::vector<TestCase>& registry() {
    static std::vector<TestCase> r;
    return r;
}

int g_failures = 0;
const char* g_current = "";

bool dies_of_assert(const std::function<void()>& fn) {
    // Flush before forking, not just skip re-flushing in the child: under a
    // piped/redirected stdout (exactly how ctest runs this), the parent's
    // buffered test output would otherwise get re-dumped by the child's exit
    // -- same reasoning as performance_tests.cpp's fork()-based helper.
    std::fflush(nullptr);
    const pid_t pid = fork();
    if (pid == 0) {
        fn();      // reached only if fn() doesn't hit the assert under test
        _exit(0);  // not exit(): skip static destructors racing the parent's stdio
    }
    // Bounded wait, not a blocking waitpid(): a genuine assert failure aborts
    // in milliseconds. If fn() hangs instead -- e.g. a real deadlock, which
    // is exactly the kind of failure this helper exists to surface cleanly
    // rather than let corrupt or freeze a live process (see model.cpp's
    // reap_waiters_/reaper_stop_ asserts: this exact scenario, a still-
    // registered wait_for_reclamation() caller racing ~Model(), hung for
    // real during development because only ONE of the two asserts that
    // TOGETHER close that race existed at the time) -- don't let that hang
    // propagate into an indefinite CI hang. Poll with a generous deadline,
    // SIGKILL the child if it's blown past it, and report "didn't die of an
    // assert": a normal, fast, unambiguous test FAILURE instead of a stuck
    // process someone has to notice and kill by hand.
    // 3s: a genuine assert failure aborts in single-digit milliseconds, and
    // some callers (e.g. model_destructor_asserts_if_a_wait_for_reclamation_
    // caller_is_still_in_flight's retry loop) call this up to ~30 times in a
    // row -- the per-call deadline has to stay small enough that even a
    // total wipeout (every attempt legitimately hangs) finishes well inside
    // model_tests' own default 300s ctest TIMEOUT, not brush up against it.
    using namespace std::chrono;
    const auto deadline = steady_clock::now() + seconds(3);
    int status = 0;
    for (;;) {
        const pid_t r = waitpid(pid, &status, WNOHANG);
        if (r == pid) return WIFSIGNALED(status) && WTERMSIG(status) == SIGABRT;
        if (steady_clock::now() >= deadline) {
            kill(pid, SIGKILL);
            waitpid(pid, &status, 0);  // reap it -- SIGKILL always succeeds, this never blocks long
            return false;
        }
        std::this_thread::sleep_for(milliseconds(5));
    }
}

bool completes_cleanly(const std::function<void()>& fn) {
    std::fflush(nullptr);  // same reasoning as dies_of_assert(): don't let the child re-dump
                            // the parent's buffered, piped-under-ctest stdout on exit
    const pid_t pid = fork();
    if (pid == 0) {
        fn();
        _exit(0);
    }
    // Same bounded-wait/SIGKILL contract as dies_of_assert(): a real hang here
    // is exactly the failure mode this helper exists to report as a normal,
    // fast test FAILURE instead of a stuck process.
    using namespace std::chrono;
    const auto deadline = steady_clock::now() + seconds(3);
    int status = 0;
    for (;;) {
        const pid_t r = waitpid(pid, &status, WNOHANG);
        if (r == pid) return WIFEXITED(status) && WEXITSTATUS(status) == 0;
        if (steady_clock::now() >= deadline) {
            kill(pid, SIGKILL);
            waitpid(pid, &status, 0);
            return false;
        }
        std::this_thread::sleep_for(milliseconds(5));
    }
}

Registrar::Registrar(const char* name, const char* filename, int linenum, std::function<void()> fn) {
    registry().push_back({name, filename, linenum, std::move(fn)});
}

int main(int argc, char** argv) {
    // --list: print each registered test name, one per line, and exit. Used
    // by CMake's mvcc_discover_tests() (cmake/DiscoverTests.cmake) to turn
    // every TEST() case into its own CTest entry, so IDE test explorers (and
    // `ctest -R`) see them individually instead of one opaque binary.
    if (argc > 1 && std::string(argv[1]) == "--list") {
        for (const auto& t : registry()) std::printf("%s|%s:%d\n", t.name, t.filename, t.linenum);
        return 0;
    }

    // --exact <name>: run precisely the named test, not a substring match.
    // This is what the discovered per-test CTest entries invoke, so a test
    // named e.g. "cascade" can't accidentally also run "cascade_delete_...".
    const bool exact = argc > 2 && std::string(argv[1]) == "--exact";
    const char* filter = exact ? argv[2] : (argc > 1 ? argv[1] : nullptr);
    int run = 0;

    for (const auto& t : registry()) {
        if (filter) {
            const bool match = exact ? std::string(t.name) == filter
                                      : std::string(t.name).find(filter) != std::string::npos;
            if (!match) continue;
        }
        g_current = t.name;
        const int before = g_failures;
        std::printf("[ RUN  ] %s\n", t.name);
        const auto t0 = std::chrono::steady_clock::now();
        t.fn();
        const auto elapsed_ms =
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        ++run;
        std::printf("[%s] %s (%.2f ms)\n", g_failures == before ? "  OK  " : " FAIL ", t.name,
                    elapsed_ms);
    }

    std::printf("\n%d test(s) run, %d check(s) failed\n", run, g_failures);
    return g_failures == 0 ? 0 : 1;
}
