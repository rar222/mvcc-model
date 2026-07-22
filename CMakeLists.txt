cmake_minimum_required(VERSION 3.20)
project(mvcc_model LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)  # clangd / IntelliSense

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE RelWithDebInfo CACHE STRING "" FORCE)
endif()

# --- options ----------------------------------------------------------------

option(MODEL_ASAN "Build with AddressSanitizer + UBSan" OFF)
option(MODEL_TSAN "Build with ThreadSanitizer" OFF)

if(MODEL_ASAN AND MODEL_TSAN)
  message(FATAL_ERROR "ASan and TSan cannot be enabled at the same time")
endif()

add_library(model_flags INTERFACE)

# Warnings are always errors, in every preset -- there is no "warm-up" mode
# where they're merely advisory. Fix the code, not the flag.
if(MSVC)
  target_compile_options(model_flags INTERFACE /W4 /permissive- /WX)
else()
  target_compile_options(model_flags INTERFACE
    -Wall
    -Wextra
    -Wpedantic
    -Werror
    -Wshadow
    -Wnon-virtual-dtor
    -Woverloaded-virtual
    -Wcast-align
    -Wcast-qual
    -Wunused
    -Wnull-dereference
    -Wdouble-promotion
    -Wformat=2
    -Wimplicit-fallthrough
    -Wold-style-cast
    -Wuseless-cast
    -Wredundant-decls
    -Wextra-semi
  )
  if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    target_compile_options(model_flags INTERFACE
      -Wduplicated-cond
      -Wduplicated-branches
      -Wlogical-op
      -Wmisleading-indentation
    )
  endif()
  # No C++ exceptions anywhere in this project (see CLAUDE.md): errors are
  # reported through CommitResult / return values, and anything unrecoverable
  # (e.g. bad_alloc out of the standard library) terminates loudly instead of
  # unwinding. The flag makes an accidental `throw` or `try` a compile error.
  target_compile_options(model_flags INTERFACE -fno-exceptions)
  if(MODEL_ASAN)
    target_compile_options(model_flags INTERFACE -fsanitize=address,undefined -fno-omit-frame-pointer -g)
    target_link_options(model_flags INTERFACE -fsanitize=address,undefined)
  endif()
  if(MODEL_TSAN)
    target_compile_options(model_flags INTERFACE -fsanitize=thread -fno-omit-frame-pointer -g)
    target_link_options(model_flags INTERFACE -fsanitize=thread)
  endif()
endif()

# --- library ----------------------------------------------------------------

add_library(model STATIC src/model.cpp)
target_include_directories(model PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(model PUBLIC model_flags)

find_package(Threads REQUIRED)
target_link_libraries(model PUBLIC Threads::Threads)

# model.h is ~3000 lines of heavy templates (parsing it dominates compile
# time -- see `ftime-report`), and nearly every test/example TU includes it.
# Precompile it once here; every other target below reuses this same .gch
# (REUSE_FROM) instead of paying to reparse and reinstantiate it per TU.
target_precompile_headers(model PUBLIC include/model/model.h)

# --- demo -------------------------------------------------------------------

add_executable(demo examples/demo.cpp)
target_link_libraries(demo PRIVATE model)
target_precompile_headers(demo REUSE_FROM model)

# Two writers racing ~10-change transactions on a small shared pool (so
# conflicts are common), one subscriber printing every object it receives.
add_executable(two_writer_demo examples/two_writer_demo.cpp)
target_link_libraries(two_writer_demo PRIVATE model)
target_precompile_headers(two_writer_demo REUSE_FROM model)

# Why View holds its Snapshot by pointer and not by value.
add_executable(bench examples/bench.cpp)
target_link_libraries(bench PRIVATE model)
target_precompile_headers(bench REUSE_FROM model)

# try_commit latency vs model size, and vs concurrent writer-thread count --
# shows apply is still change-proportional (not size-proportional), and that
# commit_mu_ serializes apply (so throughput does not scale linearly with
# writer threads -- that's the documented scope boundary, not a bug).
add_executable(commit_bench examples/commit_bench.cpp)
target_link_libraries(commit_bench PRIVATE model)
target_precompile_headers(commit_bench REUSE_FROM model)

# find_referrers (O(#items) scan) vs find_cached_referrers (O(log n +
# matches), Root::by_cached_reference) -- read speedup vs. the per-write
# cost that pays for it. See DESIGN.md's "Lookup families" section.
add_executable(cached_reference_bench examples/cached_reference_bench.cpp)
target_link_libraries(cached_reference_bench PRIVATE model)
target_precompile_headers(cached_reference_bench REUSE_FROM model)

# 1,000,000 creates as one single-threaded Transaction vs. as 4 threads each
# building+committing their own 250,000-create Transaction -- the single
# transaction wins, because apply is serialized behind commit_mu_ (invariant
# 7) and build (the genuinely parallel part) is cheap next to apply. See the
# file's own header comment for the measured numbers and the full reasoning.
add_executable(single_vs_multi_txn_bench examples/single_vs_multi_txn_bench.cpp)
target_link_libraries(single_vs_multi_txn_bench PRIVATE model)
target_precompile_headers(single_vs_multi_txn_bench REUSE_FROM model)

# --- tests ------------------------------------------------------------------

enable_testing()
add_executable(model_tests
    tests/test_harness.cpp
    tests/test_helpers.cpp
    tests/test_basics.cpp
    tests/test_events.cpp
    tests/test_cascade.cpp
    tests/test_lookup.cpp
    tests/test_views.cpp
    tests/test_transactions.cpp
    tests/test_hooks.cpp
    tests/test_commit.cpp
    tests/test_bulk_load.cpp
    tests/test_concurrency_stress.cpp
    tests/test_undo.cpp
)
target_link_libraries(model_tests PRIVATE model)
target_precompile_headers(model_tests REUSE_FROM model)

add_test(NAME model_tests COMMAND model_tests)
set_tests_properties(model_tests PROPERTIES TIMEOUT 300)

# The persistent index (HAMT) is differential-tested against std::unordered_map.
# Shares tests/test_harness.cpp/.h with model_tests (TEST()/CHECK()/g_failures/
# main()) -- test_harness.h has no model.h dependency, so this stays independent
# of libmodel (model_flags only: compiler flags, not the model library itself).
add_executable(persistent_map_tests tests/persistent_map_tests.cpp tests/test_harness.cpp)
target_link_libraries(persistent_map_tests PRIVATE model_flags)
target_include_directories(persistent_map_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
add_test(NAME persistent_map_tests COMMAND persistent_map_tests)
set_tests_properties(persistent_map_tests PROPERTIES TIMEOUT 120)

# The demo doubles as an integration test: its internal asserts fire if a Ref
# ever dangles or a conflict is silently missed, so run it under CTest too
# (needs a Debug-ish build for asserts).
add_test(NAME demo_integration COMMAND demo)
set_tests_properties(demo_integration PROPERTIES TIMEOUT 300)

add_test(NAME two_writer_demo_integration COMMAND two_writer_demo)
set_tests_properties(two_writer_demo_integration PROPERTIES TIMEOUT 300)

# This project's own documented complexity/memory claims (CLAUDE.md,
# DESIGN.md, README.md), asserted across several sizes instead of only
# ever printed for a human to eyeball -- see the file's own header comment.
add_executable(performance_tests tests/performance_tests.cpp)
target_link_libraries(performance_tests PRIVATE model)
target_precompile_headers(performance_tests REUSE_FROM model)
add_test(NAME performance_tests COMMAND performance_tests)
set_tests_properties(performance_tests PROPERTIES TIMEOUT 300)
