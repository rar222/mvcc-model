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
    -Wdouble-promotion
    -Wformat=2
    -Wimplicit-fallthrough
    -Wold-style-cast
    -Wuseless-cast
    -Wredundant-decls
    -Wextra-semi
  )
  # -Wnull-dereference deliberately omitted: at -O2 (the default preset) it
  # false-positives inside libstdc++'s own inlined basic_string copy ctor,
  # not just this project's code -- a known GCC false-positive pattern with
  # heavy inlining, not a real bug to chase here.
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

# range_by_cached_field/range_cached_referrers (range-based for) vs.
# for_each_by_cached_field/for_each_cached_referrers (callback, never stops
# early) vs. all_of_by_cached_field/all_of_cached_referrers stopping after
# the first match vs. visiting every match -- across a population sweep.
add_executable(range_vs_callback_bench examples/range_vs_callback_bench.cpp)
target_link_libraries(range_vs_callback_bench PRIVATE model)
target_precompile_headers(range_vs_callback_bench REUSE_FROM model)

# 1,000,000 creates as one single-threaded Transaction vs. as 4 threads each
# building+committing their own 250,000-create Transaction -- the single
# transaction wins, because apply is serialized behind commit_mu_ (invariant
# 7) and build (the genuinely parallel part) is cheap next to apply. See the
# file's own header comment for the measured numbers and the full reasoning.
add_executable(single_vs_multi_txn_bench examples/single_vs_multi_txn_bench.cpp)
target_link_libraries(single_vs_multi_txn_bench PRIVATE model)
target_precompile_headers(single_vs_multi_txn_bench REUSE_FROM model)

# Isolates apply_transaction_contents()'s per-object index-maintenance cost
# (by_type_/by_field_/by_cached_field_/by_cached_reference_/referrers_) for
# one huge bulk create, one huge bulk update, and one transaction with many
# separate cascading remove() intents -- times only the try_commit() call.
add_executable(large_txn_bench examples/large_txn_bench.cpp)
target_link_libraries(large_txn_bench PRIVATE model)
target_precompile_headers(large_txn_bench REUSE_FROM model)

# Per-commit latency of many SMALL (one-object) transactions -- the
# target-scale workload per CLAUDE.md -- to check whether apply_transaction_
# contents()'s large-transaction optimizations (large_txn_bench.cpp) cost
# anything when there's nothing to dedupe (N=1 object per commit).
add_executable(small_txn_bench examples/small_txn_bench.cpp)
target_link_libraries(small_txn_bench PRIVATE model)
target_precompile_headers(small_txn_bench REUSE_FROM model)

# Stand-alone copy of performance_tests.cpp's one_million_random_basic_
# records_in_a_single_transaction_without_undo, isolated into its own binary
# so it can be profiled (gprof/callgrind) without the rest of the
# performance suite running alongside it. BasicRecord declares no keys and
# no references, so this is close to the model's own per-object floor.
add_executable(basic_record_bench examples/basic_record_bench.cpp)
target_link_libraries(basic_record_bench PRIVATE model)
target_precompile_headers(basic_record_bench REUSE_FROM model)
# CALLGRIND_{ZERO_STATS,START/STOP_INSTRUMENTATION,DUMP_STATS} bracket just the
# try_commit_without_undo() call so `callgrind --instr-atstart=no` profiles
# only the commit phase, not the random-data build loop before it. These are
# inline no-op asm hints outside of valgrind, so defining this unconditionally
# costs nothing on a normal run.
target_compile_definitions(basic_record_bench PRIVATE MODEL_BENCH_CALLGRIND)

# Same as basic_record_bench, except the commit call is try_commit() (the
# undo-tracking path) instead of try_commit_without_undo() -- isolates what
# keeping an undo entry costs on top of the same per-object apply work.
add_executable(basic_record_bench_with_undo examples/basic_record_bench_with_undo.cpp)
target_link_libraries(basic_record_bench_with_undo PRIVATE model)
target_precompile_headers(basic_record_bench_with_undo REUSE_FROM model)
target_compile_definitions(basic_record_bench_with_undo PRIVATE MODEL_BENCH_CALLGRIND)

# Node::slots' vector capacity() vs size() at steady state, single-threaded
# vs. under concurrent snapshot() pressure vs. one bulk transaction -- backed
# by Model::slot_stats_diagnostics(). See the file's own header comment.
add_executable(slot_stats_probe examples/slot_stats_probe.cpp)
target_link_libraries(slot_stats_probe PRIVATE model)
target_precompile_headers(slot_stats_probe REUSE_FROM model)

# Same shape as large_txn_bench.cpp's bench_bulk_create, isolated into its
# own binary (build/commit phase split, argv count override) so it can be
# profiled the same way as basic_record_bench -- Order touches every index
# (by_type_, by_field_, by_cached_field_, by_cached_reference_, referrers_),
# the natural next question after basic_record_bench's by_type_-only finding.
add_executable(order_record_bench examples/order_record_bench.cpp)
target_link_libraries(order_record_bench PRIVATE model)
target_precompile_headers(order_record_bench REUSE_FROM model)

# Demonstrates -- and measures -- explicit `extern template` instantiation
# as a compile-time lever: a generated example/types_extern.h declares
# every Snapshot/Transaction/Model/BulkTransaction/View entry point
# reachable for Account/Order `extern`, and a generated types_extern.cpp is
# the one place that actually instantiates them. See
# scripts/gen_extern_templates.py's own docstring for why these two files
# are generated (from example/types.h, at configure/build time, never
# committed) instead of hand-maintained -- in short: full coverage instead
# of a hand-curated subset, so a field added to types.h can never silently
# fall back to implicit instantiation because someone forgot to update a
# checked-in list. See extern_template_demo.cpp's comment for measured
# compile-time numbers and when this technique is (and isn't) worth it.
# Two source files, not REUSE_FROM model's PCH split further: the
# instantiation TU needs to be its own translation unit for the `extern
# template` declarations to have anything to link against.
find_package(Python3 REQUIRED COMPONENTS Interpreter)

set(EXTERN_TEMPLATE_GEN ${CMAKE_CURRENT_SOURCE_DIR}/scripts/gen_extern_templates.py)
set(EXTERN_TEMPLATE_SRC ${CMAKE_CURRENT_SOURCE_DIR}/include/example/types.h)
set(EXTERN_TEMPLATE_GEN_DIR ${CMAKE_CURRENT_BINARY_DIR}/generated)
set(EXTERN_TEMPLATE_H ${EXTERN_TEMPLATE_GEN_DIR}/example/types_extern.h)
set(EXTERN_TEMPLATE_CPP ${EXTERN_TEMPLATE_GEN_DIR}/types_extern.cpp)

add_custom_command(
  OUTPUT ${EXTERN_TEMPLATE_H} ${EXTERN_TEMPLATE_CPP}
  COMMAND ${CMAKE_COMMAND} -E make_directory ${EXTERN_TEMPLATE_GEN_DIR}/example
  COMMAND ${Python3_EXECUTABLE} ${EXTERN_TEMPLATE_GEN} ${EXTERN_TEMPLATE_SRC}
          --namespace example --mode header
          --include-path example/types.h --output ${EXTERN_TEMPLATE_H}
  COMMAND ${Python3_EXECUTABLE} ${EXTERN_TEMPLATE_GEN} ${EXTERN_TEMPLATE_SRC}
          --namespace example --mode source
          --include-path example/types_extern.h --output ${EXTERN_TEMPLATE_CPP}
  DEPENDS ${EXTERN_TEMPLATE_SRC} ${EXTERN_TEMPLATE_GEN}
  COMMENT "Generating extern template coverage for example/types.h"
  VERBATIM
)
add_custom_target(generate_types_extern DEPENDS ${EXTERN_TEMPLATE_H} ${EXTERN_TEMPLATE_CPP})

add_executable(extern_template_demo examples/extern_template_demo.cpp ${EXTERN_TEMPLATE_CPP})
add_dependencies(extern_template_demo generate_types_extern)
target_include_directories(extern_template_demo PRIVATE ${EXTERN_TEMPLATE_GEN_DIR})
target_link_libraries(extern_template_demo PRIVATE model)
target_precompile_headers(extern_template_demo REUSE_FROM model)

# --- tests ------------------------------------------------------------------

include(CTest)
include(cmake/DiscoverTests.cmake)

########################
# model_tests
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
    tests/test_inheritance.cpp
    tests/test_assert_invariants.cpp
)
target_link_libraries(model_tests PRIVATE model)
target_precompile_headers(model_tests REUSE_FROM model)
# Registers every TEST() in the sources above as its own CTest entry (see
# cmake/DiscoverTests.cmake) so IDE test explorers and `ctest -R <name>`
# see each case individually, not just one "model_tests" binary.
mvcc_discover_tests(model_tests)

########################
# performance_tests
#
# This project's own documented complexity/memory claims (CLAUDE.md,
# DESIGN.md, README.md), asserted across several sizes instead of only
# ever printed for a human to eyeball -- see the file's own header comment.
add_executable(performance_tests tests/performance_tests.cpp tests/test_harness.cpp)
target_link_libraries(performance_tests PRIVATE model)
target_precompile_headers(performance_tests REUSE_FROM model)
# RUN_SERIAL: these assert on wall-clock timing/scaling, not just
# correctness (see the file's own header comment). Before this file's tests
# were individually discovered, they only ever ran as one serial process;
# splitting them into per-test CTest entries would otherwise let `ctest -j`
# schedule several of THIS binary's own timing-sensitive subtests
# concurrently, competing for the same CPU cores and reintroducing exactly
# the jitter check_scaling()'s bands are designed to tolerate but not
# multiply.
mvcc_discover_tests(performance_tests RUN_SERIAL)

########################
# persistent_map_tests
#
# The persistent index (HAMT) is differential-tested against std::unordered_map.
# Shares tests/test_harness.cpp/.h with model_tests (TEST()/CHECK()/g_failures/
# main()) -- test_harness.h has no model.h dependency, so this stays independent
# of libmodel (model_flags only: compiler flags, not the model library itself).
add_executable(persistent_map_tests tests/persistent_map_tests.cpp tests/test_harness.cpp)
target_link_libraries(persistent_map_tests PRIVATE model_flags)
target_include_directories(persistent_map_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
mvcc_discover_tests(persistent_map_tests TIMEOUT 120)

########################
# stand_alone tests
#
# Added with "stand_alone" prefix for grouping

# The demo doubles as an test: its internal asserts fire if a Ref
# ever dangles or a conflict is silently missed, so run it under CTest too
# (needs a Debug-ish build for asserts).
add_test(NAME "stand_alone.demo" COMMAND demo)
set_tests_properties("stand_alone.demo" PROPERTIES TIMEOUT 300)

add_test(NAME "stand_alone.two_writer_demo_integration" COMMAND two_writer_demo)
set_tests_properties("stand_alone.two_writer_demo_integration" PROPERTIES TIMEOUT 300)

add_test(NAME "stand_alone.extern_template_demo" COMMAND extern_template_demo)
set_tests_properties("stand_alone.extern_template_demo" PROPERTIES TIMEOUT 300)

# Automated form of scripts/gen_extern_templates.py's own "HOW TO CHECK IF
# THIS SCRIPT IS MISSING A SYMBOL" docstring section: fails if
# extern_template_demo.cpp.o had to define its own copy of a model.h entry
# point instead of resolving it from generated/types_extern.cpp.o -- proof,
# from actual compiled output, that a new (or newly-used) template entry
# point in example/types.h/model.h isn't covered by render_entries() yet.
# Only meaningful at -O2 (the default preset): at -O0 (debug/asan/tsan) the
# compiler doesn't inline as aggressively, so the checker's allowlist
# (tuned against this preset's specific inlining behavior -- see the
# checker script's own comment) doesn't hold there.
if(CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo" AND NOT MODEL_ASAN AND NOT MODEL_TSAN)
  add_test(
    NAME "extern_template_demo.gen_extern_templates_has_no_missing_coverage"
    COMMAND ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/scripts/check_extern_template_coverage.py
            --object ${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/extern_template_demo.dir/examples/extern_template_demo.cpp.o
            --nm ${CMAKE_NM}
  )
  set_tests_properties("extern_template_demo.gen_extern_templates_has_no_missing_coverage" PROPERTIES
    TIMEOUT 60)
endif()

