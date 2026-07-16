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
option(MODEL_WERROR "Treat warnings as errors" OFF)

if(MODEL_ASAN AND MODEL_TSAN)
  message(FATAL_ERROR "ASan and TSan cannot be enabled at the same time")
endif()

add_library(model_flags INTERFACE)

if(MSVC)
  target_compile_options(model_flags INTERFACE /W4 /permissive-)
  if(MODEL_WERROR)
    target_compile_options(model_flags INTERFACE /WX)
  endif()
else()
  target_compile_options(model_flags INTERFACE -Wall -Wextra -Wpedantic)
  # No C++ exceptions anywhere in this project (see CLAUDE.md): errors are
  # reported through CommitResult / return values, and anything unrecoverable
  # (e.g. bad_alloc out of the standard library) terminates loudly instead of
  # unwinding. The flag makes an accidental `throw` or `try` a compile error.
  target_compile_options(model_flags INTERFACE -fno-exceptions)
  if(MODEL_WERROR)
    target_compile_options(model_flags INTERFACE -Werror)
  endif()
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

# --- demo -------------------------------------------------------------------

add_executable(demo examples/demo.cpp)
target_link_libraries(demo PRIVATE model)

# Two writers racing ~10-change transactions on a small shared pool (so
# conflicts are common), one subscriber printing every object it receives.
add_executable(two_writer_demo examples/two_writer_demo.cpp)
target_link_libraries(two_writer_demo PRIVATE model)

# Why View holds its Snapshot by pointer and not by value.
add_executable(bench examples/bench.cpp)
target_link_libraries(bench PRIVATE model)

# try_commit latency vs model size, and vs concurrent writer-thread count --
# shows apply is still change-proportional (not size-proportional), and that
# commit_mu_ serializes apply (so throughput does not scale linearly with
# writer threads -- that's the documented scope boundary, not a bug).
add_executable(commit_bench examples/commit_bench.cpp)
target_link_libraries(commit_bench PRIVATE model)

# --- tests ------------------------------------------------------------------

enable_testing()
add_executable(model_tests tests/tests.cpp)
target_link_libraries(model_tests PRIVATE model)

add_test(NAME model_tests COMMAND model_tests)
set_tests_properties(model_tests PROPERTIES TIMEOUT 300)

# The persistent index (HAMT) is differential-tested against std::unordered_map.
add_executable(persistent_map_tests tests/persistent_map_tests.cpp)
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
