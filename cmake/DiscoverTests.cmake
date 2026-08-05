# mvcc_discover_tests(<target>) -- register each TEST() case in <target> as
# its own CTest test, instead of one opaque `add_test(NAME <target> ...)`.
#
# <target> must be a tests/test_harness.h-based binary: it needs to support
# `--list` (print one test name per line, exit) and `--exact <name>` (run
# only that test). See tests/test_harness.cpp.
#
# Modeled on upstream CMake's GoogleTest.cmake (TEST_INCLUDE_FILES: a script
# ctest include()s at test-run time, after the binary is built, so renaming
# or adding a TEST() doesn't require re-running cmake configure). Trimmed
# down because this project only ever builds single-config (Ninja/Make on
# Linux, per CMakePresets.json) -- no per-config file selection, no
# CROSSCOMPILING_EMULATOR, no TEST_PREFIX/PROPERTIES -- add those back only
# if a real need shows up.
function(mvcc_discover_tests target)
  cmake_parse_arguments(ARG "RUN_SERIAL" "TIMEOUT" "" ${ARGN})
  if(NOT ARG_TIMEOUT)
    set(ARG_TIMEOUT 300)
  endif()
  set(_mvcc_extra_props "")
  if(ARG_RUN_SERIAL)
    set(_mvcc_extra_props "RUN_SERIAL TRUE")
  endif()
  set(ctest_include_file "${CMAKE_CURRENT_BINARY_DIR}/${target}_discover_tests.cmake")

  file(GENERATE OUTPUT "${ctest_include_file}" CONTENT "\
execute_process(
  COMMAND \"$<TARGET_FILE:${target}>\" --list
  OUTPUT_VARIABLE _mvcc_test_list
  RESULT_VARIABLE _mvcc_list_result
)
if(NOT _mvcc_list_result EQUAL 0)
  message(FATAL_ERROR \"mvcc_discover_tests: '${target} --list' failed (${target} not built yet?)\")
endif()
string(STRIP \"\${_mvcc_test_list}\" _mvcc_test_list)
if(NOT _mvcc_test_list STREQUAL \"\")
  # Each line into a ; separated string so we can turn it into a list
  string(REPLACE \"\\n\" \";\" _mvcc_test_list \"\${_mvcc_test_list}\")
  foreach(_mvcc_test IN LISTS _mvcc_test_list)
    # replace each line's '|' with ';' so we can split into a test name and location
    string(REPLACE \"|\" \";\" _mvcc_test_parts \"\${_mvcc_test}\")

    list(GET _mvcc_test_parts 0 _mvcc_test_name)
    list(GET _mvcc_test_parts 1 _mvcc_test_location)

    # Positional add_test(<name> <command> [args...]), not the NAME/COMMAND
    # keyword form: ctest include()s this file in its own bare script
    # interpreter, with no cmake_minimum_required() in scope, so it falls
    # back to ancient default policies that don't recognize the keyword
    # form -- it silently parses \"NAME\" as a literal test name instead.
    add_test(\"${target}.\${_mvcc_test_name}\" \"$<TARGET_FILE:${target}>\" --exact \"\${_mvcc_test_name}\")
    set_tests_properties(\"${target}.\${_mvcc_test_name}\" PROPERTIES DEF_SOURCE_LINE \"\${_mvcc_test_location}\" TIMEOUT ${ARG_TIMEOUT} ${_mvcc_extra_props})
  endforeach()
endif()
")

  set_property(DIRECTORY APPEND PROPERTY TEST_INCLUDE_FILES "${ctest_include_file}")
endfunction()
