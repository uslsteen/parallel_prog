cmake_minimum_required(VERSION 3.16)
project(parallel_prog)
#
set(CMAKE_VERBOSE_MAKEFILE)
set(DIRS OpenMP MPI pthread)
#
include(cmake/subdirs.cmake)

foreach(DIR ${DIRS})
  add_subdirectory(${DIR})
endforeach()

set(MSG_STR)
string(REPLACE ";" ", " MSG_STR "${TARGETS}")
message(STATUS "Collected targets: ${MSG_STR}")

foreach(TARGET IN LISTS TARGETS)
  target_compile_options(${TARGET} PRIVATE -Wall -Wextra -Wpedantic)
  target_compile_features(${TARGET} PRIVATE cxx_std_17)
endforeach()