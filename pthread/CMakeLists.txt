cmake_minimum_required(VERSION 3.16)
project(parallel_prog)

set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

set(DIRS hello mutex sum)
set(PROBLEMS)

foreach(DIR ${DIRS})
  add_subdirectory(${DIR})
endforeach()

foreach(PROBLEM IN LISTS PROBLEMS)
  target_link_libraries(${PROBLEM} PRIVATE Threads::Threads)
  target_compile_options(${PROBLEM} PRIVATE -Wall -Wextra)
endforeach()