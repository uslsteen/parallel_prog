cmake_minimum_required(VERSION 3.16)
project(parallel_prog)

find_package(MPI REQUIRED)

set(PROBLEM lab1)

add_executable(${PROBLEM} ${PROBLEM}.cc)

target_include_directories(${PROBLEM} PRIVATE SYSTEM ${MPI_INCLUDE_PATH})
target_link_libraries(${PROBLEM} PRIVATE ${MPI_C_LIBRARIES})
target_compile_options(${PROBLEM} PRIVATE -Wall -Wextra)