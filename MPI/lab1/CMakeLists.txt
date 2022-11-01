cmake_minimum_required(VERSION 3.16)
project(parallel_prog)

find_package(MPI REQUIRED)

set(PROBLEM lab1)

add_executable(${PROBLEM}_MPI ${PROBLEM}.cc)

target_include_directories(${PROBLEM}_MPI SYSTEM PRIVATE ${MPI_CXX_INCLUDE_PATH} ${MPI_C_INCLUDE_PATH})
target_link_libraries(${PROBLEM}_MPI PRIVATE ${MPI_CXX_LIBRARIES})