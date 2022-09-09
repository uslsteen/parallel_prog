cmake_minimum_required(VERSION 3.16)
project(parallel_prog)

find_package(OpenMP)

set(DIRS int hello)

if(OpenMP_CXX_FOUND)
    foreach(TARGET ${DIRS})
        add_executable(${TARGET} ${TARGET}/${TARGET}.cc)
        target_compile_options(${TARGET} INTERFACE "-Wall -Wextra")
        target_link_libraries(${TARGET} PUBLIC OpenMP::OpenMP_CXX)
    endforeach()
endif()