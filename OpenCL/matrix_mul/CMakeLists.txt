set(PROBLEM cl_mul)

list(APPEND PROBLEMS ${PROBLEM})
set(PROBLEMS ${PROBLEM} PARENT_SCOPE)

add_executable(${PROBLEM} ${PROBLEM}.cc err_cl.cc main.cc)

target_include_directories(${PROBLEM} PRIVATE ${CMAKE_SOURCE_DIR}/include)##/home/anton/code/parallel_prog/include)
target_link_libraries(${PROBLEM} OpenCL)