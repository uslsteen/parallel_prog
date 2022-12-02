set(PROBLEM devices)

list(APPEND PROBLEMS ${PROBLEM})
set(PROBLEMS ${PROBLEM} PARENT_SCOPE)

add_executable(${PROBLEM} ${PROBLEM}.cc)
target_link_libraries(${PROBLEM} OpenCL)