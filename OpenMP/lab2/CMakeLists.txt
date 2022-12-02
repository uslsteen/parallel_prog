set(PROBLEM lab2)

list(APPEND PROBLEMS ${PROBLEM})
set(PROBLEMS ${PROBLEM} PARENT_SCOPE)

add_executable(${PROBLEM} 2b.cc)
target_link_libraries(${PROBLEM} PRIVATE OpenMP::OpenMP_CXX)