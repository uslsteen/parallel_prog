set(PROBLEM fft)

list(APPEND PROBLEMS ${PROBLEM})
set(PROBLEMS ${PROBLEM} PARENT_SCOPE)

add_executable(${PROBLEM} main.cc)
target_link_libraries(${PROBLEM} PRIVATE OpenMP::OpenMP_CXX)