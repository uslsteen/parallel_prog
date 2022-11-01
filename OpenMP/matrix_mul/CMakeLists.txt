set(PROBLEM matrix)
set(CMAKE_CXX_FLAGS "-mavx2 -mbmi2 -mpopcnt -mfma -mcx16 -mtune=znver1 -O0")

set(WARN_OPTIONS 
-Wall -Wextra -Wunused -Wpedantic 
-Wuseless-cast
-Wunreachable-code)

list(APPEND PROBLEMS ${PROBLEM})
set(PROBLEMS ${PROBLEM} PARENT_SCOPE)

add_executable(${PROBLEM} ${PROBLEM}.cc)
target_link_libraries(${PROBLEM} PRIVATE OpenMP::OpenMP_CXX)