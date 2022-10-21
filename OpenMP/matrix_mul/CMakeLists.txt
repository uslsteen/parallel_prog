
cmake_minimum_required(VERSION 3.16)

project(matrix_mul)
#
find_package(OpenMP)

set(CMAKE_CXX_FLAGS "-mavx2 -mbmi2 -mpopcnt -mfma -mcx16 -mtune=znver1 -O0")

set(WARN_OPTIONS 
  -Wall -Wextra -Wunused -Wpedantic 
  -Wuseless-cast
  -Wunreachable-code)
  
if(OpenMP_CXX_FOUND)
    add_executable(${PROJECT_NAME} main.cc)
    target_link_libraries(${PROJECT_NAME} PUBLIC OpenMP::OpenMP_CXX)
    target_compile_options(${PROJECT_NAME} PRIVATE ${WARN_OPTIONS})
endif()
