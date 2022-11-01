set(PROBLEM hello)
  
list(APPEND PROBLEMS ${PROBLEM})
set(PROBLEMS ${PROBLEM} PARENT_SCOPE)

add_executable(${PROBLEM}_MPI ${PROBLEM}.c)

target_include_directories(${PROBLEM}_MPI PRIVATE SYSTEM ${MPI_INCLUDE_PATH})
target_link_libraries(${PROBLEM}_MPI PRIVATE ${MPI_C_LIBRARIES})
