set(PROBLEM sum)
  
list(APPEND PROBLEMS ${PROBLEM})
set(PROBLEMS ${PROBLEM} PARENT_SCOPE)

add_executable(${PROBLEM}_PTHR ${PROBLEM}.cc)
target_link_libraries(${PROBLEM}_PTHR PRIVATE Threads::Threads)
