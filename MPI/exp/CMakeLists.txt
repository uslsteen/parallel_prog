set(PROBLEM exp)
  
list(APPEND PROBLEMS ${PROBLEM})
set(PROBLEMS ${PROBLEM} PARENT_SCOPE)

add_executable(${PROBLEM} ${PROBLEM}.c)