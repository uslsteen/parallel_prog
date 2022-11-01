SUBDIRS(DIRS ${CMAKE_CURRENT_SOURCE_DIR})
find_package(Threads REQUIRED)

if(Threads_FOUND)
    foreach(DIR ${DIRS})
        add_subdirectory(${DIR})
    endforeach() 
endif()

list(APPEND TARGETS ${NEW_TAR})
set(TARGETS ${TARGETS} PARENT_SCOPE)