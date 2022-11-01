find_package(MPI REQUIRED)
SUBDIRS(DIRS ${CMAKE_CURRENT_SOURCE_DIR})

if(MPI_CXX_FOUND)
    foreach(DIR ${DIRS})
        add_subdirectory(${DIR})
    endforeach() 
endif()

list(APPEND TARGETS ${NEW_TAR})
set(TARGETS ${TARGETS} PARENT_SCOPE)