find_package(OpenMP REQUIRED)
##
SUBDIRS(DIRS ${CMAKE_CURRENT_SOURCE_DIR})

if(OpenMP_CXX_FOUND)
    foreach(DIR ${DIRS})
        add_subdirectory(${DIR})
    endforeach() 
endif()

list(APPEND TARGETS ${NEW_TAR})
set(TARGETS ${TARGETS} PARENT_SCOPE)