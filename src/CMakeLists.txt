#Copyright(c) Microsoft Corporation.All rights reserved.
#Licensed under the MIT                        license.

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_COMPILE_WARNING_AS_ERROR ON)

if(MSVC)
    add_subdirectory(dll)
else()
    #file(GLOB CPP_SOURCES *.cpp)
    set(CPP_SOURCES abstract_data_store.cpp ann_exception.cpp disk_utils.cpp 
        distance.cpp index.cpp in_mem_graph_store.cpp in_mem_data_store.cpp ooc_in_mem_graph_store.cpp ooc_in_mem_data_store.cpp
        linux_aligned_file_reader.cpp linux_async_file_reader.cpp math_utils.cpp natural_number_map.cpp
        in_mem_data_store.cpp in_mem_graph_store.cpp
        natural_number_set.cpp memory_mapper.cpp partition.cpp pq.cpp
        pq_flash_index.cpp pq_flash_index_laann.cpp scratch.cpp logger.cpp utils.cpp filter_utils.cpp index_factory.cpp abstract_index.cpp pq_l2_distance.cpp pq_data_store.cpp)
    if (RESTAPI)
        list(APPEND CPP_SOURCES restapi/search_wrapper.cpp restapi/server.cpp)
    endif()
    add_library(${PROJECT_NAME} ${CPP_SOURCES})
    add_library(${PROJECT_NAME}_s STATIC ${CPP_SOURCES})
endif()

if (NOT MSVC)
    install(TARGETS ${PROJECT_NAME} LIBRARY)
endif()

# Link liburing for async I/O (Linux only)
if(NOT MSVC)
    find_library(LIBURING_LIBRARY NAMES uring HINTS $ENV{HOME}/local/lib PATHS /usr/lib /usr/local/lib)
    if(LIBURING_LIBRARY)
        target_link_libraries(${PROJECT_NAME} ${LIBURING_LIBRARY})
        target_link_libraries(${PROJECT_NAME}_s ${LIBURING_LIBRARY})
        message(STATUS "Linking liburing to diskann libraries: ${LIBURING_LIBRARY}")
    else()
        message(WARNING "liburing not found. Please ensure it's installed.")
    endif()
endif()
