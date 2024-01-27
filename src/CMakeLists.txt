include_directories(${CMAKE_CURRENT_BINARY_DIR})
add_library(vector_index
        OBJECT
        utils.cpp
        sa_tree.cpp
        small_world.cpp
        hnsw.cpp)

set(ALL_OBJECT_FILES
        ${ALL_OBJECT_FILES} $<TARGET_OBJECTS:vector_index>
        PARENT_SCOPE)

target_link_libraries(vector_index PUBLIC faiss)
set(VECTOR_INDEX_INCLUDES $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include> $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}> ${CMAKE_CURRENT_BINARY_DIR})
target_include_directories(vector_index PUBLIC VECTOR_INDEX_INCLUDES)
install(TARGETS vector_index)
