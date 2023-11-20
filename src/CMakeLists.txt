add_library(vector_index
        OBJECT
        utils.cpp
        sa_tree.cpp
        small_world.cpp)

set(ALL_OBJECT_FILES
        ${ALL_OBJECT_FILES} $<TARGET_OBJECTS:vector_index>
        PARENT_SCOPE)

target_include_directories(vector_index
        PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include> $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
