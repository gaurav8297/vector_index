cmake_minimum_required(VERSION 3.25)
project(VECTOR_INDEX LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)
set(CMAKE_CXX_VISIBILITY_PRESET hidden)
set(CMAKE_C_VISIBILITY_PRESET hidden)
set(CMAKE_EXPORT_COMPILE_COMMANDS TRUE)
set(CMAKE_FIND_PACKAGE_RESOLVE_SYMLINKS TRUE)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_VISIBILITY_INLINES_HIDDEN ON)
# On Linux, symbols in executables are not accessible by loaded shared libraries (e.g. via dlopen(3)). However, we need to export public symbols in executables so that extensions can access public symbols. This enables that behaviour.
set(CMAKE_ENABLE_EXPORTS TRUE)

# Try to find the header
FIND_PATH(LIBUV_INCLUDE_DIR NAMES uv.h REQUIRED)

# Try to find the library
FIND_LIBRARY(LIBUV_LIBRARY NAMES uv libuv REQUIRED)

# place binaries and libraries according to GNU standards
include(GNUInstallDirs)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_BINDIR})

function(add_test TEST_NAME)
    set(SRCS ${ARGN})
    add_executable(${TEST_NAME} ${SRCS})
    target_link_libraries(${TEST_NAME} PRIVATE vector_index faiss gtest_main)
#    target_include_directories(${TEST_NAME} PRIVATE ${PROJECT_SOURCE_DIR}/test/include)
    include(GoogleTest)
    gtest_discover_tests(${TEST_NAME})
endfunction()

set(CMAKE_C_FLAGS "-g")
set(CMAKE_CXX_FLAGS "-g")

# add include path
include_directories(${PROJECT_SOURCE_DIR}/src/include)
include_directories(${PROJECT_SOURCE_DIR}/third_party/spdlog)
include_directories(${PROJECT_SOURCE_DIR}/third_party/faiss)
include_directories(${LIBUV_INCLUDE_DIR})
# add source path
add_subdirectory(${PROJECT_SOURCE_DIR}/third_party/faiss)
add_subdirectory(${PROJECT_SOURCE_DIR}/src)
add_definitions(-DTEST_FILES_DIR="test/test_files")
add_subdirectory(${PROJECT_SOURCE_DIR}/test)

add_executable(vector_index_main ${PROJECT_SOURCE_DIR}/src/main.cpp)
target_link_libraries(vector_index_main PUBLIC vector_index faiss ${LIBUV_LIBRARY})
target_include_directories(vector_index_main PUBLIC
        ${PROJECT_SOURCE_DIR}/src/include
        ${PROJECT_SOURCE_DIR}/third_party/spdlog
        ${PROJECT_SOURCE_DIR}/third_party/faiss)
