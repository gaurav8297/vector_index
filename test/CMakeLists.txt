include(FetchContent)
FetchContent_Declare(gtest
        QUIET
        URL https://github.com/google/googletest/archive/release-1.10.0.tar.gz
        )
# configure build of googletest
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
set(BUILD_GMOCK OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(gtest)
add_library(GTest::GTest INTERFACE IMPORTED)
target_link_libraries(GTest::GTest INTERFACE gtest_main)
target_link_libraries(GTest::GTest INTERFACE gmock_main)

enable_testing()
add_test(sa_tree_test sa_tree_test.cpp)
add_test(swng_test swng_test.cpp)
add_test(min_queue_test min_queue_test.cpp)
add_test(hnsw_test hnsw_test.cpp)
add_test(hnsw_pq_test hnsw_pq_test.cpp)
