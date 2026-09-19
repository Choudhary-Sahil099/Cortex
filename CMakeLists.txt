 cmake_minimum_required(VERSION 3.20)

project(Cortex
    VERSION 0.1.0
    LANGUAGES CXX
)

# ============================================================
# Global C++ Configuration
# ============================================================

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ============================================================
# Cortex Core Library
# ============================================================

add_library(cortex_core

    #core files
    src/core/buffer.cpp
    src/core/cpu_features.cpp

    #vector files
    src/vector/vector.cpp
    src/vector/vector_batch.cpp
    src/vector/vector_store.cpp
    src/vector/distance.cpp
    src/vector/distance_scalar.cpp
    src/vector/dot_product.cpp
    src/vector/batch_distance.cpp
    src/vector/batch_distance_reference.cpp
    src/vector/batch_distance_simd.cpp
    src/vector/search.cpp
    src/vector/backend.cpp
    src/vector/similarity.cpp

    #index files
    src/index/hnsw_node.cpp # kepp this above
    src/index/hnsw.cpp
    src/index/hnsw_level_generator.cpp
)

target_include_directories(cortex_core
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# ============================================================
# CPU / SIMD Configuration
# ============================================================

if(MSVC)

    # target_compile_options() is used to set compiler flags for the cortex_core library.
    target_compile_options(cortex_core PRIVATE
        /arch:AVX2
    )

    # Generate MSVC assembly listing for distance.cpp
    set_source_files_properties(
        src/vector/distance.cpp
        PROPERTIES
            COMPILE_OPTIONS
                "/FAcs;/Fa${CMAKE_CURRENT_BINARY_DIR}/distance.asm"
    )

elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")

    target_compile_options(cortex_core PRIVATE
        -mavx2
        -mfma
    )

    # Generate assembly for the distance implementation.
    set_source_files_properties(
        src/vector/distance.cpp
        PROPERTIES
            COMPILE_OPTIONS
                "-S;-masm=intel"
    )

endif()

# ============================================================
# Cortex Application
# ============================================================

add_executable(cortex
    src/main.cpp
)

target_link_libraries(cortex
    PRIVATE
        cortex_core
)

# ============================================================
# Dependencies
# ============================================================

include(FetchContent)

# ------------------------------------------------------------
# Google Benchmark
# ------------------------------------------------------------

FetchContent_Declare(
    google_benchmark
    GIT_REPOSITORY https://github.com/google/benchmark.git
    GIT_TAG        v1.9.4
)

set(
    BENCHMARK_ENABLE_TESTING
    OFF
    CACHE BOOL "" FORCE
)

FetchContent_MakeAvailable(google_benchmark)

# ------------------------------------------------------------
# GoogleTest
# ------------------------------------------------------------

FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG        v1.17.0
)

# Required for MSVC runtime compatibility.
set(
    gtest_force_shared_crt
    ON
    CACHE BOOL "" FORCE
)

FetchContent_MakeAvailable(googletest)

# ============================================================
# Tests
# ============================================================

enable_testing()

add_executable(cortex_tests
    #core tests
    tests/core/buffer_test.cpp


    #vector tests
    tests/vector/distance_test.cpp
    tests/vector/vector_test.cpp
    tests/vector/vector_batch_test.cpp
    tests/vector/search_test.cpp
    tests/vector/batch_distance_test.cpp
    tests/vector/vector_store_test.cpp

    #index tests
    tests/index/hnsw_level_generator_test.cpp
)

target_link_libraries(cortex_tests
    PRIVATE
        cortex_core
        GTest::gtest_main
)

include(GoogleTest)

gtest_discover_tests(cortex_tests)

# ============================================================
# Benchmarks
# ============================================================

add_executable(cortex_benchmarks
    benchmarks/distance_benchmark.cpp
)

target_link_libraries(cortex_benchmarks
    PRIVATE
        cortex_core
        benchmark::benchmark
        benchmark::benchmark_main
)

# ============================================================
# Build Information
# ============================================================

if(MSVC)
    add_custom_command(
        TARGET cortex_core
        POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E echo
            "Cortex SIMD assembly generated at:"
            "${CMAKE_CURRENT_BINARY_DIR}/distance.asm"
    )
endif()