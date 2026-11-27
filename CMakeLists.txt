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
option(CORTEX_CORE_ONLY "Build dependency-free core and regression tests" OFF)
set(CORTEX_ENGINE_SOURCES
    src/core/buffer.cpp src/core/cpu_features.cpp src/core/vector_store.cpp
    src/vector/vector.cpp src/vector/vector_batch.cpp src/vector/vector_store.cpp
    src/vector/distance.cpp src/vector/distance_scalar.cpp src/vector/dot_product.cpp
    src/vector/batch_distance.cpp src/vector/batch_distance_reference.cpp
    src/vector/batch_distance_simd.cpp src/vector/search.cpp src/vector/backend.cpp
    src/vector/similarity.cpp src/vector/dispatch.cpp
    src/index/hnsw_node.cpp src/index/hnsw.cpp src/index/hnsw_level_generator.cpp
    src/index/neighbour_selection.cpp src/index/vector_index.cpp
    src/persistence/serializer.cpp src/persistence/index_manager.cpp
)
# Only intrinsic-bearing translation units receive AVX2 flags.
if(MSVC)
    set_source_files_properties(src/vector/distance.cpp src/vector/dot_product.cpp src/vector/batch_distance_simd.cpp PROPERTIES COMPILE_OPTIONS "/arch:AVX2")
elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    set_source_files_properties(src/vector/distance.cpp src/vector/dot_product.cpp src/vector/batch_distance_simd.cpp PROPERTIES COMPILE_OPTIONS "-mavx2;-ffp-contract=off")
endif()
if(CORTEX_CORE_ONLY)
    add_library(cortex_core ${CORTEX_ENGINE_SOURCES})
    target_include_directories(cortex_core PUBLIC include)
    enable_testing()
    add_executable(cortex_hardening_tests tests/hardening/core_regression.cpp)
    target_link_libraries(cortex_hardening_tests PRIVATE cortex_core)
    add_test(NAME cortex_hardening_tests COMMAND cortex_hardening_tests)
    return()
endif()

#==============
# ONNX 
#===========
set(ONNXRUNTIME_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/thirdParty/onnxruntime" CACHE PATH "ONNX Runtime SDK root")

set(ONNXRUNTIME_INCLUDE_DIR
    ${ONNXRUNTIME_ROOT}/include
)

if(WIN32)
    set(ONNXRUNTIME_LIBRARY ${ONNXRUNTIME_ROOT}/lib/onnxruntime.lib)
else()
    set(ONNXRUNTIME_LIBRARY ${ONNXRUNTIME_ROOT}/lib/libonnxruntime.so)
endif()

set(ONNXRUNTIME_DLL
    ${ONNXRUNTIME_ROOT}/lib/onnxruntime.dll
)



# =====================
# Cortex Core Library
# =======================

add_library(cortex_core

    ${CORTEX_ENGINE_SOURCES}

    #email ingestion
    src/email/email_document.cpp
    src/email/email_parserer.cpp
    src/email/email_cleaner.cpp
    src/email/email_chunker.cpp

    #embedding
    src/embedding/wordpiece_tokenizer.cpp
    src/embedding/bacsic_tokenizer.cpp
    src/embedding/bert_normalizer.cpp
    src/embedding/bert_tokenizer.cpp
    src/embedding/bgeEmbedder.cpp
    src/embedding/embedding_pipeline.cpp

    #RETRIVAL
    src/retrieval/retrievel.cpp
    src/retrieval/context_builder.cpp
    src/retrieval/query_transformer.cpp
    #rag
    src/rag/rag_pipeline.cpp

    #llm
    src/llm/local_llm.cpp

    #ingestion
    src/ingestion/email_indexer.cpp
    src/ingestion/email_file_loader.cpp

    #app
    src/app/cortex_app.cpp
    src/app/enviroment.cpp

    #evaluation
    src/evaluation/retrival_metric.cpp
)

target_include_directories(cortex_core
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${ONNXRUNTIME_INCLUDE_DIR}
)

#linking the libs

target_link_libraries(cortex_core
    PUBLIC
        ${ONNXRUNTIME_LIBRARY}
        CURL::libcurl
        nlohmann_json::nlohmann_json
)
# ============================================================
# CPU / SIMD Configuration
# ============================================================

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

find_package(CURL REQUIRED)
find_package(nlohmann_json CONFIG REQUIRED)

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

add_executable(cortex_hardening_tests tests/hardening/core_regression.cpp)
target_link_libraries(cortex_hardening_tests PRIVATE cortex_core)
add_test(NAME cortex_hardening_tests COMMAND cortex_hardening_tests)

add_executable(cortex_tests
    #core tests
    tests/core/buffer_test.cpp
    tests/core/vector_store_test.cpp

    #vector tests
    tests/vector/distance_test.cpp
    tests/vector/vector_test.cpp
    tests/vector/vector_batch_test.cpp
    tests/vector/search_test.cpp
    tests/vector/batch_distance_test.cpp
    tests/vector/vector_store_test.cpp

    #index tests
    tests/index/hnsw_level_generator_test.cpp
    tests/index/hnsw_test.cpp
    tests/index/neighbour_selection_test.cpp
    tests/index/hnsw_recall_test.cpp
    tests/index/vector_index_test.cpp


    #persistence
    tests/persistence/serializer_test.cpp

    # ingestion
    tests/ingestion/email_file_loader_test.cpp
    #email ingestion
    tests/email/email_document_test.cpp
    tests/email/email_parserer_test.cpp
    tests/email/email_cleaner_test.cpp
    tests/email/email_chunker_test.cpp


    #embedding
    tests/embedding/embedder_test.cpp
    tests/embedding/wordpiece_tokenizer_test.cpp
    tests/embedding/basicTokenizerTest.cpp
    tests/embedding/bertNormaizationTest.cpp
    tests/embedding/bertTokenizerTest.cpp
    tests/embedding/bge_embedder_test.cpp
    tests/embedding/embedding_pipeline_test.cpp


    #retireval tests
    tests/retrieval/context_builder_test.cpp
    tests/retrieval/query_transformer_test.cpp

    #llm
    tests/llm/llm_test.cpp

    #rag
    tests/rag/rag_pipeline_test.cpp

    #app
    tests/app/cortex_app_test.cpp

    #evaluation
    tests/evaluation/retrival_metric_test.cpp
)

target_link_libraries(cortex_tests
    PRIVATE
        cortex_core
        GTest::gtest_main
)

include(GoogleTest)

gtest_discover_tests(
    cortex_tests
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)

# ============================================================
# Benchmarks
# ============================================================

add_executable(cortex_benchmarks
    benchmarks/distance_benchmark.cpp
    benchmarks/hnsw_search_benchmark.cpp
    benchmarks/hnsw_real_data_benchmark.cpp
)

target_link_libraries(cortex_benchmarks
    PRIVATE
        cortex_core
        benchmark::benchmark
        benchmark::benchmark_main
)


#------------------------------
#----------Experiment ----------
#-------------------------------

add_executable(hnsw_recall_latency
    experiments/hnsw_recall_latency.cpp
)

target_link_libraries(hnsw_recall_latency
    PRIVATE
    cortex_core
)

# ---- bge

add_executable(bge_onnx_experiment

    experiments/embeddings/bgeOnnx.cpp

)

target_link_libraries(bge_onnx_experiment

    PRIVATE

        cortex_core

)

#--llm
add_executable(llm_test experiments/llm/llm_test.cpp)

target_link_libraries(
    llm_test
    PRIVATE
        cortex_core
)

add_executable(rag_local_llm
    experiments/llm/rag_local_llm.cpp
)

target_link_libraries(
    rag_local_llm
    PRIVATE
        cortex_core
)

if(WIN32)
    foreach(target cortex cortex_tests cortex_hardening_tests cortex_benchmarks bge_onnx_experiment hnsw_recall_latency llm_test rag_local_llm)
        add_custom_command(TARGET ${target} POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy_if_different "${ONNXRUNTIME_DLL}" $<TARGET_FILE_DIR:${target}>)
    endforeach()
endif()
# ============================================================
# Build Information
# ============================================================


#  llama-cli --version 
# winget install llama.cpp   
# where.exe llama-server
# where.exe llama-cli