#include <benchmark/benchmark.h>
#include<vector/similarity.hpp>
#include "vector/distance.hpp"
#include "vector/dot_product.hpp"
#include "vector/batch_distance.hpp"
#include <vector>

#include "vector/search.hpp"
using cortex::vector::cosine_similarity;
#define DOT_ARGS \
    ->Arg(384) \
    ->Arg(768) \
    ->Arg(1024) \
    ->Arg(1536) \
    ->Arg(1537) \
    ->Arg(1539) \
    ->Arg(3072)

static void BM_ScalarL2(benchmark::State& state) // scalar test
{
    const std::size_t dimension = state.range(0);

    cortex::vector::Vector a(dimension);
    cortex::vector::Vector b(dimension);

    for (std::size_t i = 0; i < dimension; ++i)
    {
        a[i] = static_cast<float>(i) * 0.1f;
        b[i] = static_cast<float>(i) * 0.2f;
    }

    for (auto _ : state)
    {
        benchmark::DoNotOptimize(
            cortex::vector::l2_distance_scalar(a, b)
        );
    }
}

static void BM_AVX2L2(benchmark::State& state) // only AVX2
{
    const std::size_t dimension = state.range(0);

    cortex::vector::Vector a(dimension);
    cortex::vector::Vector b(dimension);

    for (std::size_t i = 0; i < dimension; ++i)
    {
        a[i] = static_cast<float>(i) * 0.1f;
        b[i] = static_cast<float>(i) * 0.2f;
    }

    for (auto _ : state)
    {
        benchmark::DoNotOptimize(
            cortex::vector::l2_distance_avx2(a, b)
        );
    }
}

static void BM_AVX2FMAL2(benchmark::State& state) // AVX2 and Fma 
{
    const std::size_t dimension = state.range(0);

    cortex::vector::Vector a(dimension);
    cortex::vector::Vector b(dimension);

    for (std::size_t i = 0; i < dimension; ++i)
    {
        a[i] = static_cast<float>(i) * 0.1f;
        b[i] = static_cast<float>(i) * 0.2f;
    }

    for (auto _ : state)
    {
        benchmark::DoNotOptimize(
            cortex::vector::l2_distance_avx2_fma(a, b)
        );
    }
}
static void BM_AutomaticL2(benchmark::State& state)
{
    const std::size_t dimension = state.range(0);

    cortex::vector::Vector a(dimension);
    cortex::vector::Vector b(dimension);

    for (std::size_t i = 0; i < dimension; ++i)
    {
        a[i] = static_cast<float>(i) * 0.1f;
        b[i] = static_cast<float>(i) * 0.2f;
    }

    for (auto _ : state)
    {
        benchmark::DoNotOptimize(
            cortex::vector::l2_distance(a, b)
        );
    }

    state.SetComplexityN(dimension);
}

static void BM_DotProductScalar(benchmark::State& state)
{
    const std::size_t dimension = state.range(0);

    cortex::vector::Vector a(dimension);
    cortex::vector::Vector b(dimension);

    for (std::size_t i = 0; i < dimension; ++i)
    {
        a[i] = static_cast<float>(i) * 0.001f;
        b[i] = static_cast<float>(i) * 0.002f;
    }

    for (auto _ : state)
    {
        benchmark::DoNotOptimize(
            cortex::vector::dot_product_scalar(a, b)
        );
    }

    state.SetItemsProcessed(
        static_cast<int64_t>(state.iterations()) *
        static_cast<int64_t>(dimension)
    );
}


static void BM_DotProductAVX2(benchmark::State& state)
{
    const std::size_t dimension = state.range(0);

    cortex::vector::Vector a(dimension);
    cortex::vector::Vector b(dimension);

    for (std::size_t i = 0; i < dimension; ++i)
    {
        a[i] = static_cast<float>(i) * 0.001f;
        b[i] = static_cast<float>(i) * 0.002f;
    }

    for (auto _ : state)
    {
        benchmark::DoNotOptimize(
            cortex::vector::dot_product_avx2(a, b)
        );
    }

    state.SetItemsProcessed(
        static_cast<int64_t>(state.iterations()) *
        static_cast<int64_t>(dimension)
    );
}


static void BM_DotProductAVX2FMA(benchmark::State& state)
{
    const std::size_t dimension = state.range(0);

    cortex::vector::Vector a(dimension);
    cortex::vector::Vector b(dimension);

    for (std::size_t i = 0; i < dimension; ++i)
    {
        a[i] = static_cast<float>(i) * 0.001f;
        b[i] = static_cast<float>(i) * 0.002f;
    }

    for (auto _ : state)
    {
        benchmark::DoNotOptimize(
            cortex::vector::dot_product_avx2_fma(a, b)
        );
    }

    state.SetItemsProcessed(
        static_cast<int64_t>(state.iterations()) *
        static_cast<int64_t>(dimension)
    );
}


static void BM_DotProductAutomatic(benchmark::State& state)
{
    const std::size_t dimension = state.range(0);

    cortex::vector::Vector a(dimension);
    cortex::vector::Vector b(dimension);

    for (std::size_t i = 0; i < dimension; ++i)
    {
        a[i] = static_cast<float>(i) * 0.001f;
        b[i] = static_cast<float>(i) * 0.002f;
    }

    for (auto _ : state)
    {
        benchmark::DoNotOptimize(
            cortex::vector::dot_product(a, b)
        );
    }

    state.SetItemsProcessed(
        static_cast<int64_t>(state.iterations()) *
        static_cast<int64_t>(dimension)
    );
}


static void BM_CosineSimilarity(
    benchmark::State& state)
{
    const std::size_t dimension =
        static_cast<std::size_t>(state.range(0));

    cortex::vector::Vector a(dimension);
    cortex::vector::Vector b(dimension);

    for (std::size_t i = 0; i < dimension; ++i)
    {
        a[i] = static_cast<float>(i) * 0.1f;
        b[i] = static_cast<float>(i) * 0.2f;
    }

    for (auto _ : state)
    {
        benchmark::DoNotOptimize(
            cosine_similarity(a, b)
        );
    }

    state.SetItemsProcessed(
        state.iterations() * dimension
    );
}


static void BM_BatchL2Distance(benchmark::State& state)
{
    constexpr std::size_t dimension = 1536;
    constexpr std::size_t vector_count = 1000;

    cortex::vector::Vector query(dimension);

    std::vector<cortex::vector::Vector> vectors;
    vectors.reserve(vector_count);

    for (std::size_t i = 0; i < vector_count; ++i) {
        cortex::vector::Vector vector(dimension);

        for (std::size_t j = 0; j < dimension; ++j) {
            vector[j] = static_cast<float>((i + j) % 100) / 100.0f;
        }

        vectors.push_back(std::move(vector));
    }

    std::vector<float> distances;

    for (auto _ : state) {
        cortex::vector::batch_l2_distance(
            query,
            vectors,
            distances
        );

        benchmark::DoNotOptimize(distances.data());
    }

    state.SetItemsProcessed(
        state.iterations() * vector_count
    );
}

static void BM_TopKSearch(benchmark::State& state)
{
    constexpr std::size_t dimension = 1536;

    const std::size_t vector_count =
        static_cast<std::size_t>(state.range(0));

    constexpr std::size_t k = 10;

    cortex::vector::Vector query(dimension);

    for (std::size_t j = 0; j < dimension; ++j)
    {
        query[j] =
            static_cast<float>(j % 100) * 0.001f;
    }

    std::vector<cortex::vector::Vector> vectors;
    vectors.reserve(vector_count);

    for (std::size_t i = 0; i < vector_count; ++i)
    {
        cortex::vector::Vector v(dimension);

        for (std::size_t j = 0; j < dimension; ++j)
        {
            v[j] =
                static_cast<float>((i + j) % 100) * 0.001f;
        }

        vectors.push_back(std::move(v));
    }

    for (auto _ : state)
    {
        const auto results =
            cortex::vector::search_top_k(
                query,
                vectors,
                k
            );

        benchmark::DoNotOptimize(results.data());
    }

    state.SetItemsProcessed(
        static_cast<int64_t>(state.iterations()) *
        static_cast<int64_t>(vector_count)
    );
}

BENCHMARK(BM_ScalarL2)
->Arg(384)
->Arg(768)
->Arg(1024)
->Arg(1536)
->Arg(1537)
->Arg(1539)
->Arg(3072);

BENCHMARK(BM_AVX2L2)
->Arg(384)
->Arg(768)
->Arg(1024)
->Arg(1536)
->Arg(1537)
->Arg(1539)
->Arg(3072);

BENCHMARK(BM_AVX2FMAL2)
->Arg(384)
->Arg(768)
->Arg(1024)
->Arg(1536)
->Arg(1537)
->Arg(1539)
->Arg(3072);

BENCHMARK(BM_AutomaticL2)
->Arg(384)
->Arg(768)
->Arg(1024)
->Arg(1536)
->Arg(1537)
->Arg(1539)
->Arg(3072);


BENCHMARK(BM_DotProductScalar) DOT_ARGS;
BENCHMARK(BM_DotProductAVX2) DOT_ARGS;
BENCHMARK(BM_DotProductAVX2FMA) DOT_ARGS;
BENCHMARK(BM_DotProductAutomatic) DOT_ARGS;

BENCHMARK(BM_CosineSimilarity)
->Arg(384)
->Arg(768)
->Arg(1024)
->Arg(1536)
->Arg(1537)
->Arg(3072);

BENCHMARK(BM_BatchL2Distance);
BENCHMARK_MAIN();

BENCHMARK(BM_TopKSearch)
->Arg(100)
->Arg(1000)
->Arg(5000)
->Arg(10000);