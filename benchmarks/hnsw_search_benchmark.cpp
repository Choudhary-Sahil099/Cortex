#include "index/hnsw.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <random>
#include <vector>

#include <benchmark/benchmark.h>

using cortex::index::HNSWIndex;
using cortex::index::HNSWSearchResult;
using cortex::vector::Vector;

namespace {

    std::vector<HNSWSearchResult> brute_force_search(
        const HNSWIndex& index,
        const Vector& query,
        std::size_t k
    ) {
        std::vector<HNSWSearchResult> results;

        results.reserve(index.size());

        for (std::size_t id = 0;
            id < index.size();
            ++id) {

            const float* data =
                index.vector_data(id);

            float distance = 0.0f;

            for (std::size_t d = 0;
                d < query.dimension();
                ++d) {

                const float difference =
                    query[d] - data[d];

                distance +=
                    difference * difference;
            }

            results.push_back({
                id,
                distance
                });
        }

        std::sort(
            results.begin(),
            results.end(),
            [](const HNSWSearchResult& a,
                const HNSWSearchResult& b) {
                    return a.distance < b.distance;
            }
        );

        if (results.size() > k) {
            results.resize(k);
        }

        return results;
    }


    struct BenchmarkData {
        HNSWIndex index;
        std::vector<Vector> queries;

        BenchmarkData()
            : index(
                128,
                16,
                100,
                50
            ) {

            constexpr std::size_t dataset_size = 10000;
            constexpr std::size_t query_count = 100;
            constexpr std::size_t dimension = 128;

            std::mt19937 generator(42);

            std::uniform_real_distribution<float>
                distribution(0.0f, 1.0f);

            // Build dataset.
            for (std::size_t i = 0;
                i < dataset_size;
                ++i) {

                Vector vector(dimension);

                for (std::size_t d = 0;
                    d < dimension;
                    ++d) {

                    vector[d] =
                        distribution(generator);
                }

                index.insert(vector);
            }

            // Generate queries.
            queries.reserve(query_count);

            for (std::size_t q = 0;
                q < query_count;
                ++q) {

                Vector query(dimension);

                for (std::size_t d = 0;
                    d < dimension;
                    ++d) {

                    query[d] =
                        distribution(generator);
                }

                queries.push_back(
                    std::move(query)
                );
            }
        }
    };


    BenchmarkData& benchmark_data() {
        static BenchmarkData data;
        return data;
    }

} 

static void BM_BruteForceSearch(
    benchmark::State& state
) {
    auto& data = benchmark_data();

    constexpr std::size_t k = 10;

    std::size_t query_index = 0;

    for (auto _ : state) {

        const auto& query =
            data.queries[query_index];

        const auto results =
            brute_force_search(
                data.index,
                query,
                k
            );

        benchmark::DoNotOptimize(results);

        query_index =
            (query_index + 1) %
            data.queries.size();
    }

    state.SetItemsProcessed(
        state.iterations()
    );
}

BENCHMARK(BM_BruteForceSearch);


static void BM_HNSWSearch(
    benchmark::State& state
) {
    auto& data = benchmark_data();

    constexpr std::size_t k = 10;

    const std::size_t ef =
        static_cast<std::size_t>(
            state.range(0)
            );

    std::size_t query_index = 0;

    for (auto _ : state) {

        const auto& query =
            data.queries[query_index];

        const auto results =
            data.index.search(
                query,
                k,
                ef
            );

        benchmark::DoNotOptimize(results);

        query_index =
            (query_index + 1) %
            data.queries.size();
    }

    state.SetItemsProcessed(
        state.iterations()
    );
}

BENCHMARK(BM_HNSWSearch)
->Arg(10)
->Arg(25)
->Arg(50)
->Arg(100);

// dont add benchmark main here 