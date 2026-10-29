#include "index/hnsw.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <benchmark/benchmark.h>

using cortex::index::HNSWIndex;
using cortex::index::HNSWSearchResult;
using cortex::vector::Vector;

namespace
{

    constexpr std::size_t kDatasetSize = 10'000;
    constexpr std::size_t kQueryCount = 100;
    constexpr std::size_t kDimension = 128;
    constexpr std::size_t kK = 10;

    std::vector<HNSWSearchResult> brute_force_search(
        const HNSWIndex &index,
        const Vector &query,
        std::size_t k)
    {
        if (query.dimension() != index.dimension())
        {
            throw std::invalid_argument("Query/index dimension mismatch");
        }

        std::vector<HNSWSearchResult> results;
        results.reserve(index.size());

        for (std::size_t id = 0; id < index.size(); ++id)
        {
            const float *data = index.vector_data(id);
            float squared_distance = 0.0f;

            for (std::size_t d = 0; d < query.dimension(); ++d)
            {
                const float difference = query[d] - data[d];
                squared_distance += difference * difference;
            }

            results.push_back({id, squared_distance});
        }

        std::sort(results.begin(), results.end(),
                  [](const HNSWSearchResult &a, const HNSWSearchResult &b)
                  {
                      if (a.distance != b.distance)
                          return a.distance < b.distance;
                      return a.id < b.id;
                  });

        if (results.size() > k)
            results.resize(k);
        return results;
    }
    struct QueryGroundTruth
    {
        std::vector<HNSWSearchResult> exact_top_k;
    };
    struct BenchmarkData
    {
        HNSWIndex index;
        std::vector<Vector> queries;
        std::vector<QueryGroundTruth> ground_truth;

        BenchmarkData() : index(kDimension, 16, 100, 50, 42)
        {
            std::mt19937 generator(42);
            std::uniform_real_distribution<float> distribution(0.0f, 1.0f);

            for (std::size_t i = 0; i < kDatasetSize; ++i)
            {
                Vector vector(kDimension);
                for (std::size_t d = 0; d < kDimension; ++d)
                    vector[d] = distribution(generator);
                index.insert(std::move(vector));
            }

            queries.reserve(kQueryCount);
            for (std::size_t q = 0; q < kQueryCount; ++q)
            {
                Vector query(kDimension);
                for (std::size_t d = 0; d < kDimension; ++d)
                    query[d] = distribution(generator);
                queries.push_back(std::move(query));
            }

            ground_truth.reserve(queries.size());
            for (const auto &query : queries)
                ground_truth.push_back({brute_force_search(index, query, kK)});
        }
    };

    BenchmarkData &benchmark_data()
    {
        static BenchmarkData data;
        return data;
    }
    BenchmarkData &g_benchmark_data = benchmark_data();
    double recall_at_k(
        const std::vector<HNSWSearchResult> &approximate,
        const std::vector<HNSWSearchResult> &exact,
        std::size_t k)
    {
        if (k == 0 || exact.size() < k)
            return 0.0;

        std::size_t matches = 0;

        for (const auto &result : approximate)
        {
            const auto end = exact.begin() + k;

            const auto found = std::find_if(
                exact.begin(), end,
                [&result](const HNSWSearchResult &candidate)
                {
                    return candidate.id == result.id;
                });

            if (found != end)
                ++matches;
        }

        return static_cast<double>(matches) / static_cast<double>(k);
    }

    double mean_recall_at_k(
        const BenchmarkData &data,
        std::size_t k,
        std::size_t ef_search)
    {
        if (k == 0 || k > kK)
            throw std::invalid_argument("k must be between 1 and kK");

        double total_recall = 0.0;

        for (std::size_t i = 0; i < data.queries.size(); ++i)
        {
            const auto approximate =
                data.index.search(data.queries[i], k, ef_search);

            total_recall += recall_at_k(
                approximate,
                data.ground_truth[i].exact_top_k,
                k);
        }

        return total_recall / static_cast<double>(data.queries.size());
    }
    void BM_BruteForceSearch(benchmark::State &state)
    {
        auto &data = benchmark_data();
        std::size_t query_index = 0;
        for (auto _ : state)
        {
            const auto results = brute_force_search(
                data.index, data.queries[query_index], kK);
            benchmark::DoNotOptimize(results);
            query_index = (query_index + 1) % data.queries.size();
        }

        state.SetItemsProcessed(state.iterations());
        state.counters["QPS"] = benchmark::Counter(
            static_cast<double>(state.iterations()), benchmark::Counter::kIsRate);
    }
    BENCHMARK(BM_BruteForceSearch);

    struct HNSWConfig
    {
        std::size_t M;
        std::size_t ef_construction;
        std::size_t ef_search;
    };

    constexpr HNSWConfig kHNSWConfigs[] = {
        {16, 100, 200}, // Baseline
        {8, 100, 200},
        {16, 200, 200},
        {24, 200, 200},
        {32, 200, 200},
        {32, 400, 200}};

    void BM_HNSWConfigSweep(benchmark::State &state)
    {
        const auto config_index =
            static_cast<std::size_t>(state.range(0));

        if (config_index >= std::size(kHNSWConfigs))
        {
            state.SkipWithError("Invalid HNSW configuration index");
            return;
        }

        const auto config = kHNSWConfigs[config_index];
        auto &data = benchmark_data();

        for (auto _ : state)
        {
            (void)_;
            state.PauseTiming();

            const auto build_start = std::chrono::steady_clock::now();

            HNSWIndex index(
                kDimension,
                config.M,
                config.ef_construction,
                config.ef_search,
                42);

            // Rebuild from the exact same vectors, in the same ID order.
            for (std::size_t id = 0; id < data.index.size(); ++id)
            {
                const float *source = data.index.vector_data(id);

                Vector vector(kDimension);
                for (std::size_t d = 0; d < kDimension; ++d)
                {
                    vector[d] = source[d];
                }

                const auto inserted_id = index.insert(std::move(vector));

                if (inserted_id != id)
                {
                    state.SkipWithError(
                        "Rebuilt index returned unexpected vector ID");
                    return;
                }
            }

            const auto build_end = std::chrono::steady_clock::now();

            const double build_ms =
                std::chrono::duration<double, std::milli>(
                    build_end - build_start)
                    .count();

            // Evaluate recall outside the measured query loop.
            double total_recall = 0.0;

            for (std::size_t i = 0; i < data.queries.size(); ++i)
            {
                const auto results = index.search(
                    data.queries[i], kK, config.ef_search);

                total_recall += recall_at_k(
                    results, data.ground_truth[i].exact_top_k, kK);
            }

            const double recall_at_10 =
                total_recall / static_cast<double>(data.queries.size());

            constexpr std::size_t kRounds = 50;
            constexpr std::size_t kQueriesPerRound = kQueryCount;

            std::vector<double> latency_us;
            latency_us.reserve(kRounds * kQueriesPerRound);

            double query_seconds = 0.0;
            std::size_t queries_measured = 0;

            state.ResumeTiming();

            const auto query_batch_start = std::chrono::steady_clock::now();

            for (std::size_t round = 0; round < kRounds; ++round)
            {
                for (std::size_t i = 0; i < kQueriesPerRound; ++i)
                {
                    const auto start = std::chrono::steady_clock::now();

                    auto results = index.search(
                        data.queries[i], kK, config.ef_search);

                    const auto end = std::chrono::steady_clock::now();

                    benchmark::DoNotOptimize(results);

                    latency_us.push_back(
                        std::chrono::duration<double, std::micro>(
                            end - start)
                            .count());

                    ++queries_measured;
                }
            }

            const auto query_batch_end = std::chrono::steady_clock::now();

            query_seconds =
                std::chrono::duration<double>(
                    query_batch_end - query_batch_start)
                    .count();

            state.PauseTiming();

            std::sort(latency_us.begin(), latency_us.end());

            const auto percentile = [&latency_us](double p)
            {
                const auto index = static_cast<std::size_t>(
                                       std::ceil(p * latency_us.size())) -
                                   1;

                return latency_us[std::min(index, latency_us.size() - 1)];
            };

            state.counters["M"] = static_cast<double>(config.M);
            state.counters["efConstruction"] =
                static_cast<double>(config.ef_construction);
            state.counters["efSearch"] =
                static_cast<double>(config.ef_search);

            state.counters["BuildMs"] = build_ms;
            state.counters["RecallAt10"] = recall_at_10;
            state.counters["QPS"] =
                static_cast<double>(queries_measured) / query_seconds;

            state.counters["p50_us"] = percentile(0.50);
            state.counters["p95_us"] = percentile(0.95);
            state.counters["p99_us"] = percentile(0.99);
            state.counters["QueriesMeasured"] =
                static_cast<double>(queries_measured);

            state.SetItemsProcessed(static_cast<std::int64_t>(queries_measured));
        }
    }
}

BENCHMARK(BM_HNSWConfigSweep)
    ->Arg(0)
    ->Arg(1)
    ->Arg(2)
    ->Arg(3)
    ->Arg(4)
    ->Arg(5)
    ->Iterations(1);

void BM_HNSWSearch(benchmark::State &state)
{
    const std::size_t ef =
        static_cast<std::size_t>(state.range(0));

    auto &data = benchmark_data();

    const double recall1 = mean_recall_at_k(data, 1, ef);
    const double recall10 = mean_recall_at_k(data, 10, ef);

    std::size_t query_index = 0;

    for (auto _ : state)
    {
        const auto results = data.index.search(
            data.queries[query_index], kK, ef);

        benchmark::DoNotOptimize(results);

        query_index = (query_index + 1) % data.queries.size();
    }

    state.SetItemsProcessed(state.iterations());

    state.counters["QPS"] = benchmark::Counter(
        static_cast<double>(state.iterations()),
        benchmark::Counter::kIsRate);

    state.counters["RecallAt1"] = recall1;
    state.counters["RecallAt10"] = recall10;
}

BENCHMARK(BM_HNSWSearch)
    ->Arg(50)
    ->Arg(100)
    ->Arg(150)
    ->Arg(200)
    ->Arg(300);
void BM_HNSWTop1Search(benchmark::State &state)
{
    const std::size_t ef =
        static_cast<std::size_t>(state.range(0));

    auto &data = benchmark_data();

    // Evaluate exact top-1 accuracy outside the timed loop.
    const double recall1 = mean_recall_at_k(data, 1, ef);

    std::size_t query_index = 0;

    for (auto _ : state)
    {
        const auto results =
            data.index.search(data.queries[query_index], 1, ef);

        benchmark::DoNotOptimize(results);

        query_index = (query_index + 1) % data.queries.size();
    }

    state.SetItemsProcessed(state.iterations());

    state.counters["QPS"] = benchmark::Counter(
        static_cast<double>(state.iterations()),
        benchmark::Counter::kIsRate);

    state.counters["RecallAt1"] = recall1;
}

BENCHMARK(BM_HNSWTop1Search)
    ->Arg(1)
    ->Arg(10)
    ->Arg(25)
    ->Arg(50)
    ->Arg(100);
void BM_HNSWLatencyPercentiles(benchmark::State &state)
{
    const std::size_t ef =
        static_cast<std::size_t>(state.range(0));

    auto &data = benchmark_data();

    constexpr std::size_t kRounds = 50;
    constexpr std::size_t kQueriesPerRound = kQueryCount;

    std::vector<double> latency_us;
    latency_us.reserve(kRounds * kQueriesPerRound);

    for (auto _ : state)
    {
        for (std::size_t i = 0; i < kQueriesPerRound; ++i)
        {
            const auto start = std::chrono::steady_clock::now();

            auto results = data.index.search(
                data.queries[i], kK, ef);

            const auto end = std::chrono::steady_clock::now();

            benchmark::DoNotOptimize(results);

            const auto elapsed =
                std::chrono::duration<double, std::micro>(
                    end - start)
                    .count();

            latency_us.push_back(elapsed);
        }
    }

    state.PauseTiming();

    if (!latency_us.empty())
    {
        std::sort(latency_us.begin(), latency_us.end());

        const auto get_percentile =
            [&latency_us](double p) -> double
        {
            const std::size_t idx =
                static_cast<std::size_t>(
                    std::ceil(p * latency_us.size())) -
                1;

            return latency_us[std::min(idx, latency_us.size() - 1)];
        };

        state.counters["QueriesMeasured"] =
            static_cast<double>(latency_us.size());
        state.counters["p50_us"] = get_percentile(0.50);
        state.counters["p95_us"] = get_percentile(0.95);
        state.counters["p99_us"] = get_percentile(0.99);
    }
}
BENCHMARK(BM_HNSWLatencyPercentiles)
    ->Arg(10)
    ->Arg(25)
    ->Arg(50)
    ->Arg(100)
    ->Iterations(50);