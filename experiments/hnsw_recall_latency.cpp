#include "index/hnsw.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

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

            results.push_back({ id, distance });
        }

        std::sort(
            results.begin(),
            results.end(),
            [](const auto& a, const auto& b) {
                return a.distance < b.distance;
            }
        );

        if (results.size() > k)
            results.resize(k);

        return results;
    }


    float calculate_recall(
        const std::vector<HNSWSearchResult>& exact,
        const std::vector<HNSWSearchResult>& approximate
    ) {
        if (exact.empty())
            return 1.0f;

        std::size_t matches = 0;

        for (const auto& result : approximate) {

            const auto it =
                std::find_if(
                    exact.begin(),
                    exact.end(),
                    [&](const auto& exact_result) {
                        return exact_result.id == result.id;
                    }
                );

            if (it != exact.end())
                ++matches;
        }

        return static_cast<float>(matches) /
            static_cast<float>(exact.size());
    }

}


int main() {

    constexpr std::size_t dimension = 128;
    constexpr std::size_t dataset_size = 10000;
    constexpr std::size_t query_count = 100;
    constexpr std::size_t k = 10;

    constexpr std::size_t M_VALUES[] = {
        8,
        16,
        32
    };

    constexpr std::size_t ef_construction = 100;

    constexpr std::size_t ef_values[] = {
        100,
        200,
        400,
        800
    };

    // --------------------------------------------------
    // Generate dataset once
    // --------------------------------------------------

    std::mt19937 generator(42);

    std::uniform_real_distribution<float>
        distribution(0.0f, 1.0f);

    std::vector<Vector> dataset;
    dataset.reserve(dataset_size);

    for (std::size_t i = 0;
        i < dataset_size;
        ++i) {

        Vector vector(dimension);

        for (std::size_t d = 0;
            d < dimension;
            ++d) {

            vector[d] = distribution(generator);
        }

        dataset.push_back(std::move(vector));
    }

    // --------------------------------------------------
    // Generate queries once
    // --------------------------------------------------

    std::vector<Vector> queries;
    queries.reserve(query_count);

    for (std::size_t q = 0;
        q < query_count;
        ++q) {

        Vector query(dimension);

        for (std::size_t d = 0;
            d < dimension;
            ++d) {

            query[d] = distribution(generator);
        }

        queries.push_back(std::move(query));
    }

    // --------------------------------------------------
    // Header
    // --------------------------------------------------

    std::cout
        << "\nHNSW M Parameter Experiment\n";

    std::cout
        << "Dataset: "
        << dataset_size
        << " vectors\n";

    std::cout
        << "Dimension: "
        << dimension
        << '\n';

    std::cout
        << "Queries: "
        << query_count
        << '\n';

    std::cout
        << "K: "
        << k
        << '\n';

    std::cout
        << "ef_construction: "
        << ef_construction
        << "\n\n";


    // ==================================================
    // Experiment for each M
    // ==================================================

    for (const std::size_t M : M_VALUES) {

        std::cout
            << "========================================\n";

        std::cout
            << "M = "
            << M
            << '\n';

        std::cout
            << "========================================\n";


        // --------------------------------------------------
        // Construct HNSW index
        // --------------------------------------------------

        HNSWIndex index(
            dimension,
            M,
            ef_construction,
            100
        );


        // --------------------------------------------------
        // Build HNSW graph
        // --------------------------------------------------

        const auto build_start =
            std::chrono::steady_clock::now();

        for (const auto& vector : dataset) {
            index.insert(vector);
        }

        const auto build_end =
            std::chrono::steady_clock::now();

        const double build_time_ms =
            std::chrono::duration<double, std::milli>(
                build_end - build_start
            ).count();

        std::cout
            << "Build time: "
            << build_time_ms
            << " ms\n";


        // --------------------------------------------------
        // Compute exact ground truth once
        // --------------------------------------------------

        std::vector<
            std::vector<HNSWSearchResult>
        > ground_truth;

        ground_truth.reserve(query_count);

        for (const auto& query : queries) {

            ground_truth.push_back(
                brute_force_search(
                    index,
                    query,
                    k
                )
            );
        }


        // --------------------------------------------------
        // Brute-force latency
        // --------------------------------------------------

        const auto brute_start =
            std::chrono::steady_clock::now();

        for (const auto& query : queries) {

            const auto result =
                brute_force_search(
                    index,
                    query,
                    k
                );

            volatile std::size_t result_id =
                result[0].id;

            (void)result_id;
        }

        const auto brute_end =
            std::chrono::steady_clock::now();

        const double brute_total_ms =
            std::chrono::duration<double, std::milli>(
                brute_end - brute_start
            ).count();

        const double brute_avg_us =
            (brute_total_ms * 1000.0) /
            static_cast<double>(query_count);

        std::cout
            << "\nBrute Force\n";

        std::cout
            << "  Recall@10: 100%\n";

        std::cout
            << "  Avg latency: "
            << brute_avg_us
            << " us/query\n\n";


        // --------------------------------------------------
        // HNSW search experiments
        // --------------------------------------------------

        std::cout
            << std::left
            << std::setw(12)
            << "ef_search"
            << std::setw(15)
            << "Recall@10"
            << "Avg Latency (us)\n";

        std::cout
            << "----------------------------------------\n";


        for (const std::size_t ef : ef_values) {

            float total_recall = 0.0f;

            const auto start =
                std::chrono::steady_clock::now();


            for (std::size_t q = 0;
                q < query_count;
                ++q) {

                const auto result =
                    index.search(
                        queries[q],
                        k,
                        ef
                    );

                total_recall +=
                    calculate_recall(
                        ground_truth[q],
                        result
                    );

                volatile std::size_t result_id =
                    result[0].id;

                (void)result_id;
            }


            const auto end =
                std::chrono::steady_clock::now();


            const double total_ms =
                std::chrono::duration<double, std::milli>(
                    end - start
                ).count();

            const double average_us =
                (total_ms * 1000.0) /
                static_cast<double>(query_count);

            const double recall =
                static_cast<double>(total_recall) /
                static_cast<double>(query_count);


            std::cout
                << std::left
                << std::setw(12)
                << ef
                << std::setw(15)
                << std::fixed
                << std::setprecision(3)
                << recall * 100.0
                << average_us
                << '\n';
        }

        std::cout << '\n';
    }

    return 0;
}