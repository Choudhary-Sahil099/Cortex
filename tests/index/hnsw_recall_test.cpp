#include "index/hnsw.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <random>
#include <vector>
#include <iostream>
#include <gtest/gtest.h>

using cortex::index::HNSWIndex;
using cortex::index::HNSWSearchResult;
using cortex::vector::Vector;


//Of the true K nearest vectors found by brute force, what fraction did HNSW retrieve?
namespace {

    std::vector<HNSWSearchResult> brute_force_search(
        const HNSWIndex& index,
        const Vector& query,
        std::size_t k
    ) {
        std::vector<HNSWSearchResult> results;

        results.reserve(index.size());

        for (std::size_t id = 0; id < index.size(); ++id) {

            const float* data =
                index.vector_data(id);

            float squared_distance = 0.0f;

            for (std::size_t i = 0;
                i < query.dimension();
                ++i) {

                const float difference =
                    query[i] - data[i];

                squared_distance +=
                    difference * difference;
            }

            results.push_back({
                id,
                std::sqrt(squared_distance)
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


    float calculate_recall(
        const std::vector<HNSWSearchResult>& exact,
        const std::vector<HNSWSearchResult>& approximate
    ) {
        if (exact.empty()) {
            return 1.0f;
        }

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

            if (it != exact.end()) {
                ++matches;
            }
        }

        return static_cast<float>(matches) /
            static_cast<float>(exact.size());
    }

} // namespace


TEST(HNSWRecallTest, RecallAtK) {

    constexpr std::size_t dimension = 128;
    constexpr std::size_t dataset_size = 1000;
    constexpr std::size_t query_count = 20;

    constexpr std::size_t M = 16;
    constexpr std::size_t ef_construction = 100;

    HNSWIndex index(
        dimension,
        M,
        ef_construction,
        50,
        42
    );

    std::mt19937 generator(42);

    std::uniform_real_distribution<float> distribution(
        0.0f,
        1.0f
    );

    // Build the dataset.
    for (std::size_t i = 0;
        i < dataset_size;
        ++i) {

        Vector vector(dimension);

        for (std::size_t d = 0;
            d < dimension;
            ++d) {

            vector[d] = distribution(generator);
        }

        index.insert(std::move(vector));
    }

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

    constexpr std::size_t ef_values[] = {
     10,
     25,
     50,
     100
    };

    constexpr std::size_t k_values[] = {
        1,
        5,
        10
    };

    for (const std::size_t ef : ef_values) {

        std::cout
            << "\nef_search = "
            << ef
            << '\n';

        for (const std::size_t k : k_values) {

            float total_recall = 0.0f;

            for (const auto& query : queries) {

                const auto exact =
                    brute_force_search(
                        index,
                        query,
                        k
                    );

                const auto approximate =
                    index.search(
                        query,
                        k,
                        ef
                    );

                total_recall +=
                    calculate_recall(
                        exact,
                        approximate
                    );
            }

            const float average_recall =
                total_recall /
                static_cast<float>(query_count);

            std::cout
                << "Recall@" << k
                << " = "
                << average_recall
                << '\n';
        }
    }
}