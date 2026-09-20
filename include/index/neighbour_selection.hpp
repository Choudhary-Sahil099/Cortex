#pragma once

#include "index/hnsw_search.hpp"

#include <cstddef>
#include <vector>
#include <functional> // for the use of the lambda function

namespace cortex::index {

    using neighbourDistanceFunction = std::function<float(std::size_t, std::size_t)>;

    std::vector<HNSWSearchResult> select_neighbors(
        const std::vector<HNSWSearchResult>& candidates,
        std::size_t max_neighbours,
        const neighbourDistanceFunction& distanceFunction // adding neighbour distance function
    );
}