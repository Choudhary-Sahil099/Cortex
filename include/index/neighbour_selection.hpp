#pragma once

#include "index/hnsw_search.hpp"

#include <cstddef>
#include <vector>

namespace cortex::index {

    std::vector<HNSWSearchResult> select_neighbors(
        const std::vector<HNSWSearchResult>& candidates,
        std::size_t max_neighbors
    );

}