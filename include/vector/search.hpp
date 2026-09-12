#pragma once

#include "vector/vector.hpp"

#include <cstddef>
#include <vector>

namespace cortex::vector
{
    struct SearchResult
    {
        std::size_t index;
        float distance;
    };

    std::vector<SearchResult> search_top_k(
        const Vector& query,
        const std::vector<Vector>& vectors,
        std::size_t k
    );
}