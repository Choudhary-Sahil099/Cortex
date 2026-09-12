#pragma once

#include "vector/vector.hpp"
#include "vector/vector_store.hpp"

#include <cstddef>
#include <vector>

namespace cortex::vector
{
    void batch_l2_distance(
        const Vector& query,
        const std::vector<Vector>& vectors,
        std::vector<float>& distances
    );

    void batch_l2_distance_scalar(
        const Vector& query,
        const std::vector<Vector>& vectors,
        std::vector<float>& distances
    );

    void batch_l2_distance_simd(
        const Vector& query,
        const std::vector<Vector>& vectors,
        std::vector<float>& distances
    );

    
}