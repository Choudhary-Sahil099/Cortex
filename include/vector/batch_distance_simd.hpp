#pragma once

#include "vector/vector.hpp"

#include <cstddef>
#include <vector>

namespace cortex::vector
{
    void batch_l2_distance_avx2(
        const Vector& query,
        const std::vector<Vector>& vectors,
        std::vector<float>& distances);

    void batch_l2_distance_avx2_fma(
        const Vector& query,
        const std::vector<Vector>& vectors,
        std::vector<float>& distances);
}