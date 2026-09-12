#pragma once

#include "vector/vector.hpp"

#include <vector>

namespace cortex::vector {

    void batch_l2_distance_reference(
        const Vector& query,
        const std::vector<Vector>& vectors,
        std::vector<float>& distances);

}