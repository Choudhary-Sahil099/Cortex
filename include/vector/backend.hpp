#pragma once

#include "vector/vector.hpp"

#include <cstddef>

namespace cortex::vector
{
    using DistanceFunction =
        float (*)(const Vector&, const Vector&);

    using DotProductFunction =
        float (*)(const Vector&, const Vector&);

    using RawDistanceFunction =
        float (*)(const float*, const float*, std::size_t);

    enum class DistanceBackend
    {
        Scalar,
        AVX2,
        AVX2_FMA
    };

    struct VectorBackend
    {
        DistanceFunction l2_distance;
        DotProductFunction dot_product;
        RawDistanceFunction raw_l2_distance;
    };

    DistanceBackend select_distance_backend();

    const VectorBackend& get_vector_backend();
    const char* backend_name(DistanceBackend backend);
}