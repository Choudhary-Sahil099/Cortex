#pragma once

#include "vector/vector.hpp"

namespace cortex::vector
{
    float l2_distance(const Vector& a, const Vector& b);

    float l2_distance_scalar(const Vector& a, const Vector& b);

    float l2_distance_avx2(const Vector& a, const Vector& b);

    float l2_distance_avx2_fma(const Vector& a, const Vector& b);


    float l2_distance_scalar(
        const float* a,
        const float* b,
        std::size_t dimension
    );

    float l2_distance_avx2(
        const float* a,
        const float* b,
        std::size_t dimension
    );

    float l2_distance_avx2_fma(
        const float* a,
        const float* b,
        std::size_t dimension
    );

    float l2_distance(
        const float* a,
        const float* b,
        std::size_t dimension
    );
}