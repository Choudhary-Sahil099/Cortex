#pragma once

#include "vector/vector.hpp"

namespace cortex::vector
{
    float dot_product_scalar(
        const Vector& a,
        const Vector& b
    );

    float dot_product_avx2(
        const Vector& a,
        const Vector& b
    );

    float dot_product_avx2_fma(
        const Vector& a,
        const Vector& b
    );

    float dot_product(
        const Vector& a,
        const Vector& b
    );
}