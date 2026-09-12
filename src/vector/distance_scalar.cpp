#include "vector/distance.hpp"
#include <cmath>
#include <stdexcept>

namespace cortex::vector
{
    float l2_distance_scalar(const Vector& a, const Vector& b)
    {
        if (a.dimension() != b.dimension())
        {
            throw std::invalid_argument(
                "Vector must have the same dimensions"
            );
        }

        return l2_distance_scalar(
            a.data(),
            b.data(),
            a.dimension()
        );
    }
    float l2_distance_scalar(
        const float* a,
        const float* b,
        std::size_t dimension)
    {
        float sum = 0.0f;

        for (std::size_t i = 0; i < dimension; ++i)
        {
            const float difference =
                a[i] - b[i];

            sum += difference * difference;
        }

        return std::sqrt(sum);
    }
}