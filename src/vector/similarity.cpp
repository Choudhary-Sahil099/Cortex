#include "vector/similarity.hpp"
#include "vector/dot_product.hpp"

#include <stdexcept>
#include <cmath>

namespace cortex::vector
{
    float cosine_similarity(
        const Vector& a,
        const Vector& b)
    {
        if (a.dimension() != b.dimension())
        {
            throw std::invalid_argument(
                "Vectors must have the same dimensions"
            );
        }

        const float norm_a_squared =
            dot_product(a, a);

        const float norm_b_squared =
            dot_product(b, b);

        if (norm_a_squared == 0.0f ||
            norm_b_squared == 0.0f)
        {
            throw std::invalid_argument(
                "Undefined cosine similarity"
            );
        }

        const float dot =
            dot_product(a, b);

        const float denominator =
            std::sqrt(norm_a_squared * norm_b_squared);

        return dot / denominator;
    }
}
