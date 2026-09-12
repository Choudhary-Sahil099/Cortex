#include "vector/batch_distance_reference.hpp"

#include <cmath>
#include <stdexcept>

namespace cortex::vector {

    void batch_l2_distance_reference(
        const Vector& query,
        const std::vector<Vector>& vectors,
        std::vector<float>& distances)
    {
        for (const Vector& vector : vectors) {
            if (vector.dimension() != query.dimension()) {
                throw std::invalid_argument(
                    "All vectors must have the same dimensions as the query"
                );
            }
        }

        distances.resize(vectors.size());

        for (std::size_t i = 0; i < vectors.size(); ++i) {
            float sum = 0.0f;

            for (std::size_t j = 0; j < query.dimension(); ++j) {
                const float difference = query[j] - vectors[i][j];
                sum += difference * difference;
            }

            distances[i] = std::sqrt(sum);
        }
    }

}