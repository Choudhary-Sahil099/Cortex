#include "vector/batch_distance.hpp"

#include "vector/backend.hpp"
#include "vector/distance.hpp"

#include <stdexcept>

namespace cortex::vector
{
    namespace
    {
        void validate_dimensions(
            const Vector& query,
            const std::vector<Vector>& vectors)
        {
            for (const Vector& vector : vectors)
            {
                if (vector.dimension() != query.dimension())
                {
                    throw std::invalid_argument(
                        "All vectors must have the same dimensions as the query"
                    );
                }
            }
        }
    }

    void batch_l2_distance_scalar(
        const Vector& query,
        const std::vector<Vector>& vectors,
        std::vector<float>& distances)
    {
        validate_dimensions(query, vectors);

        distances.resize(vectors.size());

        for (std::size_t i = 0; i < vectors.size(); ++i)
        {
            distances[i] =
                l2_distance_scalar(query, vectors[i]);
        }
    }

    void batch_l2_distance_simd(
        const Vector& query,
        const std::vector<Vector>& vectors,
        std::vector<float>& distances)
    {
        validate_dimensions(query, vectors);

        distances.resize(vectors.size());

        const auto& backend = get_vector_backend();

        for (std::size_t i = 0; i < vectors.size(); ++i)
        {
            distances[i] =
                backend.l2_distance(query, vectors[i]);
        }
    }

    void batch_l2_distance(
        const Vector& query,
        const std::vector<Vector>& vectors,
        std::vector<float>& distances)
    {
        validate_dimensions(query, vectors);

        distances.resize(vectors.size());

        const auto& backend = get_vector_backend();

        const float* query_data = query.data();
        const std::size_t dimension = query.dimension();

        for (std::size_t i = 0; i < vectors.size(); ++i)
        {
            distances[i] =
                backend.raw_l2_distance(
                    query_data,
                    vectors[i].data(),
                    dimension
                );
        }
    }
}