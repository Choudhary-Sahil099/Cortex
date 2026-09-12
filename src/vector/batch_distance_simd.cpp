#include "vector/batch_distance_simd.hpp"

#include <immintrin.h>
#include <cmath>
#include <stdexcept>

namespace cortex::vector
{
    namespace
    {
        void validate_batch(
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

        float horizontal_sum(__m256 value)
        {
            __m128 low =
                _mm256_castps256_ps128(value);

            __m128 high =
                _mm256_extractf128_ps(value, 1);

            __m128 sum =
                _mm_add_ps(low, high);

            sum =
                _mm_hadd_ps(sum, sum);

            sum =
                _mm_hadd_ps(sum, sum);

            return _mm_cvtss_f32(sum);
        }
    }


    void batch_l2_distance_avx2(
        const Vector& query,
        const std::vector<Vector>& vectors,
        std::vector<float>& distances)
    {
        validate_batch(query, vectors);

        distances.resize(vectors.size());

        const std::size_t n = query.dimension();
        const float* q = query.data();

        for (std::size_t v = 0; v < vectors.size(); ++v)
        {
            const float* x = vectors[v].data();

            __m256 sum0 = _mm256_setzero_ps();
            __m256 sum1 = _mm256_setzero_ps();
            __m256 sum2 = _mm256_setzero_ps();
            __m256 sum3 = _mm256_setzero_ps();

            std::size_t i = 0;

            for (; i + 32 <= n; i += 32)
            {
                __m256 q0 = _mm256_loadu_ps(q + i);
                __m256 x0 = _mm256_loadu_ps(x + i);

                __m256 q1 = _mm256_loadu_ps(q + i + 8);
                __m256 x1 = _mm256_loadu_ps(x + i + 8);

                __m256 q2 = _mm256_loadu_ps(q + i + 16);
                __m256 x2 = _mm256_loadu_ps(x + i + 16);

                __m256 q3 = _mm256_loadu_ps(q + i + 24);
                __m256 x3 = _mm256_loadu_ps(x + i + 24);

                __m256 d0 = _mm256_sub_ps(q0, x0);
                __m256 d1 = _mm256_sub_ps(q1, x1);
                __m256 d2 = _mm256_sub_ps(q2, x2);
                __m256 d3 = _mm256_sub_ps(q3, x3);

                sum0 = _mm256_add_ps(
                    sum0,
                    _mm256_mul_ps(d0, d0)
                );

                sum1 = _mm256_add_ps(
                    sum1,
                    _mm256_mul_ps(d1, d1)
                );

                sum2 = _mm256_add_ps(
                    sum2,
                    _mm256_mul_ps(d2, d2)
                );

                sum3 = _mm256_add_ps(
                    sum3,
                    _mm256_mul_ps(d3, d3)
                );
            }

            __m256 sum =
                _mm256_add_ps(sum0, sum1);

            sum =
                _mm256_add_ps(sum, sum2);

            sum =
                _mm256_add_ps(sum, sum3);

            float result = horizontal_sum(sum);

            for (; i < n; ++i)
            {
                const float d = q[i] - x[i];
                result += d * d;
            }

            distances[v] = std::sqrt(result);
        }
    }


    void batch_l2_distance_avx2_fma(
        const Vector& query,
        const std::vector<Vector>& vectors,
        std::vector<float>& distances)
    {
        validate_batch(query, vectors);

        distances.resize(vectors.size());

        const std::size_t n = query.dimension();
        const float* q = query.data();

        for (std::size_t v = 0; v < vectors.size(); ++v)
        {
            const float* x = vectors[v].data();

            __m256 sum0 = _mm256_setzero_ps();
            __m256 sum1 = _mm256_setzero_ps();
            __m256 sum2 = _mm256_setzero_ps();
            __m256 sum3 = _mm256_setzero_ps();

            std::size_t i = 0;

            for (; i + 32 <= n; i += 32)
            {
                __m256 q0 = _mm256_loadu_ps(q + i);
                __m256 x0 = _mm256_loadu_ps(x + i);

                __m256 q1 = _mm256_loadu_ps(q + i + 8);
                __m256 x1 = _mm256_loadu_ps(x + i + 8);

                __m256 q2 = _mm256_loadu_ps(q + i + 16);
                __m256 x2 = _mm256_loadu_ps(x + i + 16);

                __m256 q3 = _mm256_loadu_ps(q + i + 24);
                __m256 x3 = _mm256_loadu_ps(x + i + 24);

                __m256 d0 = _mm256_sub_ps(q0, x0);
                __m256 d1 = _mm256_sub_ps(q1, x1);
                __m256 d2 = _mm256_sub_ps(q2, x2);
                __m256 d3 = _mm256_sub_ps(q3, x3);

                sum0 = _mm256_fmadd_ps(
                    d0, d0, sum0
                );

                sum1 = _mm256_fmadd_ps(
                    d1, d1, sum1
                );

                sum2 = _mm256_fmadd_ps(
                    d2, d2, sum2
                );

                sum3 = _mm256_fmadd_ps(
                    d3, d3, sum3
                );
            }

            __m256 sum =
                _mm256_add_ps(sum0, sum1);

            sum =
                _mm256_add_ps(sum, sum2);

            sum =
                _mm256_add_ps(sum, sum3);

            float result = horizontal_sum(sum);

            for (; i < n; ++i)
            {
                const float d = q[i] - x[i];
                result += d * d;
            }

            distances[v] = std::sqrt(result);
        }
    }
}