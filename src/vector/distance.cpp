#include "vector/distance.hpp"
#include "vector/backend.hpp"

#include <cmath>
#include <stdexcept>
#include <immintrin.h>

namespace cortex::vector
{


    float l2_distance(
        const Vector& a,
        const Vector& b)
    {
        return get_vector_backend().l2_distance(a, b);
    }


    float l2_distance_avx2(
        const Vector& a,
        const Vector& b)
    {
        if (a.dimension() != b.dimension())
        {
            throw std::invalid_argument(
                "Vector must have same dimension"
            );
        }

        const float* a_data = a.data();
        const float* b_data = b.data();

        std::size_t i = 0;

        __m256 sum_vector =
            _mm256_setzero_ps();


        for (; i + 8 <= a.dimension(); i += 8)
        {
            __m256 va =
                _mm256_load_ps(a_data + i);

            __m256 vb =
                _mm256_load_ps(b_data + i);

            __m256 difference =
                _mm256_sub_ps(va, vb);

            __m256 squared =
                _mm256_mul_ps(
                    difference,
                    difference
                );

            sum_vector =
                _mm256_add_ps(
                    sum_vector,
                    squared
                );
        }

        alignas(32) float values[8];

        _mm256_store_ps(
            values,
            sum_vector
        );

        float sum = 0.0f;

        for (int j = 0; j < 8; ++j)
        {
            sum += values[j];
        }

        for (; i < a.dimension(); ++i)
        {
            float difference =
                a[i] - b[i];

            sum +=
                difference * difference;
        }

        return std::sqrt(sum);
    }


    float l2_distance_avx2_fma(
        const Vector& a,
        const Vector& b)
    {
        if (a.dimension() != b.dimension())
        {
            throw std::invalid_argument(
                "Vectors must have same dimensions"
            );
        }

        const std::size_t n =
            a.dimension();

        const float* x =
            a.data();

        const float* y =
            b.data();


        __m256 sum0 =
            _mm256_setzero_ps();

        __m256 sum1 =
            _mm256_setzero_ps();

        __m256 sum2 =
            _mm256_setzero_ps();

        __m256 sum3 =
            _mm256_setzero_ps();

        std::size_t i = 0;


        for (; i + 32 <= n; i += 32)
        {
            __m256 x0 =
                _mm256_load_ps(x + i);

            __m256 y0 =
                _mm256_load_ps(y + i);


            __m256 x1 =
                _mm256_load_ps(x + i + 8);

            __m256 y1 =
                _mm256_load_ps(y + i + 8);


            __m256 x2 =
                _mm256_load_ps(x + i + 16);

            __m256 y2 =
                _mm256_load_ps(y + i + 16);


            __m256 x3 =
                _mm256_load_ps(x + i + 24);

            __m256 y3 =
                _mm256_load_ps(y + i + 24);


            __m256 d0 =
                _mm256_sub_ps(
                    x0,
                    y0
                );

            __m256 d1 =
                _mm256_sub_ps(
                    x1,
                    y1
                );

            __m256 d2 =
                _mm256_sub_ps(
                    x2,
                    y2
                );

            __m256 d3 =
                _mm256_sub_ps(
                    x3,
                    y3
                );
   
            sum0 =
                _mm256_fmadd_ps(
                    d0,
                    d0,
                    sum0
                );

            sum1 =
                _mm256_fmadd_ps(
                    d1,
                    d1,
                    sum1
                );

            sum2 =
                _mm256_fmadd_ps(
                    d2,
                    d2,
                    sum2
                );

            sum3 =
                _mm256_fmadd_ps(
                    d3,
                    d3,
                    sum3
                );
        }

        __m256 sum =
            _mm256_add_ps(
                sum0,
                sum1
            );

        sum =
            _mm256_add_ps(
                sum,
                sum2
            );

        sum =
            _mm256_add_ps(
                sum,
                sum3
            );

        alignas(32) float values[8];

        _mm256_store_ps(
            values,
            sum
        );

        float result = 0.0f;

        for (int j = 0; j < 8; ++j)
        {
            result += values[j];
        }

        for (; i < n; ++i)
        {
            float difference =
                x[i] - y[i];

            result +=
                difference * difference;
        }

        return std::sqrt(result);
    }
    float l2_distance_avx2(
        const float* x,
        const float* y,
        std::size_t n)
    {
        __m256 sum_vector =
            _mm256_setzero_ps();

        std::size_t i = 0;

        for (; i + 8 <= n; i += 8)
        {
            __m256 a =
                _mm256_loadu_ps(x + i);

            __m256 b =
                _mm256_loadu_ps(y + i);

            __m256 diff =
                _mm256_sub_ps(a, b);

            __m256 squared =
                _mm256_mul_ps(diff, diff);

            sum_vector =
                _mm256_add_ps(
                    sum_vector,
                    squared
                );
        }

        alignas(32) float values[8];

        _mm256_store_ps(
            values,
            sum_vector
        );

        float sum = 0.0f;

        for (int j = 0; j < 8; ++j)
        {
            sum += values[j];
        }

        for (; i < n; ++i)
        {
            const float diff =
                x[i] - y[i];

            sum += diff * diff;
        }

        return std::sqrt(sum);
    }
    float l2_distance_avx2_fma(
        const float* x,
        const float* y,
        std::size_t n)
    {
        __m256 sum0 = _mm256_setzero_ps();
        __m256 sum1 = _mm256_setzero_ps();
        __m256 sum2 = _mm256_setzero_ps();
        __m256 sum3 = _mm256_setzero_ps();

        std::size_t i = 0;

        for (; i + 32 <= n; i += 32)
        {
            __m256 x0 = _mm256_loadu_ps(x + i);
            __m256 y0 = _mm256_loadu_ps(y + i);

            __m256 x1 = _mm256_loadu_ps(x + i + 8);
            __m256 y1 = _mm256_loadu_ps(y + i + 8);

            __m256 x2 = _mm256_loadu_ps(x + i + 16);
            __m256 y2 = _mm256_loadu_ps(y + i + 16);

            __m256 x3 = _mm256_loadu_ps(x + i + 24);
            __m256 y3 = _mm256_loadu_ps(y + i + 24);

            __m256 d0 =
                _mm256_sub_ps(x0, y0);

            __m256 d1 =
                _mm256_sub_ps(x1, y1);

            __m256 d2 =
                _mm256_sub_ps(x2, y2);

            __m256 d3 =
                _mm256_sub_ps(x3, y3);

            sum0 =
                _mm256_fmadd_ps(d0, d0, sum0);

            sum1 =
                _mm256_fmadd_ps(d1, d1, sum1);

            sum2 =
                _mm256_fmadd_ps(d2, d2, sum2);

            sum3 =
                _mm256_fmadd_ps(d3, d3, sum3);
        }

        __m256 sum =
            _mm256_add_ps(sum0, sum1);

        sum =
            _mm256_add_ps(sum, sum2);

        sum =
            _mm256_add_ps(sum, sum3);

        alignas(32) float values[8];

        _mm256_store_ps(values, sum);

        float result = 0.0f;

        for (int j = 0; j < 8; ++j)
        {
            result += values[j];
        }

        for (; i < n; ++i)
        {
            const float diff =
                x[i] - y[i];

            result += diff * diff;
        }

        return std::sqrt(result);
    }

    float l2_distance(
        const float* a,
        const float* b,
        std::size_t dimension)
    {
        const auto& backend = get_vector_backend();

        return backend.raw_l2_distance(
            a,
            b,
            dimension
        );
    }
}