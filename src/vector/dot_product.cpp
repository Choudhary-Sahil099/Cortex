#include "vector/dot_product.hpp"
#include "vector/backend.hpp"
#include <immintrin.h>
#include <stdexcept>

namespace cortex::vector
{

    float dot_product_scalar(
        const Vector& a,
        const Vector& b)
    {
        if (a.dimension() != b.dimension())
        {
            throw std::invalid_argument(
                "Vectors must have same dimensions"
            );
        }

        float sum = 0.0f;

        for (std::size_t i = 0; i < a.dimension(); ++i)
        {
            sum += a[i] * b[i];
        }

        return sum;
    }

    float dot_product_avx2(
        const Vector& a,
        const Vector& b)
    {
        if (a.dimension() != b.dimension())
        {
            throw std::invalid_argument(
                "Vectors must have same dimensions"
            );
        }

        const std::size_t n = a.dimension();

        const float* x = a.data();
        const float* y = b.data();

        __m256 sum_vector = _mm256_setzero_ps();

        std::size_t i = 0;

        for (; i + 8 <= n; i += 8)
        {
            __m256 va = _mm256_loadu_ps(x + i);
            __m256 vb = _mm256_loadu_ps(y + i);

            __m256 product =
                _mm256_mul_ps(va, vb);

            sum_vector =
                _mm256_add_ps(sum_vector, product);
        }

        alignas(32) float values[8];

        _mm256_store_ps(values, sum_vector);

        float sum = 0.0f;

        for (int j = 0; j < 8; ++j)
        {
            sum += values[j];
        }
        for (; i < n; ++i)
        {
            sum += x[i] * y[i];
        }

        return sum;
    }

    float dot_product_avx2_fma(
        const Vector& a,
        const Vector& b)
    {
        if (a.dimension() != b.dimension())
        {
            throw std::invalid_argument(
                "Vectors must have same dimensions"
            );
        }

        const std::size_t n = a.dimension();

        const float* x = a.data();
        const float* y = b.data();

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

            sum0 = _mm256_fmadd_ps(x0, y0, sum0);
            sum1 = _mm256_fmadd_ps(x1, y1, sum1);
            sum2 = _mm256_fmadd_ps(x2, y2, sum2);
            sum3 = _mm256_fmadd_ps(x3, y3, sum3);
        }

        __m256 sum = _mm256_add_ps(sum0, sum1);
        sum = _mm256_add_ps(sum, sum2);
        sum = _mm256_add_ps(sum, sum3);

        alignas(32) float values[8];

        _mm256_store_ps(values, sum);

        float result = 0.0f;

        for (int j = 0; j < 8; ++j)
        {
            result += values[j];
        }

        for (; i < n; ++i)
        {
            result += x[i] * y[i];
        }

        return result;
    }
    float dot_product(
        const Vector& a,
        const Vector& b)
    {
        return get_vector_backend().dot_product(a, b);
    }
}
