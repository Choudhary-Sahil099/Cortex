#include "vector/distance.hpp"
#include "vector/dot_product.hpp"
#include "vector/backend.hpp"
#include "vector/batch_distance.hpp"
#include "vector/batch_distance_simd.hpp"
#include "simd_kernels.hpp"
#include <stdexcept>
namespace cortex::vector {
namespace {
const auto backend = select_distance_backend();
bool avx2_available() {
#ifdef _MSC_VER
    // /arch:AVX2 permits FMA instructions, even in non-FMA functions.
    return backend == DistanceBackend::AVX2_FMA;
#else
    return backend != DistanceBackend::Scalar;
#endif
}
bool fma_available() { return backend == DistanceBackend::AVX2_FMA; }
}
    float l2_distance(
        const Vector& a,
        const Vector& b)
    {
        return get_vector_backend().l2_distance(a, b);
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
    float dot_product(
        const Vector& a,
        const Vector& b)
    {
        return get_vector_backend().dot_product(a, b);
    }
float l2_distance_avx2(const Vector& a, const Vector& b) { return avx2_available() ? l2_distance_avx2_impl(a,b) : l2_distance_scalar(a,b); }
float l2_distance_avx2_fma(const Vector& a, const Vector& b) { return fma_available() ? l2_distance_avx2_fma_impl(a,b) : l2_distance_avx2(a,b); }
float l2_distance_avx2(const float* a, const float* b, std::size_t n) { return avx2_available() ? l2_distance_avx2_impl(a,b,n) : l2_distance_scalar(a,b,n); }
float l2_distance_avx2_fma(const float* a, const float* b, std::size_t n) { return fma_available() ? l2_distance_avx2_fma_impl(a,b,n) : l2_distance_avx2(a,b,n); }
float dot_product_avx2(const Vector& a, const Vector& b) { return avx2_available() ? dot_product_avx2_impl(a,b) : dot_product_scalar(a,b); }
float dot_product_avx2_fma(const Vector& a, const Vector& b) { return fma_available() ? dot_product_avx2_fma_impl(a,b) : dot_product_avx2(a,b); }
void batch_l2_distance_avx2(const Vector& q,const std::vector<Vector>& vs,std::vector<float>& out) { if(avx2_available()) batch_l2_distance_avx2_impl(q,vs,out); else batch_l2_distance_scalar(q,vs,out); }
void batch_l2_distance_avx2_fma(const Vector& q,const std::vector<Vector>& vs,std::vector<float>& out) { if(fma_available()) batch_l2_distance_avx2_fma_impl(q,vs,out); else batch_l2_distance_avx2(q,vs,out); }
}
