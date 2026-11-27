#pragma once
#include "vector/vector.hpp"
#include <vector>
namespace cortex::vector {
float l2_distance_avx2_impl(const Vector&, const Vector&);
float l2_distance_avx2_fma_impl(const Vector&, const Vector&);
float l2_distance_avx2_impl(const float*, const float*, std::size_t);
float l2_distance_avx2_fma_impl(const float*, const float*, std::size_t);
float dot_product_avx2_impl(const Vector&, const Vector&);
float dot_product_avx2_fma_impl(const Vector&, const Vector&);
void batch_l2_distance_avx2_impl(const Vector&, const std::vector<Vector>&, std::vector<float>&);
void batch_l2_distance_avx2_fma_impl(const Vector&, const std::vector<Vector>&, std::vector<float>&);
}
