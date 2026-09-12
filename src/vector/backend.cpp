#include "vector/backend.hpp"

#include "core/cpu_features.hpp"
#include "vector/distance.hpp"
#include "vector/dot_product.hpp"

namespace cortex::vector
{
    DistanceBackend select_distance_backend()
    {
        const auto features =
            cortex::core::detect_cpu_features();

        if (features.avx2 && features.fma)
            return DistanceBackend::AVX2_FMA;

        if (features.avx2)
            return DistanceBackend::AVX2;

        return DistanceBackend::Scalar;
    }

    const VectorBackend& get_vector_backend()
    {
        static const VectorBackend backend = []()
            {
                switch (select_distance_backend())
                {
                case DistanceBackend::AVX2_FMA:
                    return VectorBackend{
                        l2_distance_avx2_fma,
                        dot_product_avx2_fma,
                        l2_distance_avx2_fma
                    };

                case DistanceBackend::AVX2:
                    return VectorBackend{
                        l2_distance_avx2,
                        dot_product_avx2,
                        l2_distance_avx2
                    };

                case DistanceBackend::Scalar:
                default:
                    return VectorBackend{
                        l2_distance_scalar,
                        dot_product_scalar,
                        l2_distance_scalar
                    };
                }
            }();

        return backend;
    }
    const char* backend_name(DistanceBackend backend)
    {
        switch (backend)
        {
        case DistanceBackend::AVX2_FMA:
            return "AVX2 + FMA";

        case DistanceBackend::AVX2:
            return "AVX2";

        case DistanceBackend::Scalar:
        default:
            return "Scalar";
        }
    }
}