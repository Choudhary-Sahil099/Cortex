#include "core/cpu_features.hpp"

#if defined(_MSC_VER)
#include <intrin.h>
#endif

namespace cortex::core
{
    CPUFeatures detect_cpu_features()
    {
        CPUFeatures features{};

#if defined(_MSC_VER)

        int cpu_info[4] = {};
        __cpuid(cpu_info, 1);
        features.fma =
            (cpu_info[2] & (1 << 12)) != 0;
        __cpuidex(cpu_info, 7, 0);

        features.avx2 =
            (cpu_info[1] & (1 << 5)) != 0;

#elif defined(__GNUC__) || defined(__clang__)

#if defined(__x86_64__) || defined(__i386__)

        features.avx2 =
            __builtin_cpu_supports("avx2");

        features.fma =
            __builtin_cpu_supports("fma");

#endif

#endif

        return features;
    }
}