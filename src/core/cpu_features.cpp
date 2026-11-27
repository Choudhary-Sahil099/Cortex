#include "core/cpu_features.hpp"

#if defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))
#include <intrin.h>
#endif

namespace cortex::core
{
    CPUFeatures detect_cpu_features()
    {
        CPUFeatures features{};

#if defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))

        int cpu_info[4] = {};
        __cpuid(cpu_info, 0);
        const int max_leaf = cpu_info[0];
        if (max_leaf < 1) return features;
        __cpuid(cpu_info, 1);
        const bool osxsave = (cpu_info[2] & (1 << 27)) != 0;
        const bool avx = (cpu_info[2] & (1 << 28)) != 0;
        if (!osxsave || !avx || (_xgetbv(0) & 6) != 6) return features;
        features.fma =
            (cpu_info[2] & (1 << 12)) != 0;
        if (max_leaf < 7) return features;
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