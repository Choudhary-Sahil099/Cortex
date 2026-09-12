#pragma once

namespace cortex::core
{
    struct CPUFeatures
    {
        
        bool avx2 = false;
        bool fma = false;
    };

    CPUFeatures detect_cpu_features();
}