#pragma once

#include <cstddef>
#include <random>

namespace cortex::index {

    //Info -> Rand Level Generation for HNSW -> randomly selects minimum level
    class HNSWLevelGenerator {
    public:
        explicit HNSWLevelGenerator(
            double level_multiplier = 1.0
        );

        std::size_t generate();

    private:
        double level_multiplier_;

        std::mt19937_64 generator_;

        std::uniform_real_distribution<double> distribution_;
    };

}