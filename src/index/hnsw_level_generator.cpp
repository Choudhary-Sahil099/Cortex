#include "index/hnsw_level_generator.hpp"

#include <cmath>
#include <limits>
#include <stdexcept>

namespace cortex::index {

    HNSWLevelGenerator::HNSWLevelGenerator(
        double level_multiplier
    )
        : level_multiplier_(level_multiplier),
        generator_(std::random_device{}()),
        distribution_(0.0, 1.0) {

        if (level_multiplier <= 0.0) {
            throw std::invalid_argument(
                "Level multiplier must be greater than zero"
            );
        }
    }

    std::size_t HNSWLevelGenerator::generate() {

        double uniform = distribution_(generator_);
        
        //conditon for the log 0
        if (uniform == 0.0) {
            uniform = std::numeric_limits<double>::min();
        }

        const double level =
            -std::log(uniform) * level_multiplier_;  // using the logarithm to generate level

        return static_cast<std::size_t>(level);
    }

} 