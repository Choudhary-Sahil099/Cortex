#include "index/hnsw_level_generator.hpp"

#include <gtest/gtest.h>

using cortex::index::HNSWLevelGenerator;

TEST(HNSWLevelGeneratorTest, GeneratesValidLevels) {

    HNSWLevelGenerator generator;

    for (int i = 0; i < 1000; ++i) {
        const std::size_t level = generator.generate();

        EXPECT_GE(level, 0u);
    }
}

TEST(HNSWLevelGeneratorTest, ProducesMostlyLowLevels) {

    HNSWLevelGenerator generator;

    std::size_t high_levels = 0;

    constexpr std::size_t samples = 10000;

    for (std::size_t i = 0; i < samples; ++i) {

        const std::size_t level = generator.generate();

        if (level >= 5) {
            ++high_levels;
        }
    }

    EXPECT_LT(high_levels, samples / 10);
}