#include "embedding/bert_normalizer.hpp"

#include <gtest/gtest.h>

TEST(
    BertNormalizerTest,
    ConvertsTextToLowercase
)
{
    cortex::embedding::BertNormalizer normalizer;

    const auto result =
        normalizer.normalize(
            "Your Technical Interview"
        );

    EXPECT_EQ(
        result,
        "your technical interview"
    );
}

TEST(
    BertNormalizerTest,
    PreservesAlreadyLowercaseText
)
{
    cortex::embedding::BertNormalizer normalizer;

    const auto result =
        normalizer.normalize(
            "hello world"
        );

    EXPECT_EQ(
        result,
        "hello world"
    );
}