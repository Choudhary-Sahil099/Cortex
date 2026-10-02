#include "embedding/bgeEmbedder.hpp"
#include <gtest/gtest.h>
#include <cmath>
TEST(BGEEmbedderTest, Produces384DimensionalEmbedding)
{
    cortex::embedding::BGEEmbedder embedder(
        "models/bge-small-en-v1.5/onnx/model.onnx",
        "models/bge-small-en-v1.5/vocab.txt"
    );

    const auto embedding =
        embedder.embed(
            "Your technical interview is scheduled for Monday."
        );

    EXPECT_EQ(
        embedder.dimension(),
        std::size_t{384}
    );

    EXPECT_EQ(
        embedding.size(),
        std::size_t{384}
    );
}

TEST(BGEEmbedderTest, ProducesNormalizedEmbedding)
{
    cortex::embedding::BGEEmbedder embedder(
        "models/bge-small-en-v1.5/onnx/model.onnx",
        "models/bge-small-en-v1.5/vocab.txt"
    );

    const auto embedding =
        embedder.embed(
            "Your technical interview is scheduled for Monday."
        );

    float squared_norm = 0.0f;

    for (const float value : embedding) {
        squared_norm += value * value;
    }

    const float norm =
        std::sqrt(squared_norm);

    EXPECT_NEAR(
        norm,
        1.0f,
        1e-5f
    );
}

TEST(BGEEmbedderTest, MatchesKnownReferenceEmbedding)
{
    cortex::embedding::BGEEmbedder embedder(
        "models/bge-small-en-v1.5/onnx/model.onnx",
        "models/bge-small-en-v1.5/vocab.txt"
    );

    const auto embedding =
        embedder.embed(
            "Your technical interview is scheduled for Monday."
        );

    ASSERT_EQ(
        embedding.size(),
        std::size_t{384}
    );

    const std::vector<float> expected = {
        -0.00718002f,
         0.0678497f,
        -0.0469356f,
        -0.0634190f,
        -0.00205826f,
         0.0333369f,
         0.0721765f,
         0.0274644f,
        -0.0670283f,
        -0.0356693f
    };

    for (std::size_t i = 0; i < expected.size(); ++i) {
        EXPECT_NEAR(
            embedding[i],
            expected[i],
            1e-5f
        );
    }
}