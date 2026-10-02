#include "embedding/bgeEmbedder.hpp"
#include <gtest/gtest.h>

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