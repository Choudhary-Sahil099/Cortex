#include "embedding/embedder.hpp"
#include <gtest/gtest.h>

class TestEmbedder : public cortex::embedding::Embedder {
public:

    std::vector<float> embed(
        const std::string& text
    ) const override
    {
        return {
            0.1f,
            0.2f,
            0.3f
        };
    }

    std::size_t dimension() const override
    {
        return 3;
    }
};

TEST(EmbedderTest, ReturnsExpectedDimension)
{
    TestEmbedder embedder;

    EXPECT_EQ(
        embedder.dimension(),
        std::size_t{ 3 }
    );
}

TEST(EmbedderTest, ProducesVector)
{
    TestEmbedder embedder;

    const auto vector =
        embedder.embed(
            "Hello world"
        );

    ASSERT_EQ(
        vector.size(),
        std::size_t{ 3 }
    );

    EXPECT_FLOAT_EQ(vector[0], 0.1f);
    EXPECT_FLOAT_EQ(vector[1], 0.2f);
    EXPECT_FLOAT_EQ(vector[2], 0.3f);
}