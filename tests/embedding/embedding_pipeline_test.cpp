#include "embedding/embedding_pipeline.hpp"

#include <gtest/gtest.h>

namespace
{
    class TestEmbedder
        : public cortex::embedding::Embedder
    {
    public:

        std::vector<float> embed(
            const std::string&
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
}

TEST(EmbeddingPipelineTest, EmbedsChunk)
{
    TestEmbedder embedder;

    cortex::embedding::EmbeddingPipeline pipeline(
        embedder
    );

    cortex::email::emailChunk chunk;

    chunk.id = "email1_0";
    chunk.email_id = "email1";
    chunk.thread_id = "thread1";
    chunk.text = "Hello world";
    chunk.index = 0;

    auto result = pipeline.embed(chunk);

    EXPECT_EQ(result.id, "email1_0");
    EXPECT_EQ(result.email_id, "email1");
    EXPECT_EQ(result.thread_id, "thread1");
    EXPECT_EQ(result.text, "Hello world");
    EXPECT_EQ(result.index, 0);

    ASSERT_EQ(result.embedding.size(), 3);

    EXPECT_FLOAT_EQ(result.embedding[0], 0.1f);
    EXPECT_FLOAT_EQ(result.embedding[1], 0.2f);
    EXPECT_FLOAT_EQ(result.embedding[2], 0.3f);
}

TEST(EmbeddingPipelineTest, EmbedsMultipleChunks)
{
    TestEmbedder embedder;

    cortex::embedding::EmbeddingPipeline pipeline(
        embedder
    );

    std::vector<cortex::email::emailChunk> chunks(3);

    chunks[0].id = "email1_0";
    chunks[0].text = "First";

    chunks[1].id = "email1_1";
    chunks[1].text = "Second";

    chunks[2].id = "email1_2";
    chunks[2].text = "Third";

    auto results = pipeline.embed(chunks);

    ASSERT_EQ(results.size(), 3);

    EXPECT_EQ(results[0].id, "email1_0");
    EXPECT_EQ(results[1].id, "email1_1");
    EXPECT_EQ(results[2].id, "email1_2");
}