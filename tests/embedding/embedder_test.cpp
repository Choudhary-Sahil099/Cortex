#include "index/vector_index.hpp"
#include "embedding/embeded_chunk.hpp"
#include "embedding/embedder.hpp"
#include "retrival/retriever.hpp"
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


//test retrieval
TEST(RetrieverTest, SearchesUsingEmbeddedQuery)
{
    TestEmbedder embedder;
    cortex::index::VectorIndex index(3);
    cortex::embedding::EmbeddedChunk chunk;
    
    chunk.id = "chunk_1";
    chunk.email_id = "email_1";
    chunk.thread_id = "thread_1";
    chunk.text = "Technical interview scheduled Monday.";
    chunk.index = 0;
    chunk.embedding = {
        0.1f,
        0.2f,
        0.3f
    };

    index.add(chunk);

    cortex::retrieval::Retriever retriever(
        embedder,
        index
    );

    const auto results =
        retriever.search(
            "When is my interview?",
            1
        );

    ASSERT_EQ(results.size(), 1);

    EXPECT_EQ(
        results[0].record->metadata.at("chunk_id"),
        "chunk_1"
    );
}

TEST(RetrieverTest, RejectsNegativeDistance)
{
    TestEmbedder embedder;
    cortex::index::VectorIndex index(3);

    cortex::retrieval::Retriever retriever(
        embedder,
        index
    );

    EXPECT_THROW(
        retriever.search(
            "test",
            1,
            -0.1f
        ),
        std::invalid_argument
    );
}

TEST(RetrieverTest, FiltersResultsByDistance)
{
    TestEmbedder embedder;

    cortex::index::VectorIndex index(3);

    cortex::embedding::EmbeddedChunk close_chunk;

    close_chunk.id = "close";
    close_chunk.email_id = "email_close";
    close_chunk.thread_id = "thread_close";
    close_chunk.text = "Close result.";
    close_chunk.index = 0;
    close_chunk.embedding = {
        0.1f,
        0.2f,
        0.3f
    };

    cortex::embedding::EmbeddedChunk far_chunk;

    far_chunk.id = "far";
    far_chunk.email_id = "email_far";
    far_chunk.thread_id = "thread_far";
    far_chunk.text = "Far result.";
    far_chunk.index = 0;
    far_chunk.embedding = {
        10.0f,
        10.0f,
        10.0f
    };

    index.add(close_chunk);
    index.add(far_chunk);

    cortex::retrieval::Retriever retriever(
        embedder,
        index
    );

    const auto results =
        retriever.search(
            "test",
            2,
            1.0f
        );

    ASSERT_EQ(results.size(), 1);

    EXPECT_EQ(
        results[0].record->metadata.at("chunk_id"),
        "close"
    );
}