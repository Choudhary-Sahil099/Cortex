#include "index/vector_index.hpp"
#include "embedding/embeded_chunk.hpp"
#include "embedding/embedder.hpp"
#include "retrival/retriever.hpp"
#include <gtest/gtest.h>
class TestEmbedder : public cortex::embedding::Embedder
{
public:
    std::vector<float> embed(
        const std::string &text) const override
    {
        return {
            0.1f,
            0.2f,
            0.3f};
    }

    std::size_t dimension() const override
    {
        return 3;
    }
};

class ControlledEmbedder : public cortex::embedding::Embedder
{
public:
    std::vector<float> embed(const std::string &text) const override
    {
        if (text == "technical interview schedule date time")
        {
            return {1.0f, 0.0f, 0.0f};
        }

        return {0.0f, 1.0f, 0.0f};
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
        std::size_t{3});
}

TEST(EmbedderTest, ProducesVector)
{
    TestEmbedder embedder;

    const auto vector =
        embedder.embed(
            "Hello world");

    ASSERT_EQ(
        vector.size(),
        std::size_t{3});

    EXPECT_FLOAT_EQ(vector[0], 0.1f);
    EXPECT_FLOAT_EQ(vector[1], 0.2f);
    EXPECT_FLOAT_EQ(vector[2], 0.3f);
}

// test retrieval
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
        0.3f};

    index.add(chunk);

    cortex::retrieval::Retriever retriever(
        embedder,
        index);

    const auto results =
        retriever.search(
            "When is my interview?",
            1);

    ASSERT_EQ(results.size(), 1);

    EXPECT_EQ(
        results[0].record->metadata.at("chunk_id"),
        "chunk_1");
}

TEST(RetrieverTest, RejectsNegativeDistance)
{
    TestEmbedder embedder;
    cortex::index::VectorIndex index(3);

    cortex::retrieval::Retriever retriever(
        embedder,
        index);

    EXPECT_THROW(
        retriever.search(
            "test",
            1,
            -0.1f),
        std::invalid_argument);
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
        0.3f};

    cortex::embedding::EmbeddedChunk far_chunk;

    far_chunk.id = "far";
    far_chunk.email_id = "email_far";
    far_chunk.thread_id = "thread_far";
    far_chunk.text = "Far result.";
    far_chunk.index = 0;
    far_chunk.embedding = {
        10.0f,
        10.0f,
        10.0f};

    index.add(close_chunk);
    index.add(far_chunk);

    cortex::retrieval::Retriever retriever(
        embedder,
        index);

    const auto results =
        retriever.search(
            "test",
            2,
            1.0f);

    ASSERT_EQ(results.size(), 1);

    EXPECT_EQ(
        results[0].record->metadata.at("chunk_id"),
        "close");
}

TEST(RetrieverTest, HybridSearchReturnsMatchingChunk)
{
    TestEmbedder embedder;
    cortex::index::VectorIndex index(3);

    cortex::embedding::EmbeddedChunk chunk;
    chunk.id = "chunk_interview";
    chunk.email_id = "email_interview";
    chunk.thread_id = "thread_interview";
    chunk.text = "Your technical interview is scheduled for Monday.";
    chunk.index = 0;
    chunk.embedding = {0.1f, 0.2f, 0.3f};

    index.add(chunk);

    cortex::retrieval::Retriever retriever(embedder, index);

    const auto results = retriever.hybridSearch("technical interview", 5);

    ASSERT_EQ(results.size(), 1);
    EXPECT_EQ(
        results[0].record->metadata.at("chunk_id"),
        "chunk_interview");
}

TEST(RetrieverTest, HybridSearchDoesNotReturnDuplicateChunks)
{
    TestEmbedder embedder;
    cortex::index::VectorIndex index(3);

    cortex::embedding::EmbeddedChunk first;
    first.id = "chunk_one";
    first.email_id = "email_one";
    first.thread_id = "thread_one";
    first.text = "Technical interview scheduled for Monday.";
    first.index = 0;
    first.embedding = {0.1f, 0.2f, 0.3f};

    cortex::embedding::EmbeddedChunk second;
    second.id = "chunk_two";
    second.email_id = "email_two";
    second.thread_id = "thread_two";
    second.text = "Interview preparation notes.";
    second.index = 0;
    second.embedding = {0.2f, 0.3f, 0.4f};

    index.add(first);
    index.add(second);

    cortex::retrieval::Retriever retriever(embedder, index);

    const auto results = retriever.hybridSearch("technical interview", 5);

    ASSERT_EQ(results.size(), 2);

    EXPECT_NE(
        results[0].record->metadata.at("chunk_id"),
        results[1].record->metadata.at("chunk_id"));
}


TEST(RetrieverTest, HybridSearchFusesSemanticAndLexicalRankings)
{
    ControlledEmbedder embedder;
    cortex::index::VectorIndex index(3);

    const auto add_chunk = [&index](
        const std::string& id,
        const std::string& text,
        std::vector<float> embedding)
    {
        cortex::embedding::EmbeddedChunk chunk;
        chunk.id = id;
        chunk.email_id = id;
        chunk.thread_id = id;
        chunk.text = text;
        chunk.index = 0;
        chunk.embedding = std::move(embedding);

        index.add(chunk);
    };

    add_chunk(
        "chunk-a",
        "technical interview",
        {0.0f, 1.0f, 0.0f});

    add_chunk(
        "chunk-b",
        "technical interview schedule date",
        {0.0f, 0.0f, 1.0f});

    add_chunk(
        "chunk-c",
        "technical interview schedule date time",
        {1.0f, 0.0f, 0.0f});

    add_chunk(
        "chunk-d",
        "technical interview schedule",
        {0.8f, 0.6f, 0.0f});

    cortex::retrieval::Retriever retriever(embedder, index);

    const auto lexical_results =
        index.lexicalSearch(
            "technical interview schedule date time",
            4);

    const auto results =
        retriever.hybridSearch("technical interview", 4);

    ASSERT_EQ(results.size(), 4u);

    EXPECT_EQ(results[0].record->metadata.at("chunk_id"), "chunk-c");
    EXPECT_EQ(results[1].record->metadata.at("chunk_id"), "chunk-d");
    EXPECT_EQ(results[2].record->metadata.at("chunk_id"), "chunk-b");
    EXPECT_EQ(results[3].record->metadata.at("chunk_id"), "chunk-a");
}
