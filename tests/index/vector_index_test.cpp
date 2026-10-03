#include "index/vector_index.hpp"

#include <gtest/gtest.h>
#include "embedding/bgeEmbedder.hpp"
using cortex::embedding::EmbeddedChunk;
using cortex::index::VectorIndex;

TEST(VectorIndexTest, AddsEmbeddedChunk)
{
    VectorIndex index(3);

    EmbeddedChunk chunk;

    chunk.id = "chunk_1";
    chunk.email_id = "email_1";
    chunk.thread_id = "thread_1";
    chunk.text = "Your interview is scheduled for Monday.";
    chunk.index = 0;
    chunk.embedding = {
        0.1f,
        0.2f,
        0.3f
    };

    const std::size_t id = index.add(chunk);
    const auto& record = index.get(id);

    EXPECT_EQ(
        record.metadata.at("chunk_id"),
        "chunk_1"
    );

    EXPECT_EQ(
        record.metadata.at("email_id"),
        "email_1"
    );

    EXPECT_EQ(
        record.metadata.at("thread_id"),
        "thread_1"
    );

    EXPECT_EQ(
        record.metadata.at("text"),
        "Your interview is scheduled for Monday."
    );

    EXPECT_EQ(
        record.metadata.at("chunk_index"),
        "0"
    );
    EXPECT_EQ(id, 0);
    EXPECT_EQ(index.size(), 1);
    EXPECT_EQ(index.dimension(), 3);
}

TEST(VectorIndexTest, SearchReturnsNearestChunk)
{
    VectorIndex index(3);

    EmbeddedChunk first;
    first.id = "chunk_1";
    first.email_id = "email_1";
    first.thread_id = "thread_1";
    first.text = "Your technical interview is scheduled for Monday.";
    first.index = 0;
    first.embedding = {
        1.0f,
        0.0f,
        0.0f
    };

    EmbeddedChunk second;
    second.id = "chunk_2";
    second.email_id = "email_2";
    second.thread_id = "thread_2";
    second.text = "The team meeting is scheduled for Friday.";
    second.index = 0;
    second.embedding = {
        0.0f,
        1.0f,
        0.0f
    };

    const std::size_t first_id = index.add(first);
    const std::size_t second_id = index.add(second);

    const std::vector<float> query{
        0.9f,
        0.1f,
        0.0f
    };

    const auto results =
        index.search(query, 1);

    ASSERT_EQ(results.size(), 1);

    EXPECT_EQ(results[0].id, first_id);

    EXPECT_EQ(
        results[0].record->metadata.at("chunk_id"),
        "chunk_1"
    );

    EXPECT_EQ(
        results[0].record->metadata.at("email_id"),
        "email_1"
    );

    EXPECT_EQ(
        results[0].record->metadata.at("text"),
        "Your technical interview is scheduled for Monday."
    );

    EXPECT_LT(
        results[0].distance,
        1.0f
    );
}

TEST(VectorIndexTest, SearchRejectsWrongDimension)
{
    VectorIndex index(3);

    EXPECT_THROW(
        index.search({1.0f, 2.0f}, 1),
        std::invalid_argument
    );
}

TEST(VectorIndexTest, SearchRejectsZeroK)
{
    VectorIndex index(3);

    EXPECT_THROW(
        index.search({1.0f, 2.0f, 3.0f}, 0),
        std::invalid_argument
    );
}

TEST(VectorIndexTest, BGERetrievesSemanticallyRelevantChunk)
{
    const std::string model_path = "models/bge-small-en-v1.5/onnx/model.onnx";

    const std::string vocab_path = "models/bge-small-en-v1.5/vocab.txt";

    cortex::embedding::BGEEmbedder embedder(
        model_path,
        vocab_path
    );

    VectorIndex index(embedder.dimension());

    EmbeddedChunk interview_chunk;

    interview_chunk.id = "chunk_interview";
    interview_chunk.email_id = "email_interview";
    interview_chunk.thread_id = "thread_interview";
    interview_chunk.text =
        "Your technical interview is scheduled for Monday.";
    interview_chunk.index = 0;
    interview_chunk.embedding =
        embedder.embed(interview_chunk.text);

    EmbeddedChunk meeting_chunk;

    meeting_chunk.id = "chunk_meeting";
    meeting_chunk.email_id = "email_meeting";
    meeting_chunk.thread_id = "thread_meeting";
    meeting_chunk.text =
        "The quarterly team meeting will happen on Friday.";
    meeting_chunk.index = 0;
    meeting_chunk.embedding =
        embedder.embed(meeting_chunk.text);

    index.add(interview_chunk);
    index.add(meeting_chunk);

    const std::vector<float> query_embedding =
        embedder.embed(
            "When is my technical interview?"
        );

    const auto results =
        index.search(query_embedding, 1);

    ASSERT_EQ(results.size(), 1);

    EXPECT_EQ(
        results[0].record->metadata.at("chunk_id"),
        "chunk_interview"
    );

    EXPECT_EQ(
        results[0].record->metadata.at("email_id"),
        "email_interview"
    );
}