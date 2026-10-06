#include "index/vector_index.hpp"
#include "embedding/embedding_pipeline.hpp"

#include "email/email_parserer.hpp"
#include "email/email_cleaner.hpp"
#include "email/email_chunker.hpp"

#include "ingestion/email_indexer.hpp"
#include <gtest/gtest.h>
#include <stdexcept>
#include <string>
#include <vector>
#include "embedding/bgeEmbedder.hpp"
using cortex::embedding::EmbeddedChunk;
using cortex::index::VectorIndex;

class FailingEmbedder : public cortex::embedding::Embedder
{
public:
    std::vector<float> embed(
        const std::string&
    ) const override
    {
        throw std::runtime_error("Embedding failed");
    }

    std::size_t dimension() const override
    {
        return 3;
    }
};
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

TEST(VectorIndexTest, RemoveVector)
{
    cortex::index::VectorIndex index(3);

    cortex::embedding::EmbeddedChunk chunk;

    chunk.id = "chunk-001";
    chunk.email_id = "email-001";
    chunk.thread_id = "thread-001";
    chunk.text = "Test email content";
    chunk.embedding = {1.0f, 2.0f, 3.0f};
    chunk.index = 0;

    const std::size_t id = index.add(chunk);

    EXPECT_EQ(index.size(), 1);

    const bool removed = index.remove(id);

    EXPECT_TRUE(removed);
    EXPECT_EQ(index.size(), 0);

    EXPECT_THROW(
        index.get(id),
        std::out_of_range
    );
}

TEST(VectorIndexTest, FindByMetadata)
{
    cortex::index::VectorIndex index(3);

    cortex::embedding::EmbeddedChunk chunk1;
    chunk1.id = "chunk-001";
    chunk1.email_id = "email-001";
    chunk1.thread_id = "thread-001";
    chunk1.text = "First chunk";
    chunk1.embedding = {1.0f, 2.0f, 3.0f};
    chunk1.index = 0;

    cortex::embedding::EmbeddedChunk chunk2;
    chunk2.id = "chunk-002";
    chunk2.email_id = "email-001";
    chunk2.thread_id = "thread-001";
    chunk2.text = "Second chunk";
    chunk2.embedding = {2.0f, 3.0f, 4.0f};
    chunk2.index = 1;

    cortex::embedding::EmbeddedChunk chunk3;
    chunk3.id = "chunk-003";
    chunk3.email_id = "email-002";
    chunk3.thread_id = "thread-002";
    chunk3.text = "Third chunk";
    chunk3.embedding = {3.0f, 4.0f, 5.0f};
    chunk3.index = 0;

    const std::size_t id1 = index.add(chunk1);
    const std::size_t id2 = index.add(chunk2);
    const std::size_t id3 = index.add(chunk3);

    const auto results =
        index.findByMetadata("email_id", "email-001");

    ASSERT_EQ(results.size(), 2);

    EXPECT_EQ(results[0], id1);
    EXPECT_EQ(results[1], id2);

    EXPECT_EQ(
        index.findByMetadata("email_id", "email-002").size(),
        1
    );

    EXPECT_EQ(
        index.findByMetadata("email_id", "does-not-exist").size(),
        0
    );
}

TEST(EmailIndexerTest, KeepsExistingEmailWhenEmbeddingFails)
{
    const std::string model_path =
        "models/bge-small-en-v1.5/onnx/model.onnx";

    const std::string vocab_path =
        "models/bge-small-en-v1.5/vocab.txt";

    cortex::embedding::BGEEmbedder working_embedder(
        model_path,
        vocab_path
    );

    cortex::embedding::EmbeddingPipeline working_pipeline(
        working_embedder
    );

    cortex::index::VectorIndex index(
        working_embedder.dimension()
    );

    cortex::email::EmailParser parser;
    cortex::email::EmailCleaner cleaner;
    cortex::email::EmailChunker chunker(500);

    cortex::ingestion::EmailIndexer working_indexer(
        parser,
        cleaner,
        chunker,
        working_pipeline,
        index
    );

    const std::string email =
        "ID: email-001\n"
        "Thread-ID: thread-001\n"
        "From: recruiter@example.com\n"
        "To: sahil@example.com\n"
        "Subject: Interview\n"
        "Date: 2026-10-05\n"
        "\n"
        "Your technical interview is scheduled for Monday at 10 AM.";

    // Successfully index the original email.
    working_indexer.addEmail(email);

    const std::size_t original_size =
        index.size();

    ASSERT_GT(original_size, 0);

    // Create an embedder that always fails.
    FailingEmbedder failing_embedder;

    cortex::embedding::EmbeddingPipeline failing_pipeline(
        failing_embedder
    );

    cortex::ingestion::EmailIndexer failing_indexer(
        parser,
        cleaner,
        chunker,
        failing_pipeline,
        index
    );

    // Re-indexing should fail during embedding.
    EXPECT_THROW(
        failing_indexer.addEmail(email),
        std::runtime_error
    );

    // The old vectors must still exist.
    EXPECT_EQ(
        index.size(),
        original_size
    );

    const auto existing_ids =
        index.findByMetadata(
            "email_id",
            "email-001"
        );

    EXPECT_EQ(
        existing_ids.size(),
        original_size
    );
}