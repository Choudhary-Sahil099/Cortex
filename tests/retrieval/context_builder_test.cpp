#include "retrival/context_builder.hpp"
#include "index/vector_index.hpp"
#include "embedding/embeded_chunk.hpp"

#include <gtest/gtest.h>

TEST(ContextBuilderTest, BuildsContextFromRetrievedChunks)
{
    cortex::index::VectorIndex index(3);

    cortex::embedding::EmbeddedChunk first;

    first.id = "chunk_1";
    first.email_id = "email_1";
    first.thread_id = "thread_1";
    first.text =
        "Your technical interview is scheduled for Monday.";
    first.index = 0;

    first.embedding = {
        1.0f,
        0.0f,
        0.0f
    };

    cortex::embedding::EmbeddedChunk second;

    second.id = "chunk_2";
    second.email_id = "email_2";
    second.thread_id = "thread_2";
    second.text =
        "Please prepare your C++ knowledge.";
    second.index = 0;

    second.embedding = {
        0.0f,
        1.0f,
        0.0f
    };

    const auto first_id = index.add(first);
    const auto second_id = index.add(second);

    std::vector<cortex::index::VectorSearchResult> results;

    results.push_back({
        first_id,
        0.1f,
        &index.get(first_id)
    });

    results.push_back({
        second_id,
        0.2f,
        &index.get(second_id)
    });

    cortex::retrieval::ContextBuilder builder;

    const std::string context =
        builder.build(results);

    EXPECT_NE(
        context.find(
            "Your technical interview is scheduled for Monday."
        ),
        std::string::npos
    );

    EXPECT_NE(
        context.find(
            "Please prepare your C++ knowledge."
        ),
        std::string::npos
    );

    EXPECT_NE(
        context.find("email_1"),
        std::string::npos
    );

    EXPECT_NE(
        context.find("email_2"),
        std::string::npos
    );
}


TEST(ContextBuilderTest, SkipsResultsWithMissingMetadata)
{
    cortex::index::VectorIndex index(3);

    cortex::embedding::EmbeddedChunk valid_chunk;

    valid_chunk.id = "chunk_valid";
    valid_chunk.email_id = "email_valid";
    valid_chunk.thread_id = "thread_valid";
    valid_chunk.text = "This is valid email content.";
    valid_chunk.index = 0;

    valid_chunk.embedding = {
        1.0f,
        0.0f,
        0.0f
    };

    const auto valid_id =
        index.add(valid_chunk);

    cortex::index::VectorSearchResult invalid_result{
        999,
        0.5f,
        nullptr
    };

    std::vector<cortex::index::VectorSearchResult> results;

    results.push_back({
        valid_id,
        0.1f,
        &index.get(valid_id)
    });

    results.push_back(invalid_result);

    cortex::retrieval::ContextBuilder builder;

    const std::string context =
        builder.build(results);

    EXPECT_NE(
        context.find(
            "This is valid email content."
        ),
        std::string::npos
    );

    EXPECT_EQ(
        context.find("Email 2"),
        std::string::npos
    );
}

TEST(ContextBuilderTest, SkipsResultsWithMissingText)
{
    cortex::index::VectorIndex index(3);

    cortex::embedding::EmbeddedChunk chunk;

    chunk.id = "chunk_missing_text";
    chunk.email_id = "email_1";
    chunk.thread_id = "thread_1";
    chunk.text = "Temporary text.";
    chunk.index = 0;

    chunk.embedding = {
        1.0f,
        0.0f,
        0.0f
    };

    const auto id = index.add(chunk);

    //search result pointing to the record
    auto& record =
        const_cast<cortex::core::VectorRecord&>(
            index.get(id)
        );

    record.metadata.erase("text");

    std::vector<cortex::index::VectorSearchResult> results;

    results.push_back({
        id,
        0.1f,
        &index.get(id)
    });

    cortex::retrieval::ContextBuilder builder;

    const std::string context =
        builder.build(results);

    EXPECT_TRUE(context.empty());
}

TEST(ContextBuilderTest, BuildsSourcesFromResults)
{
    cortex::index::VectorIndex index(3);

    cortex::embedding::EmbeddedChunk chunk;

    chunk.id = "chunk_1";
    chunk.email_id = "email_1";
    chunk.thread_id = "thread_1";
    chunk.text = "Your technical interview is on Monday.";
    chunk.index = 0;

    chunk.embedding = {
        1.0f,
        0.0f,
        0.0f
    };

    const auto id = index.add(chunk);

    std::vector<cortex::index::VectorSearchResult> results;

    results.push_back({
        id,
        0.1f,
        &index.get(id)
    });

    cortex::retrieval::ContextBuilder builder;

    const auto sources =
        builder.buildSources(results);

    ASSERT_EQ(sources.size(), 1);

    EXPECT_EQ(
        sources[0].email_id,
        "email_1"
    );

    EXPECT_EQ(
        sources[0].thread_id,
        "thread_1"
    );

    EXPECT_EQ(
        sources[0].text,
        "Your technical interview is on Monday."
    );
}

TEST(ContextBuilderTest, BuildSourcesSkipsResultsWithMissingText)
{
    cortex::index::VectorIndex index(3);

    cortex::embedding::EmbeddedChunk chunk;

    chunk.id = "chunk_missing_text";
    chunk.email_id = "email_1";
    chunk.thread_id = "thread_1";
    chunk.text = "Temporary text.";
    chunk.index = 0;

    chunk.embedding = {
        1.0f,
        0.0f,
        0.0f
    };

    const auto id = index.add(chunk);

    auto& record =
        const_cast<cortex::core::VectorRecord&>(
            index.get(id)
        );

    record.metadata.erase("text");

    std::vector<cortex::index::VectorSearchResult> results;

    results.push_back({
        id,
        0.1f,
        &index.get(id)
    });

    cortex::retrieval::ContextBuilder builder;

    const auto sources =
        builder.buildSources(results);

    EXPECT_TRUE(sources.empty());
}