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