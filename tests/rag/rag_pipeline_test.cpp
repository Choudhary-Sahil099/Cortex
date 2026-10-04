#include "rag/rag_pipeline.hpp"

#include "embedding/embeded_chunk.hpp"
#include "embedding/embedder.hpp"
#include "index/vector_index.hpp"
#include "retrival/context_builder.hpp"
#include "retrival/retriever.hpp"

#include <gtest/gtest.h>

class TestEmbedder : public cortex::embedding::Embedder {
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

class RAGTestLLM : public cortex::llm::LLM {
public:
    std::string generate(
        const std::string& prompt
    ) const override
    {
        return "The technical interview is on Monday.";
    }
};

TEST(RAGPipelineTest, GeneratesAnswerFromRetrievedContext)
{
    TestEmbedder embedder;

    cortex::index::VectorIndex index(3);

    cortex::embedding::EmbeddedChunk chunk;

    chunk.id = "chunk_1";
    chunk.email_id = "email_1";
    chunk.thread_id = "thread_1";
    chunk.text =
        "Your technical interview is scheduled for Monday.";
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

    cortex::retrieval::ContextBuilder context_builder;

    RAGTestLLM llm;

    cortex::rag::RAGPipeline pipeline(
        retriever,
        context_builder,
        llm
    );

    const std::string answer =
        pipeline.ask(
            "When is my technical interview?",
            1
        );

    EXPECT_EQ(
        answer,
        "The technical interview is on Monday."
    );
}