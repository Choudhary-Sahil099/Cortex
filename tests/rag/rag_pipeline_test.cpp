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



// updated to a newwer version
class RAGTestLLM : public cortex::llm::LLM {
public:
    mutable std::string last_prompt;
    mutable int call_count = 0;
    std::string generate(
        const std::string& prompt
    ) const override
    {
        last_prompt = prompt;
        ++call_count;
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

TEST(RAGPipelineTest, AskWithSourcesReturnsAnswerAndSources)
{
    cortex::index::VectorIndex index(3);

    cortex::embedding::EmbeddedChunk chunk;

    chunk.id = "chunk_1";
    chunk.email_id = "email_001";
    chunk.thread_id = "thread_001";
    chunk.text =
        "Your technical interview is scheduled for Monday at 10:00 AM.";
    chunk.index = 0;

    chunk.embedding = {
        1.0f,
        0.0f,
        0.0f
    };

    const auto id = index.add(chunk);

    TestEmbedder embedder;

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

    const auto result =
        pipeline.askWithSources(
            "When is my technical interview?",
            1
        );

    EXPECT_EQ(
        result.answer,
        "The technical interview is on Monday."
    );

    ASSERT_EQ(result.sources.size(), 1);

    EXPECT_EQ(
        result.sources[0].email_id,
        "email_001"
    );

    EXPECT_EQ(
        result.sources[0].thread_id,
        "thread_001"
    );

    EXPECT_EQ(
        result.sources[0].text,
        "Your technical interview is scheduled for Monday at 10:00 AM."
    );
}

// see the top content

TEST(RAGPipelineTest, UsesOnlyTopRankedEmailAsContext)
{
    TestEmbedder embedder;
    cortex::index::VectorIndex index(3);

    cortex::embedding::EmbeddedChunk relevant;
    relevant.id = "chunk_relevant";
    relevant.email_id = "email_relevant";
    relevant.thread_id = "thread_relevant";
    relevant.text = "Your database assignment is due Friday at 11:59 PM.";
    relevant.index = 0;
    relevant.embedding = {0.1f, 0.2f, 0.3f};
    index.add(relevant);

    cortex::embedding::EmbeddedChunk unrelated;
    unrelated.id = "chunk_unrelated";
    unrelated.email_id = "email_unrelated";
    unrelated.thread_id = "thread_unrelated";
    unrelated.text = "Your project meeting is on Monday at 2:00 PM.";
    unrelated.index = 1;
    unrelated.embedding = {1.0f, 0.0f, 0.0f};
    index.add(unrelated);

    cortex::retrieval::Retriever retriever(embedder, index);
    cortex::retrieval::ContextBuilder context_builder;
    RAGTestLLM llm;

    cortex::rag::RAGPipeline pipeline(
        retriever,
        context_builder,
        llm
    );

    const auto result = pipeline.askWithSources(
        "What is my database assignment deadline?",
        2
    );

    EXPECT_NE(
        llm.last_prompt.find(
            "Your database assignment is due Friday at 11:59 PM."
        ),
        std::string::npos
    );

    EXPECT_EQ(
        llm.last_prompt.find(
            "Your project meeting is on Monday at 2:00 PM."
        ),
        std::string::npos
    );

    ASSERT_EQ(result.sources.size(), 1);
    EXPECT_EQ(result.sources[0].email_id, "email_relevant");
}

TEST(RAGPipelineTest, EmptyResultsDoNotCallLLM)
{
    TestEmbedder embedder;
    cortex::index::VectorIndex index(3);

    cortex::retrieval::Retriever retriever(embedder, index);
    cortex::retrieval::ContextBuilder context_builder;
    RAGTestLLM llm;

    cortex::rag::RAGPipeline pipeline(
        retriever,
        context_builder,
        llm
    );

    const auto result = pipeline.askWithSources(
        "What is my database assignment deadline?",
        5
    );

    EXPECT_EQ(
        result.answer,
        "I couldn't find a relevant email to answer this question."
    );

    EXPECT_TRUE(result.sources.empty());
    EXPECT_EQ(llm.call_count, 0);
}
