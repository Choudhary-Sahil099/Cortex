#include "embedding/bgeEmbedder.hpp"
#include "embedding/embedding_pipeline.hpp"
#include "email/email_chunk.hpp"
#include "index/vector_index.hpp"
#include "retrival/retriever.hpp"
#include "retrival/context_builder.hpp"
#include "llm/local_llm.hpp"
#include "rag/rag_pipeline.hpp"
#include "email/email_parserer.hpp"
#include "email/email_cleaner.hpp"
#include "email/email_chunker.hpp"
#include "ingestion/email_indexer.hpp"
#include <iostream>

int main()
{

    //embedding model
    cortex::embedding::BGEEmbedder embedder(
        "models/bge-small-en-v1.5/onnx/model.onnx",
        "models/bge-small-en-v1.5/vocab.txt"
    );

    //raw email
    const std::string raw_email =
        "ID: email-001\n"
        "Thread-ID: thread-001\n"
        "From: recruiter@example.com\n"
        "To: sahil@example.com\n"
        "Subject: Technical Interview\n"
        "Date: 2026-10-05\n"
        "\n"
        "Your technical interview is scheduled for Monday at 10 AM.";

    cortex::email::EmailParser parser;

    cortex::email::EmailCleaner cleaner;

    cortex::email::EmailChunker chunker(500);

    cortex::embedding::EmbeddingPipeline embedding_pipeline(
        embedder
    );

    cortex::index::VectorIndex index(
        embedder.dimension()
    );

    cortex::ingestion::EmailIndexer indexer(
        parser,
        cleaner,
        chunker,
        embedding_pipeline,
        index
    );

    indexer.addEmail(raw_email);

    ///retriever
    cortex::retrieval::Retriever retriever(
        embedder,
        index
    );

    // context builder

    cortex::retrieval::ContextBuilder context_builder;

    //local llm
    cortex::llm::GenerationConfig config;

    config.max_tokens = 512;
    config.temperature = 0.2f;
    config.reasoning_effort = "low";

    cortex::llm::LocalLLM llm(
        "http://127.0.0.1:8080",
        config
    );
    //pipeline
    cortex::rag::RAGPipeline pipeline(
        retriever,
        context_builder,
        llm
    );
    //ques asking
    const std::string question =
        "When is my technical interview?";

    const std::string answer =
        pipeline.ask(question, 1);

    std::cout << "\nAnswer:\n";
    std::cout << answer << '\n';

    return 0;
}