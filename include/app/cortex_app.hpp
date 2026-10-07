#pragma once

#include "email/email_parserer.hpp"
#include "email/email_cleaner.hpp"
#include "email/email_chunker.hpp"

#include "embedding/embedding_pipeline.hpp"
#include "embedding/embedder.hpp"
#include "ingestion/email_file_loader.hpp"
#include "index/vector_index.hpp"

#include "ingestion/email_indexer.hpp"

#include "retrival/retriever.hpp"
#include "retrival/context_builder.hpp"

#include "llm/llm.hpp"

#include "rag/rag_pipeline.hpp"

#include "persistence/index_manager.hpp"

#include <cstddef>
#include <string>

namespace cortex::app
{

class CortexApp
{
public:
    CortexApp(const embedding::Embedder& embedder,const llm::LLM& llm,std::string index_path,std::size_t chunk_size = 500);

    void indexEmail(const std::string& raw_email);

    std::string ask(const std::string& question,std::size_t k = 5) const;
    rag::RAGResult askWithSources(const std::string& question,std::size_t k = 5) const;
    void indexDirectory(const std::string& directory);
    std::size_t indexSize() const;

private:
    const embedding::Embedder& embedder_;
    const llm::LLM& llm_;

    email::EmailParser parser_;
    email::EmailCleaner cleaner_;
    email::EmailChunker chunker_;

    embedding::EmbeddingPipeline embedding_pipeline_;

    persistence::IndexManager index_manager_;
    index::VectorIndex index_;
    
    ingestion::EmailFileLoader file_loader_;
    ingestion::EmailIndexer email_indexer_;

    retrieval::Retriever retriever_;
    retrieval::ContextBuilder context_builder_;

    rag::RAGPipeline rag_pipeline_;
};

}