#pragma once

#include "email/email_parserer.hpp"
#include "email/email_cleaner.hpp"
#include "email/email_chunker.hpp"
#include "embedding/embedding_pipeline.hpp"
#include "index/vector_index.hpp"

#include <string>

namespace cortex::ingestion
{
    class EmailIndexer
    {
    public:
        EmailIndexer(
            const email::EmailParser& parser,
            const email::EmailCleaner& cleaner,
            const email::EmailChunker& chunker,
            const embedding::EmbeddingPipeline& embedding_pipeline,
            index::VectorIndex& index
        );

        void addEmail(
            const std::string& raw_email
        );

    private:
        const email::EmailParser& parser_;
        const email::EmailCleaner& cleaner_;
        const email::EmailChunker& chunker_;
        const embedding::EmbeddingPipeline& embedding_pipeline_;
        index::VectorIndex& index_;
    };
}