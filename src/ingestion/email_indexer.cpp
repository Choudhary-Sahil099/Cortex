#include "ingestion/email_indexer.hpp"

namespace cortex::ingestion
{
    EmailIndexer::EmailIndexer(
        const email::EmailParser &parser,
        const email::EmailCleaner &cleaner,
        const email::EmailChunker &chunker,
        const embedding::EmbeddingPipeline &embedding_pipeline,
        index::VectorIndex &index)
        : parser_(parser),
          cleaner_(cleaner),
          chunker_(chunker),
          embedding_pipeline_(embedding_pipeline),
          index_(index)
    {
    }

    void EmailIndexer::addEmail(
        const std::string &raw_email)
    {
        const auto parsed_email =
            parser_.parse(raw_email);

        const auto cleaned_email =
            cleaner_.clean(parsed_email);

        const auto chunks =
            chunker_.chunk(cleaned_email);

        const auto embedded_chunks =
            embedding_pipeline_.embed(chunks);

        const auto existing_ids =
            index_.findByMetadata(
                "email_id",
                parsed_email.id);

        for (const auto id : existing_ids)
        {
            index_.remove(id);
        }

        for (const auto &chunk : embedded_chunks)
        {
            index_.add(chunk);
        }
    }
}