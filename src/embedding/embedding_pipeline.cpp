#include "embedding/embedding_pipeline.hpp"

namespace cortex::embedding
{
    EmbeddingPipeline::EmbeddingPipeline(
        const Embedder& embedder
    )
        : embedder_(embedder)
    {
    }

    EmbeddedChunk EmbeddingPipeline::embed(
        const email::emailChunk& chunk
    ) const
    {
        EmbeddedChunk result;

        result.id = chunk.id;
        result.email_id = chunk.email_id;
        result.thread_id = chunk.thread_id;
        result.text = chunk.text;
        result.index = chunk.index;

        result.embedding = embedder_.embed(chunk.text);

        return result;
    }

    std::vector<EmbeddedChunk> EmbeddingPipeline::embed(
        const std::vector<email::emailChunk>& chunks
    ) const
    {
        std::vector<EmbeddedChunk> results;

        results.reserve(chunks.size());

        for (const auto& chunk : chunks)
        {
            results.push_back(embed(chunk));
        }

        return results;
    }
}