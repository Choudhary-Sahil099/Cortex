#pragma once

#include "embedding/embedder.hpp"
#include "embedding/embeded_chunk.hpp"
#include "email/email_chunk.hpp"

#include <vector>

namespace cortex::embedding
{
    class EmbeddingPipeline
    {
    public:

        explicit EmbeddingPipeline(
            const Embedder& embedder
        );

        EmbeddedChunk embed(
            const email::emailChunk& chunk
        ) const;

        std::vector<EmbeddedChunk> embed(
            const std::vector<email::emailChunk>& chunks
        ) const;

    private:

        const Embedder& embedder_;
    };
}