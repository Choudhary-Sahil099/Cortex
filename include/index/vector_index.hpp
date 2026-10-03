#pragma once

#include "embedding/embeded_chunk.hpp"
#include "index/hnsw.hpp"

#include <cstddef>
#include <vector>

namespace cortex::index
{

    struct VectorSearchResult
    {
        std::size_t id;
        float distance;
        const core::VectorRecord *record;
    };
    class VectorIndex
    {
    public:
        explicit VectorIndex(std::size_t dimension);

        std::size_t size() const;

        std::size_t dimension() const;

        std::size_t add(const embedding::EmbeddedChunk &chunk);

        const core::VectorRecord &get(std::size_t id) const;

        std::vector<VectorSearchResult> search(const std::vector<float> &query, std::size_t k) const;

    private:
        HNSWIndex hnsw_;
    };

}