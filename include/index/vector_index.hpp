#pragma once

#include "embedding/embeded_chunk.hpp"
#include "index/hnsw.hpp"

#include <cstddef>
#include <vector>
#include <string>

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
        explicit VectorIndex(HNSWIndex hnsw);
        std::size_t size() const;

        std::size_t dimension() const;

        std::size_t add(const embedding::EmbeddedChunk &chunk);

        const core::VectorRecord &get(std::size_t id) const;
        bool remove(std::size_t id); // required to remove a vector 

        std::vector<VectorSearchResult> search(const std::vector<float> &query, std::size_t k) const;
        // to avoid duplicacy of the metadata again as a vector
        std::vector<std::size_t> findByMetadata(const std::string& key,const std::string& value) const;
        const HNSWIndex& hnsw() const;
    private:
        HNSWIndex hnsw_;
    };

}