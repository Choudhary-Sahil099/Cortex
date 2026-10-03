#pragma once

#include "embedding/embedder.hpp"
#include "index/vector_index.hpp"

#include <cstddef>
#include <string>
#include <vector>

namespace cortex::retrieval {

class Retriever {
public:

// embedder and index
    Retriever(
        const embedding::Embedder& embedder,
        const index::VectorIndex& index
    );

    // we let the caller choose one out of the two functions 
    std::vector<index::VectorSearchResult> search(
        const std::string& query,
        std::size_t k
    ) const;
    // relievence filtering
    std::vector<index::VectorSearchResult> search(
        const std::string& query,
        std::size_t k,
        float max_distance
    ) const;

private:
    const embedding::Embedder& embedder_;
    const index::VectorIndex& index_;
};

}