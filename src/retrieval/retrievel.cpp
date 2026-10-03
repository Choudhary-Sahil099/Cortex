#include "retrival/retriever.hpp" // sorryb for the spelling mistake

#include <stdexcept>


//responsibility of the retriever is 
// text --> Embedder -->  vector --> VectorIndex -- > results || this is the workflow of the retriever
namespace cortex::retrieval {

    Retriever::Retriever(const embedding::Embedder& embedder,const index::VectorIndex& index): embedder_(embedder),index_(index){}

    std::vector<index::VectorSearchResult> Retriever::search(
        const std::string& query,
        std::size_t k
    ) const
    {
        if (query.empty()) {
            throw std::invalid_argument(
                "Search query cannit be empty"
            );
        }

        if (k == 0) {
            throw std::invalid_argument(
                "K must be greater than 0"
            );
        }

        const std::vector<float> query_embedding = embedder_.embed(query);

        return index_.search(query_embedding,k);
    }
    std::vector<index::VectorSearchResult> Retriever::search(
        const std::string& query,
        std::size_t k,
        float max_distance
    ) const
    {
        if (query.empty()) {
            throw std::invalid_argument(
                "Search query must not be empty"
            );
        }

        if (k == 0) {
            throw std::invalid_argument(
                "Search k must be greater than zero"
            );
        }

        if (max_distance < 0.0f) {
            throw std::invalid_argument(
                "max dist cannot be less than 0"
            );
        }

        const std::vector<float> query_embedding =
            embedder_.embed(query);

        const auto candidates =
            index_.search(
                query_embedding,
                k
            );

        std::vector<index::VectorSearchResult> results;

        for (const auto& candidate : candidates) {
            if (candidate.distance <= max_distance) {
                results.push_back(candidate);
            }
        }

        return results;
    }

}