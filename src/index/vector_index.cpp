#include "index/vector_index.hpp"

#include <stdexcept>
#include <string>
#include <utility>
namespace cortex::index {

    VectorIndex::VectorIndex(std::size_t dimension): hnsw_(dimension){}

    VectorIndex::VectorIndex(HNSWIndex hnsw): hnsw_(std::move(hnsw)){}

    std::size_t VectorIndex::size() const
    {
        return hnsw_.size();
    }

    std::size_t VectorIndex::dimension() const
    {
        return hnsw_.dimension();
    }

    std::size_t VectorIndex::add(
        const embedding::EmbeddedChunk& chunk
    )
    {
        if (chunk.embedding.size() != dimension()) {
            throw std::invalid_argument(
                "EmbeddedChunk embedding dimension does not match VectorIndex dimension"
            );
        }

        vector::Vector vector(chunk.embedding.size());

        for (std::size_t i = 0; i < chunk.embedding.size(); ++i) {
            vector[i] = chunk.embedding[i];
        }

        core::metaData metadata{
            {"chunk_id", chunk.id},
            {"email_id", chunk.email_id},
            {"thread_id", chunk.thread_id},
            {"text", chunk.text},
            {"chunk_index", std::to_string(chunk.index)}
        };

        return hnsw_.insert(
            std::move(vector),
            std::move(metadata)
        );
    }
    const core::VectorRecord& VectorIndex::get(std::size_t id) const
    {
        return hnsw_.vector_store().get(id);
    }

    std::vector<VectorSearchResult> VectorIndex::search(
        const std::vector<float>& query,
        std::size_t k
    ) const
    {
        if (query.size() != dimension()) {
            throw std::invalid_argument(
                "Query dimension does not match VectorIndex dimension"
            );
        }

        if (k == 0) {
            throw std::invalid_argument(
                "Search k must be greater than zero"
            );
        }

        vector::Vector query_vector(query.size());

        for (std::size_t i = 0; i < query.size(); ++i) {
            query_vector[i] = query[i];
        }

        const auto hnsw_results =
            hnsw_.search(
                query_vector,
                k
            );

        std::vector<VectorSearchResult> results;
        results.reserve(hnsw_results.size());

        for (const auto& result : hnsw_results) {
            results.push_back({
                result.id,
                result.distance,
                &hnsw_.vector_store().get(result.id)
            });
        }

        return results;
    }
    bool VectorIndex::remove(std::size_t id)
    {
        return hnsw_.remove(id);
    }
    std::vector<std::size_t> VectorIndex::findByMetadata(
        const std::string& key,
        const std::string& value
    ) const
    {
        std::vector<std::size_t> ids;

        for (const auto& record : hnsw_.vector_store().records())
        {
            const auto it = record.metadata.find(key);

            if (it != record.metadata.end() &&
                it->second == value)
            {
                ids.push_back(record.id);
            }
        }

        return ids;
    }
    const HNSWIndex& VectorIndex::hnsw() const
    {
        return hnsw_;
    }
}