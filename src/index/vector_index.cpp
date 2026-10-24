#include "index/vector_index.hpp"

#include <stdexcept>
#include <string>
#include <utility>
#include <algorithm>
#include <cctype>
#include <sstream>
#include <unordered_map>
#include <unordered_set>
namespace cortex::index
{

    VectorIndex::VectorIndex(std::size_t dimension) : hnsw_(dimension) {}

    VectorIndex::VectorIndex(HNSWIndex hnsw) : hnsw_(std::move(hnsw)) {}

    std::size_t VectorIndex::size() const
    {
        return hnsw_.size();
    }

    std::size_t VectorIndex::dimension() const
    {
        return hnsw_.dimension();
    }

    std::size_t VectorIndex::add(
        const embedding::EmbeddedChunk &chunk)
    {
        if (chunk.embedding.size() != dimension())
        {
            throw std::invalid_argument(
                "EmbeddedChunk embedding dimension does not match VectorIndex dimension");
        }

        vector::Vector vector(chunk.embedding.size());

        for (std::size_t i = 0; i < chunk.embedding.size(); ++i)
        {
            vector[i] = chunk.embedding[i];
        }

        core::metaData metadata{
            {"chunk_id", chunk.id},
            {"email_id", chunk.email_id},
            {"thread_id", chunk.thread_id},
            {"text", chunk.text},
            {"chunk_index", std::to_string(chunk.index)}};

        return hnsw_.insert(
            std::move(vector),
            std::move(metadata));
    }
    const core::VectorRecord &VectorIndex::get(std::size_t id) const
    {
        return hnsw_.vector_store().get(id);
    }

    std::vector<VectorSearchResult> VectorIndex::search(
        const std::vector<float> &query,
        std::size_t k) const
    {
        if (query.size() != dimension())
        {
            throw std::invalid_argument(
                "Query dimension does not match VectorIndex dimension");
        }

        if (k == 0)
        {
            throw std::invalid_argument(
                "Search k must be greater than zero");
        }

        vector::Vector query_vector(query.size());

        for (std::size_t i = 0; i < query.size(); ++i)
        {
            query_vector[i] = query[i];
        }

        const auto hnsw_results =
            hnsw_.search(
                query_vector,
                k);

        std::vector<VectorSearchResult> results;
        results.reserve(hnsw_results.size());

        for (const auto &result : hnsw_results)
        {
            results.push_back({result.id,
                               result.distance,
                               &hnsw_.vector_store().get(result.id)});
        }

        return results;
    }

    // lexical search
    std::vector<VectorSearchResult> VectorIndex::lexicalSearch(const std::string &query, std::size_t k) const
    {
        if (query.empty())
        {
            throw std::invalid_argument("Lexical search query must not be empty");
        }

        if (k == 0)
        {
            throw std::invalid_argument("Lexical search k must be greater than zero");
        }

        // Convert text to lowercase whitespace-separated tokens.
        const auto tokenize = [](const std::string &text)
        {
            static const std::unordered_set<std::string> stop_words = {
                "a", "an", "and", "are", "as", "at", "be", "been",
                "by", "for", "from", "has", "have", "he", "her",
                "his", "i", "in", "is", "it", "its", "me", "my",
                "of", "on", "or", "our", "she", "that", "the",
                "their", "them", "there", "this", "to", "was",
                "we", "were", "what", "when", "where", "which",
                "who", "will", "with", "you", "your"
            };

            std::unordered_set<std::string> tokens;
            std::istringstream stream(text);
            std::string token;

            while (stream >> token)
            {
                std::string normalized;

                for (unsigned char ch : token)
                {
                    if (std::isalnum(ch))
                    {
                        normalized += static_cast<char>(
                            std::tolower(ch));
                    }
                }

                if (!normalized.empty() &&
                    !stop_words.contains(normalized))
                {
                    tokens.insert(normalized);
                }
            }

            return tokens;
        };

        const auto query_terms = tokenize(query);

        if (query_terms.empty())
        {
            return {};
        }

        std::vector<VectorSearchResult> results;

        for (const auto &record : hnsw_.vector_store().records())
        {
            const auto text_it = record.metadata.find("text");

            if (text_it == record.metadata.end())
            {
                continue;
            }

            const auto document_terms = tokenize(text_it->second);

            std::size_t matched_terms = 0;

            for (const auto &term : query_terms)
            {
                if (document_terms.contains(term))
                {
                    ++matched_terms;
                }
            }

            if (matched_terms == 0)
            {
                continue;
            }

            // Temporary lexical score: higher means more matched terms.
            results.push_back({record.id,
                               -static_cast<float>(matched_terms),
                               &record});
        }

        std::sort(
            results.begin(),
            results.end(),
            [](const VectorSearchResult &a,
               const VectorSearchResult &b)
            {
                if (a.distance != b.distance)
                {
                    return a.distance < b.distance;
                }

                return a.id < b.id;
            });

        if (results.size() > k)
        {
            results.resize(k);
        }

        return results;
    }

    bool VectorIndex::remove(std::size_t id)
    {
        return hnsw_.remove(id);
    }
    std::vector<std::size_t> VectorIndex::findByMetadata(
        const std::string &key,
        const std::string &value) const
    {
        std::vector<std::size_t> ids;

        for (const auto &record : hnsw_.vector_store().records())
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
    const HNSWIndex &VectorIndex::hnsw() const
    {
        return hnsw_;
    }
}