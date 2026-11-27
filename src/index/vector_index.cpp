#include "index/vector_index.hpp"

#include <stdexcept>
#include <string>
#include <utility>
#include <algorithm>
#include <cctype>
#include <cmath>
#include <map>
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

        if (!std::all_of(chunk.embedding.begin(), chunk.embedding.end(), [](float x) { return std::isfinite(x); }))
            throw std::invalid_argument("Embedding contains a non-finite value");
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

        lexical_dirty_ = true;
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

        if (!std::all_of(query.begin(), query.end(), [](float x) { return std::isfinite(x); }))
            throw std::invalid_argument("Query contains a non-finite value");
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

    namespace {
        // ASCII lexical tokens; non-ASCII bytes are separators. Dense retrieval
        // remains available for text outside this tokenizer's scope.
        std::map<std::string, std::size_t> terms(const std::string& text) {
            std::map<std::string, std::size_t> counts;
            std::string token;
            auto flush = [&] { if (!token.empty()) { ++counts[token]; token.clear(); } };
            for (unsigned char c : text) {
                if (c >= 'A' && c <= 'Z') token += static_cast<char>(c + ('a' - 'A'));
                else if ((c >= 'a' && c <= 'z') || (c >= '0' && c <= '9')) token += static_cast<char>(c);
                else flush();
            }
            flush();
            return counts;
        }
    }

    void VectorIndex::rebuildLexicalIndex() const {
        if (!lexical_dirty_) return;
        decltype(postings_) postings;
        decltype(lengths_) lengths;
        double total = 0;
        for (const auto& record : hnsw_.vector_store().records()) {
            const auto it = record.metadata.find("text");
            if (it == record.metadata.end()) continue;
            std::size_t length = 0;
            for (const auto& [term, frequency] : terms(it->second)) {
                postings[term].push_back({record.id, frequency});
                length += frequency;
            }
            lengths.emplace(record.id, length);
            total += static_cast<double>(length);
        }
        average_length_ = lengths.empty() ? 0.0 : total / lengths.size();
        postings_.swap(postings);
        lengths_.swap(lengths);
        lexical_dirty_ = false;
    }

    std::vector<VectorSearchResult> VectorIndex::lexicalSearch(const std::string& query, std::size_t k) const {
        if (query.empty() || k == 0) throw std::invalid_argument("Invalid lexical query or k");
        rebuildLexicalIndex();
        if (average_length_ == 0.0) return {};
        constexpr double k1 = 1.2, b = 0.75;
        std::unordered_map<std::size_t, double> scores;
        for (const auto& [term, frequency] : terms(query)) {
            const auto it = postings_.find(term);
            if (it == postings_.end()) continue;
            const double df = static_cast<double>(it->second.size());
            const double idf = std::log1p((lengths_.size() - df + 0.5) / (df + 0.5));
            for (const auto& posting : it->second) {
                const double tf = static_cast<double>(posting.frequency);
                const double norm = 1.0 - b + b * lengths_.at(posting.id) / average_length_;
                scores[posting.id] += idf * tf * (k1 + 1.0) / (tf + k1 * norm);
            }
        }
        std::vector<VectorSearchResult> results;
        results.reserve(scores.size());
        for (const auto& [id, score] : scores)
            results.push_back({id, -static_cast<float>(score), &get(id)});
        const auto order = [](const auto& a, const auto& b) {
            return a.distance == b.distance ? a.id < b.id : a.distance < b.distance;
        };
        const auto limit = std::min(k, results.size());
        std::partial_sort(results.begin(), results.begin() + limit, results.end(), order);
        results.resize(limit);
        return results;
    }

    bool VectorIndex::remove(std::size_t id)
    {
        lexical_dirty_ = true;
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