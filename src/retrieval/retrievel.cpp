#include "retrival/retriever.hpp" //AFter the version 2 spelling mistakes will be corrected
#include "retrival/query_transformer.hpp"
#include <stdexcept>
#include <algorithm>
#include <unordered_map>
#include <unordered_set>
#include <iostream>
// responsibility of the retriever is
//  text --> Embedder -->  vector --> VectorIndex -- > results || this is the workflow of the retriever

namespace
{
    constexpr double RRF_CONSTANT = 60.0;

    std::vector<cortex::index::VectorSearchResult> fuseRankings(
        const std::vector<cortex::index::VectorSearchResult> &semantic_results,
        const std::vector<cortex::index::VectorSearchResult> &lexical_results,
        std::size_t k)
    {
        using Result = cortex::index::VectorSearchResult;

        std::unordered_map<std::size_t, double> scores;
        std::unordered_map<std::size_t, Result> result_by_id;

        const auto accumulate = [&](const std::vector<Result> &ranked_results)
        {
            for (std::size_t rank = 0; rank < ranked_results.size(); ++rank)
            {
                const auto &result = ranked_results[rank];

                scores[result.id] +=
                    1.0 / (RRF_CONSTANT + static_cast<double>(rank + 1));

                result_by_id.try_emplace(result.id, result);
            }
        };

        accumulate(semantic_results);
        accumulate(lexical_results);

        std::vector<Result> results;
        results.reserve(scores.size());

        for (const auto &[id, score] : scores)
        {
            results.push_back(result_by_id.at(id));
        }

        std::sort(
            results.begin(),
            results.end(),
            [&scores](const Result &a, const Result &b)
            {
                if (scores.at(a.id) != scores.at(b.id))
                {
                    return scores.at(a.id) > scores.at(b.id);
                }

                return a.id < b.id;
            });

        if (results.size() > k)
        {
            results.resize(k);
        }

        return results;
    }
}
namespace cortex::retrieval
{

    Retriever::Retriever(const embedding::Embedder &embedder, const index::VectorIndex &index) : embedder_(embedder), index_(index) {}

    std::vector<index::VectorSearchResult> Retriever::search(
        const std::string &query,
        std::size_t k) const
    {
        if (query.empty())
        {
            throw std::invalid_argument(
                "Search query cannit be empty");
        }

        if (k == 0)
        {
            throw std::invalid_argument(
                "K must be greater than 0");
        }

        const std::string transformed_query = QueryTransformer::transform(query);

        const std::vector<float> query_embedding = embedder_.embed(transformed_query);

        return index_.search(query_embedding, k);
    }
    std::vector<index::VectorSearchResult> Retriever::search(
        const std::string &query,
        std::size_t k,
        float max_distance) const
    {
        if (query.empty())
        {
            throw std::invalid_argument(
                "Search query must not be empty");
        }

        if (k == 0)
        {
            throw std::invalid_argument(
                "Search k must be greater than zero");
        }

        if (max_distance < 0.0f)
        {
            throw std::invalid_argument(
                "max dist cannot be less than 0");
        }

        const std::string transformed_query = QueryTransformer::transform(query);
        const std::vector<float> query_embedding = embedder_.embed(transformed_query);
        const auto candidates =
            index_.search(
                query_embedding,
                k);

        std::vector<index::VectorSearchResult> results;

        for (const auto &candidate : candidates)
        {
            if (candidate.distance <= max_distance)
            {
                results.push_back(candidate);
            }
        }

        return results;
    }
    std::vector<index::VectorSearchResult> Retriever::hybridSearch(
        const std::string &query,
        std::size_t k) const
    {
        if (query.empty())
        {
            throw std::invalid_argument("Hybrid search query must not be empty");
        }

        if (k == 0)
        {
            throw std::invalid_argument("Hybrid search k must be greater than zero");
        }

        // semantic results using query transformation
        const std::string transformed_query =QueryTransformer::transform(query);
        const std::vector<float> query_embedding =embedder_.embed(transformed_query);
        const auto semantic_results = index_.search(query_embedding, k);

        // contain the lexical results
        const auto lexical_results = index_.lexicalSearch(transformed_query, k);
        
        std::cout << "\nSemantic ranking:\n";
        for (std::size_t i = 0; i < semantic_results.size(); ++i)
        {
            const auto &result = semantic_results[i];
            std::cout << i + 1 << ". ";

            if (result.record != nullptr &&
                result.record->metadata.count("chunk_id"))
            {
                std::cout << result.record->metadata.at("chunk_id");
            }

            std::cout << " (distance=" << result.distance << ")\n";
        }

        std::cout << "\nLexical ranking:\n";
        for (std::size_t i = 0; i < lexical_results.size(); ++i)
        {
            const auto &result = lexical_results[i];
            std::cout << i + 1 << ". ";

            if (result.record != nullptr &&
                result.record->metadata.count("chunk_id"))
            {
                std::cout << result.record->metadata.at("chunk_id");
            }

            std::cout << " (lexical score=" << result.distance << ")\n";
        }

        return fuseRankings(semantic_results, lexical_results, k);
    }
}