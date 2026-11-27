#include "retrival/retriever.hpp" //AFter the version 2 spelling mistakes will be corrected
#include "retrival/query_transformer.hpp"
#include <stdexcept>
#include <cmath>
#include <algorithm>
#include <unordered_map>
#include <unordered_set>
// responsibility of the retriever is
//  text --> Embedder -->  vector --> VectorIndex -- > results || this is the workflow of the retriever

namespace
{
    constexpr double RRF_CONSTANT = 60.0;

    std::vector<cortex::index::VectorSearchResult> fuseRankings(
        const std::vector<cortex::index::VectorSearchResult> &semantic_results,
        const std::vector<cortex::index::VectorSearchResult> &lexical_results,
        std::size_t k, double lexical_weight)
    {
        using Result = cortex::index::VectorSearchResult;

        std::unordered_map<std::size_t, double> scores;
        std::unordered_map<std::size_t, Result> result_by_id;

        const auto accumulate = [&](const std::vector<Result> &ranked_results, double weight)
        {
            for (std::size_t rank = 0; rank < ranked_results.size(); ++rank)
            {
                const auto &result = ranked_results[rank];

                scores[result.id] +=
                    weight / (RRF_CONSTANT + static_cast<double>(rank + 1));

                result_by_id.try_emplace(result.id, result);
            }
        };

        accumulate(semantic_results, 1.0);
        accumulate(lexical_results, lexical_weight);

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
        std::size_t k,
        std::size_t candidate_k,
        double lexical_weight) const
    {
        if (query.empty())
        {
            throw std::invalid_argument("Hybrid search query must not be empty");
        }

        if (k == 0)
        {
            throw std::invalid_argument("Hybrid search k must be greater than zero");
        }

        if (!std::isfinite(lexical_weight) || lexical_weight <= 0.0 || lexical_weight > 10.0)
            throw std::invalid_argument("Lexical weight must be finite and in (0, 10]");

        // Zero preserves the original behavior for existing callers.
        if (candidate_k == 0) candidate_k = k;
        if (candidate_k < k)
            throw std::invalid_argument("Candidate count must be at least k");

        // semantic results using query transformation
        const std::string transformed_query =QueryTransformer::transform(query);
        const std::vector<float> query_embedding =embedder_.embed(transformed_query);
        const auto semantic_results = index_.search(query_embedding, candidate_k);

        // contain the lexical results
        const auto lexical_results = index_.lexicalSearch(transformed_query, candidate_k);
        
        return fuseRankings(semantic_results, lexical_results, k, lexical_weight);
    }
}