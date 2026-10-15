
#include "evaluation/retrieval_metric.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

namespace cortex::evaluation
{

double RetrievalMetrics::recallAtK(
    const std::vector<std::string>& ranked_ids,
    const std::unordered_set<std::string>& relevant_ids,
    std::size_t k
)
{
    if (relevant_ids.empty() || k == 0)
    {
        return 0.0;
    }

    const std::size_t limit = std::min(k, ranked_ids.size());
    std::unordered_set<std::string> found;

    for (std::size_t i = 0; i < limit; ++i)
    {
        if (relevant_ids.contains(ranked_ids[i]))
        {
            found.insert(ranked_ids[i]);
        }
    }

    return static_cast<double>(found.size())
        / static_cast<double>(relevant_ids.size());
}

double RetrievalMetrics::reciprocalRank(
    const std::vector<std::string>& ranked_ids,
    const std::unordered_set<std::string>& relevant_ids
)
{
    if (relevant_ids.empty())
    {
        return 0.0;
    }

    for (std::size_t i = 0; i < ranked_ids.size(); ++i)
    {
        if (relevant_ids.contains(ranked_ids[i]))
        {
            return 1.0 / static_cast<double>(i + 1);
        }
    }

    return 0.0;
}

double RetrievalMetrics::ndcgAtK(
    const std::vector<std::string>& ranked_ids,
    const std::unordered_map<std::string, double>& relevance,
    std::size_t k
)
{
    if (k == 0 || relevance.empty())
    {
        return 0.0;
    }

    const auto gain = [](double rel) -> double
    {
        return std::pow(2.0, rel) - 1.0;
    };

    const auto discount = [](std::size_t rank) -> double
    {
        // one based ramks
        return std::log2(static_cast<double>(rank) + 1.0);
    };

    double dcg = 0.0;
    const std::size_t limit = std::min(k, ranked_ids.size());

    for (std::size_t i = 0; i < limit; ++i)
    {
        const auto it = relevance.find(ranked_ids[i]);

        if (it != relevance.end())
        {
            dcg += gain(it->second) / discount(i + 1);
        }
    }

    // Build the ideal ranking from all known relevance grades.
    std::vector<double> ideal_relevance;
    ideal_relevance.reserve(relevance.size());

    for (const auto& [id, score] : relevance)
    {
        (void)id;
        ideal_relevance.push_back(score);
    }

    std::sort(
        ideal_relevance.begin(),
        ideal_relevance.end(),
        std::greater<double>()
    );

    double ideal_dcg = 0.0;
    const std::size_t ideal_limit =
        std::min(k, ideal_relevance.size());

    for (std::size_t i = 0; i < ideal_limit; ++i)
    {
        ideal_dcg += gain(ideal_relevance[i]) / discount(i + 1);
    }

    if (ideal_dcg <= 0.0)
    {
        return 0.0;
    }

    return dcg / ideal_dcg;
}

}
