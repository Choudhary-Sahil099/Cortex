
#pragma once

#include <cstddef>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace cortex::evaluation
{

class RetrievalMetrics
{
public:
    // fraction of all relevant id's found in the top K results.
    // returns 0.0 when the relevant set is empty.
    static double recallAtK(
        const std::vector<std::string>& ranked_ids,
        const std::unordered_set<std::string>& relevant_ids,
        std::size_t k
    );

    // reciprocal rank of the first relevant result for one query.
    // returns 0.0 when no relevant result is found.
    static double reciprocalRank(
        const std::vector<std::string>& ranked_ids,
        const std::unordered_set<std::string>& relevant_ids
    );

    // normalized discounted cumulative gain at K.
    // relevance grade -->  0 = irrelevant, higher --> more relevant.
    // returns 0.0 when ideal dcg is zero.
    static double ndcgAtK(
        const std::vector<std::string>& ranked_ids,
        const std::unordered_map<std::string, double>& relevance,
        std::size_t k
    );
};

}
