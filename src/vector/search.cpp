#include "vector/search.hpp"

#include "vector/batch_distance.hpp"

#include <algorithm>
#include <queue>
#include <vector>

namespace cortex::vector
{
    namespace
    {
        struct SearchResultCompare
        {
            bool operator()(
                const SearchResult& a,
                const SearchResult& b) const
            {
                return a.distance < b.distance;
            }
        };
    }

    std::vector<SearchResult> search_top_k(
        const Vector& query,
        const std::vector<Vector>& vectors,
        std::size_t k)
    {
        if (k == 0 || vectors.empty())
        {
            return {};
        }

        if (k > vectors.size())
        {
            k = vectors.size();
        }

        std::vector<float> distances;

        batch_l2_distance(
            query,
            vectors,
            distances
        );

        std::priority_queue<
            SearchResult,
            std::vector<SearchResult>,
            SearchResultCompare
        > heap;

        for (std::size_t i = 0; i < distances.size(); ++i)
        {
            SearchResult candidate{
                i,
                distances[i]
            };

            if (heap.size() < k)
            {
                heap.push(candidate);
            }
            else if (candidate.distance < heap.top().distance)
            {
                heap.pop();
                heap.push(candidate);
            }
        }

        std::vector<SearchResult> results;

        results.reserve(k);

        while (!heap.empty())
        {
            results.push_back(heap.top());
            heap.pop();
        }

        std::reverse(
            results.begin(),
            results.end()
        );

        return results;
    }
}