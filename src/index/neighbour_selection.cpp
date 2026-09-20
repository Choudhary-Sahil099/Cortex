#include "index/neighbour_selection.hpp"

#include <algorithm>
#include <stdexcept>
#include <vector> // include imp
 
//Problem --> every node will have atmost M neighbors only else the edges in the graph will be too many
namespace cortex::index {

    std::vector<HNSWSearchResult> select_neighbors(
        const std::vector<HNSWSearchResult>& candidates,
        std::size_t max_neighbours,
        const neighbourDistanceFunction& distanceFunction
    ) {
        if (max_neighbours == 0) {
            throw std::invalid_argument(
                "max_neighbors must be greater than zero"
            );
        }

        if (!distanceFunction) {
            throw std::invalid_argument(
                "Neighbour Distance is not provided"
            );
        }

        if (candidates.empty()) {
            return {};
        }

        // Copy candidates so we can sort them without
        // modifying the original vector.
        std::vector<HNSWSearchResult> sorted_candidates = candidates;

        std::sort(
            sorted_candidates.begin(),
            sorted_candidates.end(),
            [](const HNSWSearchResult& a,
                const HNSWSearchResult& b) {
                    return a.distance < b.distance;
            }
        );

        // This contains ONLY the neighbors we finally select.
        std::vector<HNSWSearchResult> selected;

        selected.reserve(
            std::min(max_neighbours, sorted_candidates.size())
        );

        for (const auto& candidate : sorted_candidates) {

            // M is a hard upper bound.
            if (selected.size() >= max_neighbours) {
                break;
            }

            bool diverse = true;

            // Compare the candidate against neighbors
            // that have already been selected.
            for (const auto& neighbor : selected) {

                const float candidate_to_neighbor =
                    distanceFunction(
                        candidate.id,
                        neighbor.id
                    );

                const float candidate_to_node =
                    candidate.distance;

                if (candidate_to_neighbor < candidate_to_node) {
                    diverse = false;
                    break;
                }
            }

            if (diverse) {
                selected.push_back(candidate);
            }
        }

        return selected;
    }

} 