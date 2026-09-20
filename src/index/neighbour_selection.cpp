#include "index/neighbour_selection.hpp"

#include <algorithm>
#include <stdexcept>


//Problem --> every node will have atmost M neighbors only else the edges in the graph will be too many
namespace cortex::index {

    std::vector<HNSWSearchResult> select_neighbors(
        const std::vector<HNSWSearchResult>& candidates,
        std::size_t max_neighbors
    ) {
        if (max_neighbors == 0) {
            throw std::invalid_argument(
                "max_neighbors must be greater than zero"
            );
        }

        std::vector<HNSWSearchResult> selected =
            candidates;

        std::sort(
            selected.begin(),
            selected.end(),
            [](const HNSWSearchResult& a,
                const HNSWSearchResult& b) {
                    return a.distance < b.distance;
            }
        );

        if (selected.size() > max_neighbors) {
            selected.resize(max_neighbors);
        }

        return selected;
    }

} 