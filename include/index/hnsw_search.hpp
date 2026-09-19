#pragma once

#include <cstddef>
#include <vector>

namespace cortex::index {


    //formula -> NodeId + distance from query
    struct HNSWSearchResult {
        std::size_t id;
        float distance;
    };

} 