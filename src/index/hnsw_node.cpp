#include "index/hnsw_node.hpp"

#include <stdexcept>

namespace cortex::index {

    HNSWNode::HNSWNode(
        std::size_t id,
        std::size_t level
    )
        : id_(id),
        level_(level),
        neighbors_(level + 1) {
    }

    std::size_t HNSWNode::id() const {
        return id_;
    }

    std::size_t HNSWNode::level() const {
        return level_;
    }

    std::vector<std::size_t>& HNSWNode::neighbors(
        std::size_t level
    ) {
        if (level > level_) {
            throw std::out_of_range(
                "Requested level exceeds node level"
            );
        }

        return neighbors_[level];
    }

    const std::vector<std::size_t>& HNSWNode::neighbors(
        std::size_t level
    ) const {
        if (level > level_) {
            throw std::out_of_range(
                "Requested level exceeds node level"
            );
        }

        return neighbors_[level];
    }

} 