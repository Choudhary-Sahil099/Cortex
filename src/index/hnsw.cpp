#include "index/hnsw.hpp"

#include <limits>
#include <stdexcept>

namespace cortex::index {

    HNSWIndex::HNSWIndex(
        std::size_t dimension,
        std::size_t M,
        std::size_t ef_construction,
        std::size_t ef_search
    )
        : dimension_(dimension),
        M_(M),
        ef_construction_(ef_construction),
        ef_search_(ef_search),
        max_level_(0),
        entry_point_(std::numeric_limits<std::size_t>::max()) {

        if (dimension == 0) {
            throw std::invalid_argument(
                "HNSW dimension must be greater than zero"
            );
        }

        if (M == 0) {
            throw std::invalid_argument(
                "HNSW M must be greater than zero"
            );
        }

        if (ef_construction == 0) {
            throw std::invalid_argument(
                "ef_construction must be greater than zero"
            );
        }

        if (ef_search == 0) {
            throw std::invalid_argument(
                "ef_search must be greater than zero"
            );
        }
    }

    std::size_t HNSWIndex::dimension() const {
        return dimension_;
    }

    std::size_t HNSWIndex::M() const {
        return M_;
    }

    std::size_t HNSWIndex::ef_construction() const {
        return ef_construction_;
    }

    std::size_t HNSWIndex::ef_search() const {
        return ef_search_;
    }

    std::size_t HNSWIndex::size() const {
        return nodes_.size();
    }

    bool HNSWIndex::empty() const {
        return nodes_.empty();
    }

    std::size_t HNSWIndex::max_level() const {
        return max_level_;
    }

    bool HNSWIndex::has_entry_point() const {
        return !nodes_.empty();
    }

    const HNSWNode& HNSWIndex::node(std::size_t id) const {
        if (id >= nodes_.size()) {
            throw std::out_of_range(
                "HNSW node ID out of range"
            );
        }

        return *nodes_[id];
    }

}