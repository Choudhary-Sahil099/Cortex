#pragma once

#include "index/hnsw_node.hpp"
#include <cstddef>
#include <memory>
#include <vector>

namespace cortex::index {

    class HNSWIndex {
    public:
        HNSWIndex(
            std::size_t dimension,
            std::size_t M = 16,
            std::size_t ef_construction = 200,
            std::size_t ef_search = 50
        );

        std::size_t dimension() const;

        std::size_t M() const;

        std::size_t ef_construction() const;

        std::size_t ef_search() const;

        std::size_t size() const;

        bool empty() const;

        std::size_t max_level() const;

        bool has_entry_point() const;

        const HNSWNode& node(std::size_t id) const;

    private:
        std::size_t dimension_;

        std::size_t M_;

        std::size_t ef_construction_;

        std::size_t ef_search_;

        std::size_t max_level_;

        std::size_t entry_point_;

        std::vector<std::unique_ptr<HNSWNode>> nodes_;
    };

}