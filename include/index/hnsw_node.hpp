#pragma once

#include <cstddef>
#include <vector>

namespace cortex::index {

    class HNSWNode {
    public:
        HNSWNode(
            std::size_t id,
            std::size_t level
        );

        std::size_t id() const;
        std::size_t level() const;

        std::vector<std::size_t>& neighbors(std::size_t level);
        const std::vector<std::size_t>& neighbors(std::size_t level) const;

    private:
        std::size_t id_;
        std::size_t level_;
        std::vector<std::vector<std::size_t>> neighbors_;
    };

} // namespace cortex::index