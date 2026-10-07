#pragma once

#include "index/vector_index.hpp"

#include <cstddef>
#include <string>

namespace cortex::persistence
{
    class IndexManager
    {
    public:
        explicit IndexManager(std::string path);

        index::VectorIndex loadOrCreate(
            std::size_t dimension
        ) const;

        void save(
            const index::VectorIndex& index
        ) const;

    private:
        std::string path_;
    };
}