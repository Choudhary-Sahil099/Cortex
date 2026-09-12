#pragma once

#include <cstddef>

#include "core/buffer.hpp"

namespace cortex::vector
{
    class VectorBatch
    {
    public:
        VectorBatch(
            std::size_t vector_count,
            std::size_t dimension
        );

        std::size_t size() const;
        std::size_t dimension() const;

        float* data();
        const float* data() const;

        float* vector_data(std::size_t index);
        const float* vector_data(std::size_t index) const;

        float& operator()(std::size_t vector_index,
            std::size_t dimension_index);

        const float& operator()(std::size_t vector_index,
            std::size_t dimension_index) const;

    private:
        core::Buffer buffer_;

        std::size_t vector_count_;
        std::size_t dimension_;
    };
}