#pragma once

#include "vector/vector.hpp"

#include <cstddef>
#include <vector>

namespace cortex::vector
{
    class VectorStore
    {
    public:
        VectorStore(
            std::size_t dimension,
            std::size_t capacity = 0);

        std::size_t dimension() const;
        std::size_t size() const;
        std::size_t capacity() const;

        void reserve(std::size_t capacity);

        void add(const Vector& vector);

        const float* data() const;
        float* data();

        const float* vector_data(std::size_t index) const;
        float* vector_data(std::size_t index);

    private:
        std::vector<float> data_;
        std::size_t dimension_;
    };
}