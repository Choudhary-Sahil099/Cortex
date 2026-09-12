#include "vector/vector_batch.hpp"

#include <stdexcept>

namespace cortex::vector
{
    VectorBatch::VectorBatch(
        std::size_t vector_count,
        std::size_t dimension)
        : buffer_(
            vector_count* dimension * sizeof(float),
            32
        ),
        vector_count_(vector_count),
        dimension_(dimension)
    {
    }

    std::size_t VectorBatch::size() const
    {
        return vector_count_;
    }

    std::size_t VectorBatch::dimension() const
    {
        return dimension_;
    }

    float* VectorBatch::data()
    {
        return static_cast<float*>(buffer_.data());
    }

    const float* VectorBatch::data() const
    {
        return static_cast<const float*>(buffer_.data());
    }

    float* VectorBatch::vector_data(std::size_t index)
    {
        if (index >= vector_count_)
        {
            throw std::out_of_range(
                "VectorBatch index out of range"
            );
        }

        return data() + index * dimension_;
    }

    const float* VectorBatch::vector_data(
        std::size_t index) const
    {
        if (index >= vector_count_)
        {
            throw std::out_of_range(
                "VectorBatch index out of range"
            );
        }

        return data() + index * dimension_;
    }

    float& VectorBatch::operator()(
        std::size_t vector_index,
        std::size_t dimension_index)
    {
        if (vector_index >= vector_count_)
        {
            throw std::out_of_range(
                "VectorBatch vector index out of range"
            );
        }

        if (dimension_index >= dimension_)
        {
            throw std::out_of_range(
                "VectorBatch dimension index out of range"
            );
        }

        return data()[
            vector_index * dimension_ + dimension_index
        ];
    }

    const float& VectorBatch::operator()(
        std::size_t vector_index,
        std::size_t dimension_index) const
    {
        if (vector_index >= vector_count_)
        {
            throw std::out_of_range(
                "VectorBatch vector index out of range"
            );
        }

        if (dimension_index >= dimension_)
        {
            throw std::out_of_range(
                "VectorBatch dimension index out of range"
            );
        }

        return data()[
            vector_index * dimension_ + dimension_index
        ];
    }
}