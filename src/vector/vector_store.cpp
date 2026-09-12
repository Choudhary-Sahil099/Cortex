#include "vector/vector_store.hpp"

#include <stdexcept>

namespace cortex::vector
{
    VectorStore::VectorStore(
        std::size_t dimension,
        std::size_t capacity)
        : data_(),
        dimension_(dimension)
    {
        data_.reserve(
            dimension_ * capacity
        );
    }


    std::size_t VectorStore::dimension() const
    {
        return dimension_;
    }


    std::size_t VectorStore::size() const
    {
        if (dimension_ == 0)
        {
            return 0;
        }

        return data_.size() / dimension_;
    }


    std::size_t VectorStore::capacity() const
    {
        if (dimension_ == 0)
        {
            return 0;
        }

        return data_.capacity() / dimension_;
    }


    void VectorStore::reserve(
        std::size_t capacity)
    {
        data_.reserve(
            capacity * dimension_
        );
    }


    void VectorStore::add(
        const Vector& vector)
    {
        if (vector.dimension() != dimension_)
        {
            throw std::invalid_argument(
                "Vector dimension does not match VectorStore"
            );
        }

        const float* source = vector.data();

        data_.insert(
            data_.end(),
            source,
            source + dimension_
        );
    }


    const float* VectorStore::data() const
    {
        return data_.data();
    }


    float* VectorStore::data()
    {
        return data_.data();
    }


    const float* VectorStore::vector_data(
        std::size_t index) const
    {
        return data_.data() +
            index * dimension_;
    }


    float* VectorStore::vector_data(
        std::size_t index)
    {
        return data_.data() +
            index * dimension_;
    }
}