#include "core/buffer.hpp"

#include <cstdlib>
#include <malloc.h>
#include <cstring>
#include <stdexcept>
#include <utility>

namespace cortex::core {

    Buffer::Buffer(std::size_t size, std::size_t alignment)
        : data_(nullptr),
		size_(size), // unsigned integer type that can hold the size of any object in bytes(only positive values)
        alignment_(alignment)

    {
        if (alignment == 0 || (alignment & (alignment - 1)))
            throw std::invalid_argument("Alignment not valid");

        std::size_t allocated_size = size;
        if (size % alignment != 0) {
			allocated_size = ((size + alignment - 1) / alignment)* alignment;
        }
        if (size == 0) return;
        data_ = _aligned_malloc(allocated_size, alignment);

        if (!data_) {
            throw std::bad_alloc();
        }
        std::memset(data_, 0, allocated_size);
    }

    Buffer::~Buffer()
    {
        _aligned_free(data_);
    }

    Buffer::Buffer(Buffer&& other) noexcept
        : data_(other.data_),
        size_(other.size_),
		alignment_(other.alignment_)
    {
        other.data_ = nullptr;
        other.size_ = 0;
        other.alignment_ = 0;
    }

    Buffer& Buffer::operator=(Buffer&& other) noexcept
    {
        if (this != &other)
        {
            _aligned_free(data_);

            data_ = other.data_;
            size_ = other.size_;
            alignment_ = other.alignment_;

            other.data_ = nullptr;
            other.size_ = 0;
            other.alignment_ = 0;
        }

        return *this;
    }

    void* Buffer::data()
    {
        return data_;
    }

    const void* Buffer::data() const
    {
        return data_;
    }

    std::size_t Buffer::size() const
    {
        return size_;
    }
    std::size_t Buffer::alignment() const
    {
        return alignment_;
    }
} // namespace cortex::core