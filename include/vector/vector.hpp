#pragma once

#include <cstddef>

#include "core/buffer.hpp"

namespace cortex::vector
{
	// what does the allocated memory(from buffer) represent
    class Vector {
        public:
            explicit Vector(std::size_t dimension); // number of elements in the vector
			std::size_t dimension() const; // returns the number of elements in the vector
			float* data(); // returns a pointer to the underlying data
			const float* data() const; // returns a const pointer to the underlying data

			float& operator[](std::size_t index); // access element at index
			const float& operator[](std::size_t index) const; // access element at index (const version)
		private:
			core::Buffer buffer_;
			std::size_t dimension_;
    };
}