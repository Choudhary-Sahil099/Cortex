#pragma once // this prevents duplicate inclusions
#include <cstddef>

namespace cortex::core {
    // How memory is allocated
    class Buffer
    {
    public:
        // alignment is the size of the 2^n
        explicit Buffer(std::size_t size, std ::size_t alignment = 32);
        ~Buffer(); // destructor

        Buffer(const Buffer&) = delete;
        Buffer& operator=(const Buffer&) = delete;

        Buffer(Buffer&& other) noexcept;
        Buffer& operator=(Buffer&& other) noexcept; // promises not to throw an exception

        void* data();
        const void* data() const;

        std::size_t size() const;
        std::size_t alignment() const;

    private:
        void* data_;
        std::size_t size_;
        std::size_t alignment_;
    };

}