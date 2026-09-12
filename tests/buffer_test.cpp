#include <gtest/gtest.h>

#include "core/buffer.hpp"

#include <cstdint>
#include <stdexcept>
#include <utility> // the use of the std::move function, which is used to indicate that an object can be "moved from" rather than copied.
using cortex::core::Buffer;

TEST(BufferTest, AllocatesRequestedMemory)
{
    Buffer buffer(128, 32);

    ASSERT_NE(buffer.data(), nullptr);
    EXPECT_EQ(buffer.size(), 128);
    EXPECT_EQ(buffer.alignment(), 32);
}

TEST(BufferTest, MemoryIsCorrectlyAligned)
{
    Buffer buffer(1024, 32);

    const auto address =
        reinterpret_cast<std::uintptr_t>(buffer.data());

    EXPECT_EQ(address % 32, 0);
}

TEST(BufferTest, SupportsDifferentValidAlignments)
{
    Buffer buffer16(128, 16);
    Buffer buffer32(128, 32);
    Buffer buffer64(128, 64);

    EXPECT_EQ(
        reinterpret_cast<std::uintptr_t>(buffer16.data()) % 16,
        0
    );

    EXPECT_EQ(
        reinterpret_cast<std::uintptr_t>(buffer32.data()) % 32,
        0
    );

    EXPECT_EQ(
        reinterpret_cast<std::uintptr_t>(buffer64.data()) % 64,
        0
    );
}

TEST(BufferTest, ZeroSizeBuffer)
{
    Buffer buffer(0, 32);

    EXPECT_EQ(buffer.data(), nullptr);
    EXPECT_EQ(buffer.size(), 0);
    EXPECT_EQ(buffer.alignment(), 32);
}

TEST(BufferTest, RejectsZeroAlignment)
{
    EXPECT_THROW(
        Buffer buffer(128, 0),
        std::invalid_argument
    );
}

TEST(BufferTest, RejectsNonPowerOfTwoAlignment)
{
    EXPECT_THROW(
        Buffer buffer(128, 24),
        std::invalid_argument
    );
}

TEST(BufferTest, MoveConstructorTransfersOwnership)
{
    Buffer original(256, 32);

    void* original_data = original.data();

    Buffer moved(std::move(original));

    EXPECT_EQ(moved.data(), original_data);
    EXPECT_EQ(moved.size(), 256);
    EXPECT_EQ(moved.alignment(), 32);

    EXPECT_EQ(original.data(), nullptr);
    EXPECT_EQ(original.size(), 0);
}

TEST(BufferTest, MoveAssignmentTransfersOwnership)
{
    Buffer source(256, 32);
    void* source_data = source.data();

    Buffer destination(128, 32);

    destination = std::move(source);

    EXPECT_EQ(destination.data(), source_data);
    EXPECT_EQ(destination.size(), 256);
    EXPECT_EQ(destination.alignment(), 32);

    EXPECT_EQ(source.data(), nullptr);
    EXPECT_EQ(source.size(), 0);
}