#include <gtest/gtest.h>

#include "vector/vector.hpp"

#include <cstdint>
#include <cstddef>

using cortex::vector::Vector;

TEST(VectorTest, CreatesVectorWithCorrectDimension)
{
    Vector vector(128);

    EXPECT_EQ(vector.dimension(), 128);
}

TEST(VectorTest, DataIs32ByteAligned)
{
    Vector vector(128);

    const auto address =
        reinterpret_cast<std::uintptr_t>(vector.data());

    EXPECT_EQ(address % 32, 0);
}

TEST(VectorTest, AllowsElementAccess)
{
    Vector vector(8);

    for (std::size_t i = 0; i < vector.dimension(); ++i)
    {
        vector[i] = static_cast<float>(i) * 2.0f;
    }

    for (std::size_t i = 0; i < vector.dimension(); ++i)
    {
        EXPECT_FLOAT_EQ(
            vector[i],
            static_cast<float>(i) * 2.0f
        );
    }
}

TEST(VectorTest, DataPointerMatchesElementAccess)
{
    Vector vector(8);

    vector[0] = 42.0f;
    vector[7] = 84.0f;

    float* data = vector.data();

    EXPECT_FLOAT_EQ(data[0], 42.0f);
    EXPECT_FLOAT_EQ(data[7], 84.0f);

    EXPECT_EQ(&vector[0], data);
    EXPECT_EQ(&vector[7], data + 7);
}

TEST(VectorTest, ConstAccessWorks)
{
    Vector vector(4);

    vector[0] = 10.0f;
    vector[1] = 20.0f;

    const Vector& const_vector = vector;

    EXPECT_FLOAT_EQ(const_vector[0], 10.0f);
    EXPECT_FLOAT_EQ(const_vector[1], 20.0f);

    const float* data = const_vector.data();

    EXPECT_FLOAT_EQ(data[0], 10.0f);
    EXPECT_FLOAT_EQ(data[1], 20.0f);
}

TEST(VectorTest, SupportsZeroDimension)
{
    Vector vector(0);

    EXPECT_EQ(vector.dimension(), 0);
    EXPECT_EQ(vector.data(), nullptr);
}

TEST(VectorTest, SupportsLargeDimensions)
{
    constexpr std::size_t dimension = 100000;

    Vector vector(dimension);

    EXPECT_EQ(vector.dimension(), dimension);

    const auto address =
        reinterpret_cast<std::uintptr_t>(vector.data());

    EXPECT_EQ(address % 32, 0);
}

TEST(VectorTest, ValuesRemainIndependent)
{
    Vector a(8);
    Vector b(8);

    a[0] = 100.0f;
    b[0] = 200.0f;

    EXPECT_FLOAT_EQ(a[0], 100.0f);
    EXPECT_FLOAT_EQ(b[0], 200.0f);
}