#include <gtest/gtest.h>

#include "vector/vector_batch.hpp"
#include <cstdint>
using cortex::vector::VectorBatch;

TEST(VectorBatchTest, CreatesCorrectShape)
{
    VectorBatch batch(100, 1536);

    EXPECT_EQ(batch.size(), 100);
    EXPECT_EQ(batch.dimension(), 1536);
}

TEST(VectorBatchTest, StoresValuesCorrectly)
{
    VectorBatch batch(3, 4);

    batch(0, 0) = 1.0f;
    batch(0, 1) = 2.0f;
    batch(1, 0) = 3.0f;
    batch(2, 3) = 4.0f;

    EXPECT_FLOAT_EQ(batch(0, 0), 1.0f);
    EXPECT_FLOAT_EQ(batch(0, 1), 2.0f);
    EXPECT_FLOAT_EQ(batch(1, 0), 3.0f);
    EXPECT_FLOAT_EQ(batch(2, 3), 4.0f);
}

TEST(VectorBatchTest, VectorsAreContiguous)
{
    VectorBatch batch(3, 8);

    float* first = batch.vector_data(0);
    float* second = batch.vector_data(1);
    float* third = batch.vector_data(2);

    EXPECT_EQ(second - first, 8);
    EXPECT_EQ(third - second, 8);
}

TEST(VectorBatchTest, DataIsAligned)
{
    VectorBatch batch(10, 1536);

    const auto address =
        reinterpret_cast<std::uintptr_t>(batch.data());

    EXPECT_EQ(address % 32, 0);
}

TEST(VectorBatchTest, OutOfRangeVectorThrows)
{
    VectorBatch batch(10, 128);

    EXPECT_THROW(
        batch.vector_data(10),
        std::out_of_range
    );
}

TEST(VectorBatchTest, OutOfRangeDimensionThrows)
{
    VectorBatch batch(10, 128);

    EXPECT_THROW(
        batch(0, 128),
        std::out_of_range
    );
}
