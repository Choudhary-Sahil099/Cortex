#include <gtest/gtest.h>

#include "vector/vector_store.hpp"

#include <stdexcept>

using cortex::vector::Vector;
using cortex::vector::VectorStore;


TEST(VectorStoreTest, StartsEmpty)
{
    VectorStore store(4);

    EXPECT_EQ(store.dimension(), 4);
    EXPECT_EQ(store.size(), 0);
}


TEST(VectorStoreTest, AddsVectors)
{
    VectorStore store(3);

    Vector a(3);

    a[0] = 1.0f;
    a[1] = 2.0f;
    a[2] = 3.0f;

    store.add(a);

    EXPECT_EQ(store.size(), 1);

    EXPECT_FLOAT_EQ(store.vector_data(0)[0], 1.0f);
    EXPECT_FLOAT_EQ(store.vector_data(0)[1], 2.0f);
    EXPECT_FLOAT_EQ(store.vector_data(0)[2], 3.0f);
}


TEST(VectorStoreTest, StoresVectorsContiguously)
{
    VectorStore store(4);

    Vector a(4);
    Vector b(4);

    a[0] = 1.0f;
    b[0] = 2.0f;

    store.add(a);
    store.add(b);

    const float* first =
        store.vector_data(0);

    const float* second =
        store.vector_data(1);

    EXPECT_EQ(
        second,
        first + 4
    );
}


TEST(VectorStoreTest, RejectsMismatchedDimensions)
{
    VectorStore store(4);

    Vector invalid(8);

    EXPECT_THROW(
        store.add(invalid),
        std::invalid_argument
    );
}


TEST(VectorStoreTest, ReserveCapacity)
{
    VectorStore store(1536);

    store.reserve(1000);

    EXPECT_GE(
        store.capacity(),
        1000
    );
}


TEST(VectorStoreTest, DataIsContiguous)
{
    VectorStore store(4);

    Vector a(4);
    Vector b(4);

    for (int i = 0; i < 4; ++i)
    {
        a[i] = static_cast<float>(i);
        b[i] = static_cast<float>(i + 4);
    }

    store.add(a);
    store.add(b);

    const float* data =
        store.data();

    EXPECT_FLOAT_EQ(data[0], 0.0f);
    EXPECT_FLOAT_EQ(data[3], 3.0f);

    EXPECT_FLOAT_EQ(data[4], 4.0f);
    EXPECT_FLOAT_EQ(data[7], 7.0f);
}