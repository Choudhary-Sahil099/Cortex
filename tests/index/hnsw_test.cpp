#include "index/hnsw.hpp"
#include <gtest/gtest.h>


#include <algorithm> // for search test 

using cortex::index::HNSWIndex;
using cortex::vector::Vector;

TEST(HNSWTest, StartsEmpty) {

    HNSWIndex index(128);

    EXPECT_TRUE(index.empty());
    EXPECT_EQ(index.size(), 0);
    EXPECT_FALSE(index.has_entry_point());
}

TEST(HNSWTest, RejectsIncorrectDimension) {

    HNSWIndex index(128);

    Vector vector(256);

    EXPECT_THROW(
        index.insert(vector),
        std::invalid_argument
    );
}

TEST(HNSWTest, FirstInsertionCreatesEntryPoint) {

    HNSWIndex index(128);

    Vector vector(128);

    const std::size_t id = index.insert(vector);

    EXPECT_EQ(id, 0);
    EXPECT_EQ(index.size(), 1);
    EXPECT_FALSE(index.empty());
    EXPECT_TRUE(index.has_entry_point());
}

TEST(HNSWTest, MultipleInsertionsCreateUniqueIds) {

    HNSWIndex index(128);

    Vector a(128);
    Vector b(128);
    Vector c(128);

    const std::size_t id_a = index.insert(a);
    const std::size_t id_b = index.insert(b);
    const std::size_t id_c = index.insert(c);

    EXPECT_EQ(id_a, 0);
    EXPECT_EQ(id_b, 1);
    EXPECT_EQ(id_c, 2);

    EXPECT_EQ(index.size(), 3);
}


// -------------------------
// -------Vector Data tests ----
// --------------------------


TEST(HNSWTest, NodeIdMatchesVectorStoreIndex) {

    HNSWIndex index(128);

    Vector a(128);
    Vector b(128);
    Vector c(128);

    const std::size_t id_a = index.insert(a);
    const std::size_t id_b = index.insert(b);
    const std::size_t id_c = index.insert(c);

    EXPECT_EQ(id_a, 0);
    EXPECT_EQ(id_b, 1);
    EXPECT_EQ(id_c, 2);

    EXPECT_NE(index.vector_data(id_a), nullptr);
    EXPECT_NE(index.vector_data(id_b), nullptr);
    EXPECT_NE(index.vector_data(id_c), nullptr);
}

TEST(HNSWTest, VectorStorageIsPreserved) {

    HNSWIndex index(4);

    Vector vector(4);

    vector[0] = 10.0f;
    vector[1] = 20.0f;
    vector[2] = 30.0f;
    vector[3] = 40.0f;

    const std::size_t id = index.insert(vector);

    const float* stored = index.vector_data(id);

    ASSERT_NE(stored, nullptr);

    EXPECT_FLOAT_EQ(stored[0], 10.0f);
    EXPECT_FLOAT_EQ(stored[1], 20.0f);
    EXPECT_FLOAT_EQ(stored[2], 30.0f);
    EXPECT_FLOAT_EQ(stored[3], 40.0f);
}

TEST(HNSWTest, SearchLayerFindsEntryPoint) {

    HNSWIndex index(4);

    Vector vector(4);

    vector[0] = 1.0f;
    vector[1] = 2.0f;
    vector[2] = 3.0f;
    vector[3] = 4.0f;

    const std::size_t id =
        index.insert(vector);

    std::vector<std::size_t> entry_points{
        id
    };

    Vector query(4);

    query[0] = 1.0f;
    query[1] = 2.0f;
    query[2] = 3.0f;
    query[3] = 4.0f;

    const auto results =
        index.search_layer(
            query,
            entry_points,
            10,
            0
        );

    ASSERT_EQ(results.size(), 1);

    EXPECT_EQ(results[0].id, id);

    EXPECT_FLOAT_EQ(
        results[0].distance,
        0.0f
    );
}

TEST(HNSWTest, SearchLayerRejectsIncorrectDimension) {

    HNSWIndex index(4);

    Vector stored(4);

    index.insert(stored);

    Vector query(8);

    EXPECT_THROW(
        index.search_layer(
            query,
            { 0 },
            10,
            0
        ),
        std::invalid_argument
    );
}