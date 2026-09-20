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

TEST(HNSWTest, ConnectsNodesBidirectionally) {

    HNSWIndex index(4);

    Vector a(4);
    Vector b(4);

    const std::size_t id_a =
        index.insert(a);

    const std::size_t id_b =
        index.insert(b);

    index.connect_nodes(
        id_a,
        id_b,
        0
    );

    const auto& neighbors_a =
        index.node(id_a).neighbors(0);

    const auto& neighbors_b =
        index.node(id_b).neighbors(0);

    ASSERT_EQ(neighbors_a.size(), 1);
    ASSERT_EQ(neighbors_b.size(), 1);

    EXPECT_EQ(neighbors_a[0], id_b);
    EXPECT_EQ(neighbors_b[0], id_a);
}

TEST(HNSWTest, DoesNotCreateDuplicateConnections) {

    HNSWIndex index(4);

    Vector a(4);
    Vector b(4);

    const std::size_t id_a =
        index.insert(a);

    const std::size_t id_b =
        index.insert(b);

    index.connect_nodes(id_a, id_b, 0);
    index.connect_nodes(id_a, id_b, 0);

    EXPECT_EQ(
        index.node(id_a).neighbors(0).size(),
        1
    );

    EXPECT_EQ(
        index.node(id_b).neighbors(0).size(),
        1
    );
}

TEST(HNSWTest, RejectsSelfConnection) {

    HNSWIndex index(4);

    Vector vector(4);

    const std::size_t id =
        index.insert(vector);

    EXPECT_THROW(
        index.connect_nodes(id, id, 0),
        std::invalid_argument
    );
}


TEST(HNSWTest, SearchLayerTraversesGraph) {

    HNSWIndex index(4);

    Vector a(4);
    a[0] = 0.0f;

    Vector b(4);
    b[0] = 1.0f;

    Vector c(4);
    c[0] = 10.0f;

    const std::size_t id_a =
        index.insert(a);

    const std::size_t id_b =
        index.insert(b);

    const std::size_t id_c =
        index.insert(c);

    index.connect_nodes(id_a, id_b, 0);
    index.connect_nodes(id_b, id_c, 0);

    Vector query(4);
    query[0] = 0.25f;

    const auto results =
        index.search_layer(
            query,
            { id_a },
            10,
            0
        );

    ASSERT_EQ(results.size(), 3);

    EXPECT_FLOAT_EQ(results[0].distance, 0.25f);
    EXPECT_FLOAT_EQ(results[1].distance, 0.75f);
    EXPECT_FLOAT_EQ(results[2].distance, 9.75f);

    std::vector<std::size_t> ids;

    for (const auto& result : results) {
        ids.push_back(result.id);
    }

    EXPECT_TRUE(
        std::find(ids.begin(), ids.end(), id_a)
        != ids.end()
    );

    EXPECT_TRUE(
        std::find(ids.begin(), ids.end(), id_b)
        != ids.end()
    );

    EXPECT_TRUE(
        std::find(ids.begin(), ids.end(), id_c)
        != ids.end()
    );
}


// ------------------
//--------greedy search _----------
// ------------------------

TEST(HNSWTest, GreedySearchFindsCloserNode) {

    HNSWIndex index(4);

    Vector a(4);
    a[0] = 0.0f;

    Vector b(4);
    b[0] = 5.0f;

    Vector c(4);
    c[0] = 10.0f;

    const auto id_a = index.insert(a);
    const auto id_b = index.insert(b);
    const auto id_c = index.insert(c);

    index.connect_nodes(id_a, id_b, 0);
    index.connect_nodes(id_b, id_c, 0);

    Vector query(4);
    query[0] = 9.0f;

    const auto result =
        index.greedySearch(
            query,
            id_a,
            0
        );

    EXPECT_EQ(result, id_c);
}


//-----------------
//----------insertion and creation of the graph ----------
// -----------------

TEST(HNSWTest, InsertionCreatesConnections) {

    HNSWIndex index(
        4,
        2,      // this the value os the M
        10,     // elconstructor
        10      // efsearch
    );

    Vector a(4);
    a[0] = 0.0f;

    Vector b(4);
    b[0] = 1.0f;

    Vector c(4);
    c[0] = 2.0f;

    index.insert(a);
    index.insert(b);
    index.insert(c);

    bool found_connection = false;

    for (std::size_t id = 0; id < index.size(); ++id) {

        const auto& neighbors =
            index.node(id).neighbors(0);

        if (!neighbors.empty()) {
            found_connection = true;
            break;
        }
    }

    EXPECT_TRUE(found_connection);
}

TEST(HNSWTest, InsertedGraphCanBeSearched) {

    HNSWIndex index(
        4,
        4,
        20,
        20
    );

    Vector a(4);
    a[0] = 0.0f;

    Vector b(4);
    b[0] = 1.0f;

    Vector c(4);
    c[0] = 2.0f;

    Vector d(4);
    d[0] = 3.0f;

    index.insert(a);
    index.insert(b);
    index.insert(c);
    index.insert(d);

    Vector query(4);
    query[0] = 2.1f;

    const auto results =
        index.search_layer(
            query,
            { 0 },
            20,
            0
        );

    ASSERT_FALSE(results.empty());

    EXPECT_EQ(results.front().id, 2);
}