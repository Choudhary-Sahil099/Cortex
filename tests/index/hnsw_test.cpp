#include "index/hnsw.hpp"
#include <gtest/gtest.h>


#include <algorithm> // for search test 

//brute force test 
#include <cmath>
#include <vector>
#include <algorithm>
#include <random>
#include <utility>
using cortex::index::HNSWIndex;
using cortex::vector::Vector;


//helper function for vector creation
cortex::vector::Vector makeVector(
    float a,
    float b,
    float c,
    float d
) {
    cortex::vector::Vector v(4);

    v[0] = a;
    v[1] = b;
    v[2] = c;
    v[3] = d;

    return v;
}

// brute force impletation
namespace {

    std::vector<cortex::index::HNSWSearchResult>
        brute_force_search(
            const cortex::index::HNSWIndex& index,
            const cortex::vector::Vector& query,
            std::size_t k
        ) {
        std::vector<cortex::index::HNSWSearchResult> results;

        for (std::size_t id = 0; id < index.size(); ++id) {

            const float* data =
                index.vector_data(id);

            float distance = 0.0f;

            for (std::size_t i = 0;
                i < query.dimension();
                ++i) {

                const float difference =
                    query[i] - data[i];

                distance +=
                    difference * difference;
            }

            results.push_back({
                id,
                std::sqrt(distance)
                });
        }

        std::sort(
            results.begin(),
            results.end(),
            [](const auto& a, const auto& b) {
                return a.distance < b.distance;
            }
        );

        if (results.size() > k) {
            results.resize(k);
        }

        return results;
    }

};
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
        index.insert(std::move(vector)),
        std::invalid_argument
    );
}

TEST(HNSWTest, FirstInsertionCreatesEntryPoint) {

    HNSWIndex index(128);

    Vector vector(128);

    const std::size_t id = index.insert(std::move(vector));

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

    const std::size_t id_a = index.insert(std::move(a));
    const std::size_t id_b = index.insert(std::move(b));
    const std::size_t id_c = index.insert(std::move(c));

    EXPECT_EQ(id_a, 0);
    EXPECT_EQ(id_b, 1);
    EXPECT_EQ(id_c, 2);

    EXPECT_EQ(index.size(), 3);
}


// -------------------------
// -------Vector Data tests ----
// --------------------------


TEST(HNSWTest, NodeIdMatchesVectorStoreIndex) {
    std::cout << "INSERT BEGIN\n";
    HNSWIndex index(128);

    Vector a(128);
    Vector b(128);
    Vector c(128);

    const std::size_t id_a = index.insert(std::move(a));
    const std::size_t id_b = index.insert(std::move(b));
    const std::size_t id_c = index.insert(std::move(c));

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

    const std::size_t id = index.insert(std::move(vector));

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
        index.insert(std::move(vector));

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

    index.insert(std::move(stored));

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
        index.insert(std::move(a));

    const std::size_t id_b =
        index.insert(std::move(b));

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
        index.insert(std::move(a));

    const std::size_t id_b =
        index.insert(std::move(b));

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
        index.insert(std::move(vector));

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
        index.insert(std::move(a));

    const std::size_t id_b =
        index.insert(std::move(b));

    const std::size_t id_c =
        index.insert(std::move(c));

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

    
    const auto id_a = index.insert(std::move(a));
    const auto id_b = index.insert(std::move(b));
        const auto id_c = index.insert(std::move(c));

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

    index.insert(std::move(a));;
    index.insert(std::move(b));
    index.insert(std::move(c));

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

    index.insert(std::move(a));
    index.insert(std::move(b));
    index.insert(std::move(c));
    index.insert(std::move(d));

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

// ---------------------
///---------------Test M ----
// ---------------

TEST(HNSWTest, NeighborCountDoesNotExceedM) {

    constexpr std::size_t M = 2;

    HNSWIndex index(
        4,
        M,
        20,
        20
    );

    for (std::size_t i = 0; i < 20; ++i) {

        Vector vector(4);

        vector[0] =
            static_cast<float>(i);

        index.insert(std::move(vector));
    }

    for (std::size_t id = 0;
        id < index.size();
        ++id) {

        const auto& neighbors =
            index.node(id).neighbors(0);

        EXPECT_LE(
            neighbors.size(),
            M
        );
    }
}



//symetry
TEST(HNSWTest, NeighborConnectionsRemainBidirectional) {

    constexpr std::size_t M = 3;

    HNSWIndex index(
        4,
        M,
        20,
        20
    );

    for (std::size_t i = 0; i < 20; ++i) {

        Vector vector(4);

        vector[0] =
            static_cast<float>(i);

        index.insert(std::move(vector));
    }

    for (std::size_t id = 0;
        id < index.size();
        ++id) {

        const auto& neighbors =
            index.node(id).neighbors(0);

        for (const std::size_t neighbor :
        neighbors) {

            const auto& reverse_neighbors =
                index.node(neighbor).neighbors(0);

            EXPECT_NE(
                std::find(
                    reverse_neighbors.begin(),
                    reverse_neighbors.end(),
                    id
                ),
                reverse_neighbors.end()
            );
        }
    }
}

//--------------
//---------search tests------
//-----------------

TEST(HNSWTest, SearchReturnsNearestNeighbor) {

    HNSWIndex index(
        1,
        2,
        10,
        10
    );

    cortex::vector::Vector a(1);
    a[0] = 0.0f;

    cortex::vector::Vector b(1);
    b[0] = 5.0f;

    cortex::vector::Vector c(1);
    c[0] = 10.0f;

    index.insert(std::move(a));
    index.insert(std::move(b));
    index.insert(std::move(c));

    cortex::vector::Vector query(1);
    query[0] = 9.0f;

    const auto results =
        index.search(query, 1);

    ASSERT_EQ(results.size(), 1);

    EXPECT_EQ(results[0].id, 2);

    EXPECT_FLOAT_EQ(
        results[0].distance,
        1.0f
    );
}

TEST(HNSWTest, SearchReturnsTopKResults) {

    HNSWIndex index(
        1,
        3,
        20,
        20
    );

    cortex::vector::Vector a(1);
    a[0] = 0.0f;

    cortex::vector::Vector b(1);
    b[0] = 2.0f;

    cortex::vector::Vector c(1);
    c[0] = 5.0f;

    cortex::vector::Vector d(1);
    d[0] = 10.0f;

    index.insert(std::move(a));
    index.insert(std::move(b));
    index.insert(std::move(c));
    index.insert(std::move(d));

    cortex::vector::Vector query(1);
    query[0] = 4.0f;

    const auto results =
        index.search(query, 3); // three elements only

    ASSERT_EQ(results.size(), 3);

    EXPECT_EQ(results[0].id, 2);
    EXPECT_EQ(results[1].id, 1);
    EXPECT_EQ(results[0].id, 2);
    EXPECT_EQ(results[1].id, 1);

    EXPECT_EQ(results.size(), 3);

    EXPECT_LE(
        results[0].distance,
        results[1].distance
    );

    EXPECT_LE(
        results[1].distance,
        results[2].distance
    );
}

TEST(HNSWTest, SearchRejectsZeroK) {

    HNSWIndex index(2);

    cortex::vector::Vector query(2);

    EXPECT_THROW(
        index.search(query, 0),
        std::invalid_argument
    );
}

TEST(HNSWTest, SearchOnEmptyIndexReturnsEmpty) {

    HNSWIndex index(2);

    cortex::vector::Vector query(2);

    const auto results =
        index.search(query, 5);

    EXPECT_TRUE(results.empty());
}


TEST(HNSWTest, SearchMatchesBruteForce) {

    HNSWIndex index(
        3,
        4,
        50,
        50
    );

    for (std::size_t i = 0; i < 20; ++i) {

        cortex::vector::Vector vector(3);

        vector[0] = static_cast<float>(i);
        vector[1] = static_cast<float>(i * 2);
        vector[2] = static_cast<float>(i * 3);

        index.insert(std::move(vector));
    }

    cortex::vector::Vector query(3);

    query[0] = 9.2f;
    query[1] = 18.4f;
    query[2] = 27.6f;

    constexpr std::size_t k = 5;

    const auto exact =
        brute_force_search(
            index,
            query,
            k
        );

    const auto approximate =
        index.search(
            query,
            k
        );

    ASSERT_EQ(approximate.size(), k);

    std::size_t matches = 0;

    for (const auto& result : approximate) {

        const auto it =
            std::find_if(
                exact.begin(),
                exact.end(),
                [&](const auto& exact_result) {
                    return exact_result.id == result.id;
                }
            );

        if (it != exact.end()) {
            ++matches;
        }
    }

    const float recall =
        static_cast<float>(matches) /
        static_cast<float>(k);

    EXPECT_GE(
        recall,
        0.8f
    );
}

TEST(HNSWIndexTest, LayerZeroAllowsDoubleMNeighbors) {
    constexpr std::size_t dimension = 8;
    constexpr std::size_t M = 4;

    HNSWIndex index(
        dimension,
        M,
        100,
        50,
        42
    );

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    for (std::size_t i = 0; i < 100; ++i) {
        std::vector<float> values(dimension);

        for (float& value : values) {
            value = dist(rng);
        }

        cortex::vector::Vector vector(dimension);

        std::copy(
            values.begin(),
            values.end(),
            vector.data()
        );

        index.insert(std::move(vector));
    }

    for (std::size_t id = 0; id < index.size(); ++id) {
        const auto& node = index.node(id);

        // Layer 0 uses 2M.
        EXPECT_LE(
            node.neighbors(0).size(),
            2 * M
        );

        // Every connection must be bidirectional.
        for (const std::size_t neighbor : node.neighbors(0)) {
            const auto& reverse =
                index.node(neighbor).neighbors(0);

            EXPECT_NE(
                std::find(
                    reverse.begin(),
                    reverse.end(),
                    id
                ),
                reverse.end()
            );
        }
    }
}


TEST(HNSWIndexTest, UpperLayersRespectMNeighborLimit) {
    constexpr std::size_t dimension = 8;
    constexpr std::size_t M = 4;

    HNSWIndex index(
        dimension,
        M,
        100,
        50,
        42
    );

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    for (std::size_t i = 0; i < 200; ++i) {
        cortex::vector::Vector vector(dimension);

        for (std::size_t j = 0; j < dimension; ++j) {
            vector.data()[j] = dist(rng);
        }

        index.insert(std::move(vector));
    }

    for (std::size_t id = 0; id < index.size(); ++id) {
        const auto& node = index.node(id);

        for (std::size_t level = 1;
            level <= node.level();
            ++level) {

            EXPECT_LE(
                node.neighbors(level).size(),
                M
            );
        }
    }
}

TEST(HNSWIndexTest, ConnectionsNeverExceedNodeLevel) {
    constexpr std::size_t dimension = 8;

    HNSWIndex index(
        dimension,
        4,
        100,
        50,
        42
    );

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    for (std::size_t i = 0; i < 200; ++i) {
        cortex::vector::Vector vector(dimension);

        for (std::size_t j = 0; j < dimension; ++j) {
            vector.data()[j] = dist(rng);
        }

        index.insert(std::move(vector));
    }

    for (std::size_t id = 0; id < index.size(); ++id) {
        const auto& node = index.node(id);

        for (std::size_t level = 0;
            level <= node.level();
            ++level) {

            for (const std::size_t neighbor_id :
            node.neighbors(level)) {

                EXPECT_GE(
                    index.node(neighbor_id).level(),
                    level
                );
            }
        }
    }
}

TEST(HNSWIndexTest, EntryPointHasMaximumLevel) {
    constexpr std::size_t dimension = 8;
    constexpr std::size_t M = 4;

    HNSWIndex index(
        dimension,
        M,
        100,
        50,
        42
    );

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    for (std::size_t i = 0; i < 500; ++i) {
        cortex::vector::Vector vector(dimension);

        for (std::size_t j = 0; j < dimension; ++j) {
            vector.data()[j] = dist(rng);
        }

        index.insert(std::move(vector));
    }

    ASSERT_TRUE(index.has_entry_point());

    const std::size_t max_level = index.max_level();

    bool found_max_level_node = false;

    for (std::size_t id = 0; id < index.size(); ++id) {
        if (index.node(id).level() == max_level) {
            found_max_level_node = true;
            break;
        }
    }

    EXPECT_TRUE(found_max_level_node);
}


//------------------
//------remove tests --------------
// ---------------

TEST(HNSWTest, RemoveExistingNode) {
    HNSWIndex index(4, 4, 50, 20, 42);

    index.insert(
        makeVector(1.0f, 0.0f, 0.0f, 0.0f)
    );

    const std::size_t id1 =
        index.insert(
            makeVector(0.0f, 1.0f, 0.0f, 0.0f)
        );

    index.insert(
        makeVector(0.0f, 0.0f, 1.0f, 0.0f)
    );

    EXPECT_TRUE(index.remove(id1));

    EXPECT_EQ(index.size(), 2);
    EXPECT_FALSE(index.empty());

    EXPECT_THROW(
        index.node(id1),
        std::out_of_range
    );

    EXPECT_THROW(
        index.vector_data(id1),
        std::out_of_range
    );
}

TEST(HNSWTest, RemoveNonexistentNode) {
    HNSWIndex index(4, 4, 50, 20, 42);

    index.insert(
        makeVector(1.0f, 0.0f, 0.0f, 0.0f)
    );

    EXPECT_FALSE(index.remove(999));

    EXPECT_EQ(index.size(), 1);
}

TEST(HNSWTest, RemoveOnlyNode) {
    HNSWIndex index(4, 4, 50, 20, 42);

    const std::size_t id =
        index.insert(
            makeVector(1.0f, 0.0f, 0.0f, 0.0f)
        );

    EXPECT_TRUE(index.has_entry_point());
    EXPECT_EQ(index.size(), 1);

    EXPECT_TRUE(index.remove(id));

    EXPECT_TRUE(index.empty());
    EXPECT_FALSE(index.has_entry_point());
    EXPECT_EQ(index.size(), 0);
}

TEST(HNSWTest, RemovedIdIsNotReused) {
    HNSWIndex index(4, 4, 50, 20, 42);

    const std::size_t id0 =
        index.insert(
            makeVector(1.0f, 0.0f, 0.0f, 0.0f)
        );

    const std::size_t id1 =
        index.insert(
            makeVector(0.0f, 1.0f, 0.0f, 0.0f)
        );

    EXPECT_EQ(id0, 0);
    EXPECT_EQ(id1, 1);

    EXPECT_TRUE(index.remove(id0));

    const std::size_t id2 =
        index.insert(
            makeVector(0.0f, 0.0f, 1.0f, 0.0f)
        );

    EXPECT_EQ(id2, 2);
}

TEST(HNSWTest, UpdatePreservesIdAndChangesVector)
{
    HNSWIndex index(4, 4, 50, 20, 42);

    const std::size_t id =
        index.insert(
            makeVector(1.0f, 0.0f, 0.0f, 0.0f)
        );

    const bool updated =
        index.update(
            id,
            makeVector(0.0f, 1.0f, 0.0f, 0.0f)
        );

    EXPECT_TRUE(updated);

    EXPECT_EQ(index.size(), 1);

    EXPECT_EQ(index.node(id).id(), id);

    const float* data =
        index.vector_data(id);

    ASSERT_NE(data, nullptr);

    EXPECT_FLOAT_EQ(data[0], 0.0f);
    EXPECT_FLOAT_EQ(data[1], 1.0f);
    EXPECT_FLOAT_EQ(data[2], 0.0f);
    EXPECT_FLOAT_EQ(data[3], 0.0f);
}

TEST(HNSWTest, UpdateChangesMetadata)
{
    HNSWIndex index(4, 4, 50, 20, 42);

    const std::size_t id =
        index.insert(
            makeVector(1.0f, 0.0f, 0.0f, 0.0f)
        );

    cortex::core::metaData metadata{
        {"source", "email"},
        {"subject", "Updated subject"}
    };

    EXPECT_TRUE(
        index.update(
            id,
            makeVector(0.0f, 1.0f, 0.0f, 0.0f),
            metadata
        )
    );
}

TEST(HNSWTest, UpdateUnknownIdReturnsFalse)
{
    HNSWIndex index(4, 4, 50, 20, 42);

    EXPECT_FALSE(
        index.update(
            999,
            makeVector(1.0f, 0.0f, 0.0f, 0.0f)
        )
    );

    EXPECT_EQ(index.size(), 0);
}

TEST(HNSWTest, UpdateRejectsWrongDimension)
{
    HNSWIndex index(4, 4, 50, 20, 42);

    const std::size_t id =
        index.insert(
            makeVector(1.0f, 0.0f, 0.0f, 0.0f)
        );

    Vector wrong_vector(3);

    wrong_vector[0] = 1.0f;
    wrong_vector[1] = 2.0f;
    wrong_vector[2] = 3.0f;

    EXPECT_THROW(
        index.update(
            id,
            std::move(wrong_vector),
            {}
        ),
        std::invalid_argument
    );

    EXPECT_EQ(index.size(), 1);
}

TEST(HNSWTest, SearchUsesUpdatedVector)
{
    HNSWIndex index(4, 4, 50, 20, 42);

    const std::size_t id =
        index.insert(
            makeVector(1.0f, 0.0f, 0.0f, 0.0f)
        );

    index.insert(
        makeVector(0.0f, 1.0f, 0.0f, 0.0f)
    );

    index.insert(
        makeVector(0.0f, 0.0f, 1.0f, 0.0f)
    );

    EXPECT_TRUE(
        index.update(
            id,
            makeVector(0.0f, 0.0f, 1.0f, 0.0f)
        )
    );

    const auto results =
        index.search(
            makeVector(0.0f, 0.0f, 1.0f, 0.0f),
            1
        );

    ASSERT_EQ(results.size(), 1);

    EXPECT_EQ(results[0].id, id);

    EXPECT_FLOAT_EQ(results[0].distance, 0.0f);
}