#include <gtest/gtest.h>

#include "vector/search.hpp"

using cortex::vector::Vector;
using cortex::vector::SearchResult;
using cortex::vector::search_top_k;

TEST(SearchTest, ReturnsNearestVectors)
{
    Vector query(2);

    query[0] = 0.0f;
    query[1] = 0.0f;

    std::vector<Vector> vectors;

    Vector a(2);
    a[0] = 1.0f;
    a[1] = 0.0f;

    Vector b(2);
    b[0] = 5.0f;
    b[1] = 0.0f;

    Vector c(2);
    c[0] = 2.0f;
    c[1] = 0.0f;

    Vector d(2);
    d[0] = 10.0f;
    d[1] = 0.0f;

    vectors.push_back(std::move(a));
    vectors.push_back(std::move(b));
    vectors.push_back(std::move(c));
    vectors.push_back(std::move(d));

    const auto results =
        search_top_k(query, vectors, 2);

    ASSERT_EQ(results.size(), 2);

    EXPECT_EQ(results[0].index, 0);
    EXPECT_FLOAT_EQ(results[0].distance, 1.0f);

    EXPECT_EQ(results[1].index, 2);
    EXPECT_FLOAT_EQ(results[1].distance, 2.0f);
}

TEST(SearchTest, KGreaterThanVectorCount)
{
    Vector query(2);

    std::vector<Vector> vectors;

    for (int i = 0; i < 3; ++i)
    {
        Vector v(2);

        v[0] = static_cast<float>(i);
        v[1] = 0.0f;

        vectors.push_back(std::move(v));
    }

    const auto results =
        search_top_k(query, vectors, 10);

    EXPECT_EQ(results.size(), 3);
}

TEST(SearchTest, ZeroKReturnsEmpty)
{
    Vector query(2);

    std::vector<Vector> vectors;

    Vector v(2);

    vectors.push_back(std::move(v));

    const auto results =
        search_top_k(query, vectors, 0);

    EXPECT_TRUE(results.empty());
}

TEST(SearchTest, ResultsAreSortedByDistance)
{
    Vector query(2);

    std::vector<Vector> vectors;

    for (int i = 5; i >= 0; --i)
    {
        Vector v(2);

        v[0] = static_cast<float>(i);
        v[1] = 0.0f;

        vectors.push_back(std::move(v));
    }

    const auto results =
        search_top_k(query, vectors, 4);

    ASSERT_EQ(results.size(), 4);

    for (std::size_t i = 1; i < results.size(); ++i)
    {
        EXPECT_LE(
            results[i - 1].distance,
            results[i].distance
        );
    }

    EXPECT_EQ(results[0].index, 5);
    EXPECT_EQ(results[1].index, 4);
    EXPECT_EQ(results[2].index, 3);
    EXPECT_EQ(results[3].index, 2);
}