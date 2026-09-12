#include <gtest/gtest.h>

#include "vector/batch_distance.hpp"
#include "vector/batch_distance_simd.hpp"

#include <cmath>
#include <vector>

using cortex::vector::Vector;
using cortex::vector::batch_l2_distance;
using cortex::vector::batch_l2_distance_avx2;
using cortex::vector::batch_l2_distance_avx2_fma;


TEST(BatchSIMDTest, AVX2MatchesScalar)
{
    constexpr std::size_t dimension = 1536;
    constexpr std::size_t count = 100;

    Vector query(dimension);

    std::vector<Vector> vectors;
    vectors.reserve(count);

    for (std::size_t i = 0; i < dimension; ++i)
    {
        query[i] = static_cast<float>(i) * 0.001f;
    }

    for (std::size_t v = 0; v < count; ++v)
    {
        Vector vector(dimension);

        for (std::size_t i = 0; i < dimension; ++i)
        {
            vector[i] =
                static_cast<float>((i + v) % 100) * 0.002f;
        }

        vectors.push_back(std::move(vector));
    }

    std::vector<float> scalar;
    std::vector<float> avx2;

    batch_l2_distance(
        query,
        vectors,
        scalar
    );

    batch_l2_distance_avx2(
        query,
        vectors,
        avx2
    );

    ASSERT_EQ(scalar.size(), avx2.size());

    for (std::size_t i = 0; i < scalar.size(); ++i)
    {
        EXPECT_NEAR(
            scalar[i],
            avx2[i],
            1e-4f
        );
    }
}


TEST(BatchSIMDTest, AVX2FMAMatchesScalar)
{
    constexpr std::size_t dimension = 1536;
    constexpr std::size_t count = 100;

    Vector query(dimension);

    std::vector<Vector> vectors;
    vectors.reserve(count);

    for (std::size_t i = 0; i < dimension; ++i)
    {
        query[i] = static_cast<float>(i) * 0.001f;
    }

    for (std::size_t v = 0; v < count; ++v)
    {
        Vector vector(dimension);

        for (std::size_t i = 0; i < dimension; ++i)
        {
            vector[i] =
                static_cast<float>((i + v) % 100) * 0.002f;
        }

        vectors.push_back(std::move(vector));
    }

    std::vector<float> scalar;
    std::vector<float> fma;

    batch_l2_distance(
        query,
        vectors,
        scalar
    );

    batch_l2_distance_avx2_fma(
        query,
        vectors,
        fma
    );

    ASSERT_EQ(scalar.size(), fma.size());

    for (std::size_t i = 0; i < scalar.size(); ++i)
    {
        EXPECT_NEAR(
            scalar[i],
            fma[i],
            1e-4f
        );
    }
}


TEST(BatchSIMDTest, HandlesNonMultipleOfSixteen)
{
    constexpr std::size_t dimension = 1537;

    Vector query(dimension);
    Vector vector(dimension);

    for (std::size_t i = 0; i < dimension; ++i)
    {
        query[i] = static_cast<float>(i) * 0.1f;
        vector[i] = static_cast<float>(i) * 0.2f;
    }

    std::vector<Vector> vectors;
    vectors.push_back(std::move(vector));

    std::vector<float> scalar;
    std::vector<float> fma;

    batch_l2_distance(
        query,
        vectors,
        scalar
    );

    batch_l2_distance_avx2_fma(
        query,
        vectors,
        fma
    );

    ASSERT_EQ(scalar.size(), 1);
    ASSERT_EQ(fma.size(), 1);

    EXPECT_NEAR(
        scalar[0],
        fma[0],
        1e-3f
    );
}


TEST(BatchSIMDTest, HandlesEmptyInput)
{
    Vector query(1536);

    std::vector<Vector> vectors;
    std::vector<float> distances;

    batch_l2_distance_avx2_fma(
        query,
        vectors,
        distances
    );

    EXPECT_TRUE(distances.empty());
}


TEST(BatchSIMDTest, RejectsMismatchedDimensions)
{
    Vector query(1536);

    Vector valid(1536);
    Vector invalid(768);

    std::vector<Vector> vectors;

    vectors.push_back(std::move(valid));
    vectors.push_back(std::move(invalid));

    std::vector<float> distances;

    EXPECT_THROW(
        batch_l2_distance_avx2_fma(
            query,
            vectors,
            distances
        ),
        std::invalid_argument
    );
}