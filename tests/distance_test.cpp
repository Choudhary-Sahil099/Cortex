#include <gtest/gtest.h>
#include "vector/vector.hpp"
#include "vector/distance.hpp"
#include "vector/similarity.hpp"
#include "vector/batch_distance.hpp"
#include "vector/dot_product.hpp"
#include <stdexcept>
#include <cmath>
#include <random>
#include <vector>


using cortex::vector::Vector;
using cortex::vector::l2_distance;
using cortex::vector::l2_distance_scalar;
using cortex::vector::l2_distance_avx2;
using cortex::vector::l2_distance_avx2_fma;
using cortex::vector::dot_product;
using cortex::vector::dot_product_scalar;
using cortex::vector::dot_product_avx2;
using cortex::vector::dot_product_avx2_fma;
using cortex::vector::cosine_similarity;
using cortex::vector::batch_l2_distance;

static Vector create_vector(std::size_t dimension, float multiplier)
{
    Vector v(dimension);

    for (std::size_t i = 0; i < dimension; ++i)
    {
        v[i] = static_cast<float>(i) * multiplier;
    }

    return v;
}

static Vector create_random_vector(
    std::size_t dimension,
    std::mt19937& generator)
{
    Vector v(dimension);

    std::uniform_real_distribution<float> distribution(-10.0f, 10.0f);

    for (std::size_t i = 0; i < dimension; ++i)
    {
        v[i] = distribution(generator);
    }

    return v;
}

TEST(L2DistanceTest, ScalarCorrectness)
{
    Vector a = create_vector(384, 0.1f);
    Vector b = create_vector(384, 0.2f);

    const float result =
        l2_distance_scalar(a, b);

    EXPECT_NEAR(result, 433.5978f, 1e-3f);
}
TEST(L2DistanceTest, AllBackendsProduceSameResult)
{
    Vector a = create_vector(384, 0.1f);
    Vector b = create_vector(384, 0.2f);

    const float scalar =
        l2_distance_scalar(a, b);

    const float avx2 =
        l2_distance_avx2(a, b);

    const float fma =
        l2_distance_avx2_fma(a, b);

    const float automatic =
        l2_distance(a, b);

    EXPECT_NEAR(avx2, scalar, 1e-2f);
    EXPECT_NEAR(fma, scalar, 1e-2f);
    EXPECT_NEAR(automatic, scalar, 1e-2f);
}
TEST(L2DistanceTest, HandlesNonMultipleOfEightDimensions)
{
    const std::size_t dimensions[] =
    {
        1,
        7,
        8,
        9,
        15,
        16,
        17,
        31,
        32,
        33,
        384,
        1537,
        1539
    };

    for (const std::size_t dimension : dimensions)
    {
        Vector a = create_vector(dimension, 0.1f);
        Vector b = create_vector(dimension, 0.2f);

        const float scalar =
            l2_distance_scalar(a, b);

        const float avx2 =
            l2_distance_avx2(a, b);

        const float fma =
            l2_distance_avx2_fma(a, b);

        EXPECT_NEAR(avx2, scalar, 1e-2f)
            << "AVX2 failed for dimension "
            << dimension;

        EXPECT_NEAR(fma, scalar, 1e-2f)
            << "FMA failed for dimension "
            << dimension;
    }
}
TEST(L2DistanceTest, IdenticalVectorsHaveZeroDistance)
{
    Vector a = create_vector(1024, 0.37f);

    EXPECT_NEAR(
        l2_distance_scalar(a, a),
        0.0f,
        1e-6f
    );

    EXPECT_NEAR(
        l2_distance_avx2(a, a),
        0.0f,
        1e-6f
    );

    EXPECT_NEAR(
        l2_distance_avx2_fma(a, a),
        0.0f,
        1e-6f
    );
}
TEST(L2DistanceTest, HandlesNegativeValues)
{
    Vector a(8);
    Vector b(8);

    for (std::size_t i = 0; i < 8; ++i)
    {
        a[i] = static_cast<float>(i) - 4.0f;
        b[i] = 4.0f - static_cast<float>(i);
    }

    const float scalar =
        l2_distance_scalar(a, b);

    const float avx2 =
        l2_distance_avx2(a, b);

    const float fma =
        l2_distance_avx2_fma(a, b);

    EXPECT_NEAR(avx2, scalar, 1e-5f);
    EXPECT_NEAR(fma, scalar, 1e-5f);
}

TEST(L2DistanceTest, MismatchedDimensionsThrow)
{
    Vector a(128);
    Vector b(256);

    EXPECT_THROW(
        l2_distance_scalar(a, b),
        std::invalid_argument
    );

    EXPECT_THROW(
        l2_distance_avx2(a, b),
        std::invalid_argument
    );

    EXPECT_THROW(
        l2_distance_avx2_fma(a, b),
        std::invalid_argument
    );
}

TEST(L2DistanceTest, AutomaticMatchesSelectedBackend)
{
    Vector a = create_vector(1536, 0.1f);
    Vector b = create_vector(1536, 0.2f);

    const float automatic =
        l2_distance(a, b);

    const float fma =
        l2_distance_avx2_fma(a, b);

    EXPECT_NEAR(
        automatic,
        fma,
        1e-2f
    );
}

TEST(DotProductTest, ScalarCorrectness)
{
    Vector a = create_vector(384, 0.1f);
    Vector b = create_vector(384, 0.2f);

    const float result =
        dot_product_scalar(a, b);

    float expected = 0.0f;

    for (std::size_t i = 0; i < 384; ++i)
    {
        expected +=
            (static_cast<float>(i) * 0.1f) *
            (static_cast<float>(i) * 0.2f);
    }

    EXPECT_NEAR(result, expected, 1e-2f);
}

TEST(DotProductTest, AllBackendsProduceSameResult)
{
    Vector a = create_vector(1537, 0.1f);
    Vector b = create_vector(1537, 0.2f);

    const float scalar =
        dot_product_scalar(a, b);

    const float avx2 =
        dot_product_avx2(a, b);

    const float fma =
        dot_product_avx2_fma(a, b);

    EXPECT_NEAR(avx2, scalar, std::abs(scalar) * 1e-5f);
    EXPECT_NEAR(fma, scalar, std::abs(scalar) * 1e-5f);
}

TEST(DotProductTest, OrthogonalVectors)
{
    Vector a(4);
    Vector b(4);

    a[0] = 1.0f;
    a[1] = 0.0f;
    a[2] = 0.0f;
    a[3] = 0.0f;

    b[0] = 0.0f;
    b[1] = 1.0f;
    b[2] = 0.0f;
    b[3] = 0.0f;

    EXPECT_FLOAT_EQ(
        dot_product_scalar(a, b),
        0.0f
    );

    EXPECT_FLOAT_EQ(
        dot_product_avx2(a, b),
        0.0f
    );

    EXPECT_FLOAT_EQ(
        dot_product_avx2_fma(a, b),
        0.0f
    );
}

TEST(DotProductTest, MismatchedDimensionsThrow)
{
    Vector a(128);
    Vector b(256);

    EXPECT_THROW(
        dot_product_scalar(a, b),
        std::invalid_argument
    );

    EXPECT_THROW(
        dot_product_avx2(a, b),
        std::invalid_argument
    );

    EXPECT_THROW(
        dot_product_avx2_fma(a, b),
        std::invalid_argument
    );
}

TEST(cosineSimilarityTest, OrthogonalVectors)
{
	Vector a(4);
	a[0] = 1.0f;
	a[1] = 2.0f;
	a[2] = 3.0f;
    a[3] = 4.0f;
    EXPECT_NEAR(
        cosine_similarity(a, a),
        1.0f,
		1e-5f
    );
}

TEST(CosineSimilarityTest, OrthogonalVectors)
{
    Vector a(3);
    Vector b(3);

    a[0] = 1.0f;
    a[1] = 0.0f;
    a[2] = 0.0f;

    b[0] = 0.0f;
    b[1] = 1.0f;
    b[2] = 0.0f;

    EXPECT_NEAR(
        cosine_similarity(a, b),
        0.0f,
        1e-5f
    );
}

TEST(CosineSimilarityTest, OppositeVectors)
{
    Vector a(3);
    Vector b(3);

    a[0] = 1.0f;
    a[1] = 2.0f;
    a[2] = 3.0f;

    b[0] = -1.0f;
    b[1] = -2.0f;
    b[2] = -3.0f;

    EXPECT_NEAR(
        cosine_similarity(a, b),
        -1.0f,
        1e-5f
    );
}

TEST(CosineSimilarityTest, MismatchedDimensionsThrow)
{
    Vector a(3);
    Vector b(4);

    EXPECT_THROW(
        cosine_similarity(a, b),
        std::invalid_argument
    );
}

// this test is causing a problem because the cosine similarity is undefined for zero vectors, and the function is expected to throw an exception in this case. The test checks that the exception is thrown when one of the vectors is a zero vector.
TEST(CosineSimilarityTest, ZeroVectorThrows)
{
    Vector a(3);
    Vector b(3);

    b[0] = 1.0f;

    EXPECT_THROW(
        cosine_similarity(a, b),
        std::invalid_argument
    );
}

// this is the test for RandomizedBackendsMatch for L2Distance
TEST(L2DistanceTest, RandomizedBackendsMatch)
{
    std::mt19937 generator(12345);

    const std::size_t dimensions[] = {
        1,
        7,
        8,
        9,
        15,
        16,
        17,
        31,
        32,
        33,
        64,
        127,
        128,
        129,
        384,
        768,
        1024,
        1536,
        1537,
        1539,
        2048,
        3072
    };

    for (const std::size_t dimension : dimensions)
    {
        for (int iteration = 0; iteration < 20; ++iteration)
        {
            Vector a =
                create_random_vector(dimension, generator);

            Vector b =
                create_random_vector(dimension, generator);

            const float scalar =
                l2_distance_scalar(a, b);

            const float avx2 =
                l2_distance_avx2(a, b);

            const float fma =
                l2_distance_avx2_fma(a, b);

            EXPECT_NEAR(avx2, scalar, 1e-3f)
                << "AVX2 failed at dimension "
                << dimension
                << ", iteration "
                << iteration;

            EXPECT_NEAR(fma, scalar, 1e-3f)
                << "FMA failed at dimension "
                << dimension
                << ", iteration "
                << iteration;
        }
    }
}

// randomaized test for dot product backends matching
TEST(DotProductTest, RandomizedBackendsMatch)
{
    std::mt19937 generator(54321);

    const std::size_t dimensions[] = {
        1,
        7,
        8,
        9,
        15,
        16,
        17,
        31,
        32,
        33,
        64,
        127,
        128,
        129,
        384,
        768,
        1024,
        1536,
        1537,
        1539,
        2048,
        3072
    };

    for (const std::size_t dimension : dimensions)
    {
        for (int iteration = 0; iteration < 20; ++iteration)
        {
            Vector a =
                create_random_vector(dimension, generator);

            Vector b =
                create_random_vector(dimension, generator);

            const float scalar =
                dot_product_scalar(a, b);

            const float avx2 =
                dot_product_avx2(a, b);

            const float fma =
                dot_product_avx2_fma(a, b);

            const float tolerance =
                std::max(1.0f, std::abs(scalar)) * 1e-4f;

            EXPECT_NEAR(avx2, scalar, tolerance)
                << "AVX2 failed at dimension "
                << dimension
                << ", iteration "
                << iteration;

            EXPECT_NEAR(fma, scalar, tolerance)
                << "FMA failed at dimension "
                << dimension
                << ", iteration "
                << iteration;
        }
    }
}

TEST(CosineSimilarityTest, RandomizedValuesAreValid)
{
    std::mt19937 generator(98765);

    const std::size_t dimensions[] = {
        1,
        7,
        8,
        9,
        16,
        17,
        31,
        32,
        33,
        128,
        384,
        768,
        1024,
        1536,
        1537
    };

    for (const std::size_t dimension : dimensions)
    {
        for (int iteration = 0; iteration < 20; ++iteration)
        {
            Vector a =
                create_random_vector(dimension, generator);

            Vector b =
                create_random_vector(dimension, generator);

            const float result =
                cosine_similarity(a, b);

            EXPECT_GE(result, -1.0001f)
                << "Cosine below -1 at dimension "
                << dimension;

            EXPECT_LE(result, 1.0001f)
                << "Cosine above 1 at dimension "
                << dimension;
        }
    }
}

TEST(CosineSimilarityTest, DebugZeroVector)
{
    Vector a(3);

    EXPECT_FLOAT_EQ(a[0], 0.0f);
    EXPECT_FLOAT_EQ(a[1], 0.0f);
    EXPECT_FLOAT_EQ(a[2], 0.0f);
}

TEST(BatchDistanceTest, ComputesAllDistances)
{
    Vector query(3);
    query[0] = 1.0f;
    query[1] = 2.0f;
    query[2] = 3.0f;

    std::vector<Vector> vectors;

    Vector a(3);
    a[0] = 1.0f;
    a[1] = 2.0f;
    a[2] = 3.0f;

    Vector b(3);
    b[0] = 2.0f;
    b[1] = 3.0f;
    b[2] = 4.0f;

    Vector c(3);
    c[0] = 4.0f;
    c[1] = 6.0f;
    c[2] = 8.0f;

    vectors.push_back(std::move(a));
    vectors.push_back(std::move(b));
    vectors.push_back(std::move(c));

    std::vector<float> distances;

    batch_l2_distance(query, vectors, distances);

    ASSERT_EQ(distances.size(), 3);

    EXPECT_NEAR(distances[0], 0.0f, 1e-5f);
    EXPECT_NEAR(distances[1], std::sqrt(3.0f), 1e-5f);
    EXPECT_NEAR(distances[2], std::sqrt(50.0f), 1e-5f);
}

TEST(BatchDistanceTest, HandlesEmptyInput)
{
    Vector query(3);

    std::vector<Vector> vectors;
    std::vector<float> distances;

    batch_l2_distance(
        query,
        vectors,
        distances
    );

    EXPECT_TRUE(distances.empty());
}

TEST(BatchDistanceTest, RejectsMismatchedDimensions)
{
    Vector query(3);

    std::vector<Vector> vectors;

    vectors.emplace_back(3);
    vectors.emplace_back(4);

    std::vector<float> distances;

    EXPECT_THROW(
        batch_l2_distance(
            query,
            vectors,
            distances
        ),
        std::invalid_argument
    );
}


TEST(BatchDistanceTest, OverwritesExistingOutput)
{
    Vector query(2);

    query[0] = 0.0f;
    query[1] = 0.0f;

    std::vector<Vector> vectors;
    vectors.emplace_back(2);

    vectors[0][0] = 3.0f;
    vectors[0][1] = 4.0f;

    std::vector<float> distances = {
        999.0f,
        999.0f,
        999.0f
    };

    batch_l2_distance(
        query,
        vectors,
        distances
    );

    ASSERT_EQ(distances.size(), 1);
    EXPECT_NEAR(distances[0], 5.0f, 1e-5f);
}

TEST(BatchDistanceTest, ScalarAndSIMDProduceSameResults)
{
    constexpr std::size_t dimension = 1536;

    Vector query(dimension);

    std::vector<Vector> vectors;

    for (std::size_t i = 0; i < 100; ++i)
    {
        Vector v(dimension);

        for (std::size_t j = 0; j < dimension; ++j)
        {
            query[j] =
                static_cast<float>(j) * 0.001f;

            v[j] =
                static_cast<float>((i + j) % 100) * 0.001f;
        }

        vectors.push_back(std::move(v));
    }

    std::vector<float> scalar;
    std::vector<float> simd;

    batch_l2_distance_scalar(
        query,
        vectors,
        scalar
    );

    batch_l2_distance_simd(
        query,
        vectors,
        simd
    );

    ASSERT_EQ(scalar.size(), simd.size());

    for (std::size_t i = 0; i < scalar.size(); ++i)
    {
        EXPECT_NEAR(
            scalar[i],
            simd[i],
            1e-2f
        ) << "Mismatch at index " << i;
    }
}

TEST(BatchDistanceTest, SIMDHandlesEmptyInput)
{
    Vector query(1536);

    std::vector<Vector> vectors;
    std::vector<float> distances;

    batch_l2_distance_simd(
        query,
        vectors,
        distances
    );

    EXPECT_TRUE(distances.empty());
}
