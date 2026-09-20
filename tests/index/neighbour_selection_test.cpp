#include "index/neighbour_selection.hpp"

#include <gtest/gtest.h>

using cortex::index::HNSWSearchResult;
using cortex::index::neighbourDistanceFunction;
using cortex::index::select_neighbors;


TEST(NeighborSelectionTest, DoesNotExceedMaximumNeighbors) {

    const neighbourDistanceFunction distanceFunction =
        [](std::size_t a, std::size_t b) {

        const float difference =
            static_cast<float>(
                a > b ? a - b : b - a);

        return difference;
        };

    std::vector<HNSWSearchResult> candidates{
        {0, 5.0f},
        {1, 2.0f},
        {2, 8.0f},
        {3, 1.0f},
        {4, 4.0f}
    };

    const auto selected =
        select_neighbors(
            candidates,
            3,
            distanceFunction);

    EXPECT_LE(selected.size(), 3);
}


TEST(NeighborSelectionTest, KeepsAllWhenBelowLimit) {

    const neighbourDistanceFunction distanceFunction =
        [](std::size_t a, std::size_t b) {

        // Make every pair sufficiently far apart.
        if (a == b) {
            return 0.0f;
        }

        return 100.0f;
        };

    std::vector<HNSWSearchResult> candidates{
        {0, 2.0f},
        {1, 1.0f}
    };

    const auto selected =
        select_neighbors(
            candidates,
            5,
            distanceFunction);

    ASSERT_EQ(selected.size(), 2);

    EXPECT_EQ(selected[0].id, 1);
    EXPECT_EQ(selected[1].id, 0);
}


TEST(NeighborSelectionTest, HandlesEmptyCandidates) {

    const neighbourDistanceFunction distanceFunction =
        [](std::size_t, std::size_t) {
        return 0.0f;
        };

    std::vector<HNSWSearchResult> candidates;

    const auto selected =
        select_neighbors(
            candidates,
            5,
            distanceFunction);

    EXPECT_TRUE(selected.empty());
}

TEST(NeighborSelectionTest, RejectsRedundantNeighbor) {

    const neighbourDistanceFunction distanceFunction =
        [](std::size_t a, std::size_t b) {

        // Candidate 1 and candidate 2 are very close
        // to each other.
        if ((a == 1 && b == 2) ||
            (a == 2 && b == 1)) {
            return 0.5f;
        }

        // All other pairs are far apart.
        return 10.0f;
        };

    std::vector<HNSWSearchResult> candidates{
        {1, 1.0f},
        {2, 1.1f},
        {3, 5.0f}
    };

    const auto selected =
        select_neighbors(
            candidates,
            3,
            distanceFunction);

    ASSERT_EQ(selected.size(), 2);

    EXPECT_EQ(selected[0].id, 1);
    EXPECT_EQ(selected[1].id, 3);
}