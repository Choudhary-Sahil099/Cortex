#include "index/neighbour_selection.hpp"

#include <gtest/gtest.h>

using cortex::index::HNSWSearchResult;
using cortex::index::select_neighbors;

TEST(NeighborSelectionTest, SelectsClosestNeighbors) {

    std::vector<HNSWSearchResult> candidates{
        {0, 5.0f},
        {1, 2.0f},
        {2, 8.0f},
        {3, 1.0f},
        {4, 4.0f}
    };

    const auto selected =
        select_neighbors(candidates, 3);

    ASSERT_EQ(selected.size(), 3);

    EXPECT_EQ(selected[0].id, 3);
    EXPECT_EQ(selected[1].id, 1);
    EXPECT_EQ(selected[2].id, 4);
}

TEST(NeighborSelectionTest, KeepsAllWhenBelowLimit) {

    std::vector<HNSWSearchResult> candidates{
        {0, 2.0f},
        {1, 1.0f}
    };

    const auto selected =
        select_neighbors(candidates, 5);

    ASSERT_EQ(selected.size(), 2);

    EXPECT_EQ(selected[0].id, 1);
    EXPECT_EQ(selected[1].id, 0);
}

TEST(NeighborSelectionTest, HandlesEmptyCandidates) {

    std::vector<HNSWSearchResult> candidates;

    const auto selected =
        select_neighbors(candidates, 5);

    EXPECT_TRUE(selected.empty());
}