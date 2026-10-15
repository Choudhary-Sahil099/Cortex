
#include "evaluation/retrieval_metric.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

using cortex::evaluation::RetrievalMetrics;

TEST(RetrievalMetricsTest, RecallAtKFindsAllRelevantResults)
{
    const std::vector<std::string> ranked = {
        "email-1", "email-2", "email-3"
    };

    const std::unordered_set<std::string> relevant = {
        "email-1", "email-3"
    };

    EXPECT_DOUBLE_EQ(
        RetrievalMetrics::recallAtK(ranked, relevant, 3),
        1.0
    );
}

TEST(RetrievalMetricsTest, RecallAtKFindsHalfRelevantResults)
{
    const std::vector<std::string> ranked = {
        "email-1", "email-2", "email-3"
    };

    const std::unordered_set<std::string> relevant = {
        "email-2", "email-4"
    };

    EXPECT_DOUBLE_EQ(
        RetrievalMetrics::recallAtK(ranked, relevant, 2),
        0.5
    );
}

TEST(RetrievalMetricsTest, RecallAtKDoesNotCountDuplicateIDs)
{
    const std::vector<std::string> ranked = {
        "email-1", "email-1", "email-2"
    };

    const std::unordered_set<std::string> relevant = {
        "email-1", "email-2"
    };

    EXPECT_DOUBLE_EQ(
        RetrievalMetrics::recallAtK(ranked, relevant, 2),
        0.5
    );
}

TEST(RetrievalMetricsTest, RecallAtKReturnsZeroForEmptyRelevantSet)
{
    const std::vector<std::string> ranked = {"email-1"};
    const std::unordered_set<std::string> relevant;

    EXPECT_DOUBLE_EQ(
        RetrievalMetrics::recallAtK(ranked, relevant, 1),
        0.0
    );
}

TEST(RetrievalMetricsTest, ReciprocalRankIsOneWhenFirstResultIsRelevant)
{
    const std::vector<std::string> ranked = {
        "email-1", "email-2", "email-3"
    };

    const std::unordered_set<std::string> relevant = {"email-1"};

    EXPECT_DOUBLE_EQ(
        RetrievalMetrics::reciprocalRank(ranked, relevant),
        1.0
    );
}

TEST(RetrievalMetricsTest, ReciprocalRankIsOneThirdForThirdResult)
{
    const std::vector<std::string> ranked = {
        "email-1", "email-2", "email-3"
    };

    const std::unordered_set<std::string> relevant = {"email-3"};

    EXPECT_NEAR(
        RetrievalMetrics::reciprocalRank(ranked, relevant),
        1.0 / 3.0,
        1e-12
    );
}

TEST(RetrievalMetricsTest, ReciprocalRankReturnsZeroWhenNoRelevantResult)
{
    const std::vector<std::string> ranked = {
        "email-1", "email-2"
    };

    const std::unordered_set<std::string> relevant = {"email-3"};

    EXPECT_DOUBLE_EQ(
        RetrievalMetrics::reciprocalRank(ranked, relevant),
        0.0
    );
}

TEST(RetrievalMetricsTest, NdcgIsOneForPerfectRanking)
{
    const std::vector<std::string> ranked = {
        "email-1", "email-2", "email-3"
    };

    const std::unordered_map<std::string, double> relevance = {
        {"email-1", 3.0},
        {"email-2", 2.0},
        {"email-3", 1.0}
    };

    EXPECT_NEAR(
        RetrievalMetrics::ndcgAtK(ranked, relevance, 3),
        1.0,
        1e-12
    );
}

TEST(RetrievalMetricsTest, NdcgRewardsBetterRanking)
{
    const std::unordered_map<std::string, double> relevance = {
        {"email-1", 3.0},
        {"email-2", 1.0}
    };

    const std::vector<std::string> good_ranking = {
        "email-1", "email-2"
    };

    const std::vector<std::string> bad_ranking = {
        "email-2", "email-1"
    };

    const double good_score =
        RetrievalMetrics::ndcgAtK(good_ranking, relevance, 2);

    const double bad_score =
        RetrievalMetrics::ndcgAtK(bad_ranking, relevance, 2);

    EXPECT_NEAR(good_score, 1.0, 1e-12);
    EXPECT_LT(bad_score, good_score);
    EXPECT_GT(bad_score, 0.0);
}

TEST(RetrievalMetricsTest, NdcgReturnsZeroWhenAllRelevanceIsZero)
{
    const std::vector<std::string> ranked = {"email-1"};

    const std::unordered_map<std::string, double> relevance = {
        {"email-1", 0.0}
    };

    EXPECT_DOUBLE_EQ(
        RetrievalMetrics::ndcgAtK(ranked, relevance, 1),
        0.0
    );
}
