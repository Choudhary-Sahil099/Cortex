
#include "retrival/query_transformer.hpp"
#include "retrival/adaptive_retrieval.hpp"

#include <gtest/gtest.h>

#include <string>

namespace
{

    using cortex::retrieval::QueryTransformer;

    TEST(QueryTransformerTest, ConvertsUppercaseToLowercase)
    {
        EXPECT_EQ(
            QueryTransformer::transform("TECHNICAL INTERVIEW"),
            "technical interview schedule date time");
    }

    TEST(QueryTransformerTest, CollapsesConsecutiveWhitespace)
    {
        EXPECT_EQ(
            QueryTransformer::transform("  What   time\tis   it?  "),
            "what time is it?");
    }

    TEST(QueryTransformerTest, PreservesPunctuation)
    {
        EXPECT_EQ(
            QueryTransformer::transform("When is my interview?"),
            "when is my interview?");
    }

    TEST(QueryTransformerTest, HandlesEmptyInput)
    {
        EXPECT_EQ(
            QueryTransformer::transform(""),
            "");
    }

    TEST(QueryTransformerTest, PreservesWordsAndTheirOrder)
    {
        EXPECT_EQ(
            QueryTransformer::transform("Monday at 10 AM"),
            "monday at 10 am");
    }

    TEST(QueryTransformerTest, HandlesWhitespaceOnlyInput)
    {
        EXPECT_EQ(
            QueryTransformer::transform(" \t\n "),
            "");
    }

    TEST(QueryTransformerTest, ExpandsTechnicalInterviewQuery)
    {
        EXPECT_EQ(
            QueryTransformer::transform("When is my technical interview?"),
            "when is my technical interview? schedule date time");
    }

    TEST(QueryTransformerTest, ExpandsInterviewAttendanceQuery)
    {
        EXPECT_EQ(
            QueryTransformer::transform(
                "When do I need to attend the interview?"),
            "when do i need to attend the interview? date time schedule");
    }

    TEST(QueryTransformerTest, ExpandsScheduledInterviewQuery)
    {
        EXPECT_EQ(
            QueryTransformer::transform(
                "What time is the interview scheduled?"),
            "what time is the interview scheduled? time schedule");
    }

    TEST(QueryTransformerTest, LeavesUnrelatedQueryUnchanged)
    {
        EXPECT_EQ(
            QueryTransformer::transform("What is my project deadline?"),
            "what is my project deadline?");
    }

}

namespace {
    using namespace cortex::retrieval;
    TEST(AdaptiveRetrievalTest, SplitsTwoIntentsWithSharedScope) {
        const auto p = planAdaptiveRetrieval("Find both the approved plan and the completion report for network maintenance.");
        EXPECT_EQ(p.route, AdaptiveRoute::Compound);
        EXPECT_EQ(p.first, "Find the approved plan for network maintenance.");
        EXPECT_EQ(p.second, "Find the completion report for network maintenance.");
    }
    TEST(AdaptiveRetrievalTest, RewritesExplicitContrastOnly) {
        const auto p = planAdaptiveRetrieval("For Project A, find the approved plan, rather than the draft.");
        EXPECT_EQ(p.route, AdaptiveRoute::Contrast);
        EXPECT_EQ(p.first, "For Project A, find the approved plan.");
        EXPECT_EQ(planAdaptiveRetrieval("Do not find the draft").route, AdaptiveRoute::Hybrid);
    }
    TEST(AdaptiveRetrievalTest, RoutesAndFallsBack) {
        EXPECT_EQ(planAdaptiveRetrieval("Find reference AB-123.").route, AdaptiveRoute::Identifier);
        EXPECT_EQ(planAdaptiveRetrieval("When did the work finish?").route, AdaptiveRoute::Temporal);
        EXPECT_EQ(planAdaptiveRetrieval("Find the report").route, AdaptiveRoute::Hybrid);
        EXPECT_EQ(planAdaptiveRetrieval("Find both the plan and the estimate and the report for Project A.").route, AdaptiveRoute::Hybrid);
    }
    TEST(AdaptiveRetrievalTest, MergeDeduplicatesAndHonorsBudget) {
        struct Result { std::size_t id; };
        const std::vector<Result> a{{1},{3},{5}}, b{{2},{1},{4}};
        const auto merged = mergeAdaptiveRankings(a,b,5);
        ASSERT_EQ(merged.size(),5u);
        EXPECT_EQ(merged[0].id,1u);
        EXPECT_EQ(merged[1].id,2u);
        EXPECT_EQ(merged[2].id,3u);
        EXPECT_EQ(merged[3].id,5u);
        EXPECT_EQ(merged[4].id,4u);
        EXPECT_TRUE(mergeAdaptiveRankings(a,b,0).empty());
    }
}
