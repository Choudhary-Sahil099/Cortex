
#include "retrival/query_transformer.hpp"

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
