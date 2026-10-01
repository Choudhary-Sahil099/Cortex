#include "embedding/basic_tokenizer.hpp"

#include <gtest/gtest.h>

TEST(
    BasicTokenizerTest,
    SplitsWhitespace
)
{
    cortex::embedding::BasicTokenizer tokenizer;

    const auto tokens =
        tokenizer.tokenizer(
            "hello world"
        );

    ASSERT_EQ(tokens.size(), 2);

    EXPECT_EQ(tokens[0], "hello");
    EXPECT_EQ(tokens[1], "world");
}

TEST(
    BasicTokenizerTest,
    SplitsPunctuation
)
{
    cortex::embedding::BasicTokenizer tokenizer;

    const auto tokens =
        tokenizer.tokenizer(
            "hello, world!"
        );

    ASSERT_EQ(tokens.size(), 4);

    EXPECT_EQ(tokens[0], "hello");
    EXPECT_EQ(tokens[1], ",");
    EXPECT_EQ(tokens[2], "world");
    EXPECT_EQ(tokens[3], "!");
}

TEST(
    BasicTokenizerTest,
    HandlesBgeExample
)
{
    cortex::embedding::BasicTokenizer tokenizer;

    const auto tokens =
        tokenizer.tokenizer(
            "your technical interview is scheduled "
            "for monday."
        );

    const std::vector<std::string> expected = {
        "your",
        "technical",
        "interview",
        "is",
        "scheduled",
        "for",
        "monday",
        "."
    };

    EXPECT_EQ(
        tokens,
        expected
    );
}