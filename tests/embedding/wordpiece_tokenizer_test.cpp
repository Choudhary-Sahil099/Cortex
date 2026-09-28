#include "embedding/wordpiece_tokenizer.hpp"

#include <gtest/gtest.h>

TEST(
    WordPieceTokenizerTest,
    TokenizesKnownWord
)
{
    cortex::embedding::WordPieceTokenizer tokenizer(
        "models/bge-small-en-v1.5/vocab.txt"
    );

    const auto tokens =
        tokenizer.tokenizeWord("technical");

    ASSERT_EQ(tokens.size(), 1);

    EXPECT_EQ(
        tokens[0],
        "technical"
    );
}

TEST(
    WordPieceTokenizerTest,
    ConvertsKnownTokensToIds
)
{
    cortex::embedding::WordPieceTokenizer tokenizer(
        "models/bge-small-en-v1.5/vocab.txt"
    );

    const std::vector<std::string> tokens = {
        "your",
        "technical",
        "interview",
        "is",
        "scheduled",
        "for",
        "monday"
    };

    const auto ids =
        tokenizer.tokenIds(tokens);

    ASSERT_EQ(ids.size(), 7);

    EXPECT_EQ(ids[0], 2115);
    EXPECT_EQ(ids[1], 4087);
    EXPECT_EQ(ids[2], 4357);
    EXPECT_EQ(ids[3], 2003);
    EXPECT_EQ(ids[4], 5115);
    EXPECT_EQ(ids[5], 2005);
    EXPECT_EQ(ids[6], 6928);
}