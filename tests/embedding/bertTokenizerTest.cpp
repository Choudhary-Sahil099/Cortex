#include "embedding/bert_tokenizer.hpp"

#include <gtest/gtest.h>

TEST(BertTokenizerTest, EncodesBgeExample)
{
    cortex::embedding::BertTokenizer tokenizer(
        "models/bge-small-en-v1.5/vocab.txt"
    );

    const auto result =tokenizer.encode("Your technical interview is scheduled for Monday.");
    const std::vector<int64_t> expected_ids = {
        101,
        2115,
        4087,
        4357,
        2003,
        5115,
        2005,
        6928,
        1012,
        102
    };

    EXPECT_EQ(result.input_ids,expected_ids);

    EXPECT_EQ(result.attention_mask.size(),expected_ids.size());

    EXPECT_EQ(result.token_type_ids.size(),expected_ids.size());

    for (const auto value : result.attention_mask) {
        EXPECT_EQ(value, 1);
    }

    for (const auto value : result.token_type_ids) {
        EXPECT_EQ(value, 0);
    }
}

TEST(BertTokenizerTest, HandlesPunctuation)
{
    cortex::embedding::BertTokenizer tokenizer(
        "models/bge-small-en-v1.5/vocab.txt"
    );

    const auto result =
        tokenizer.encode("Hello, WORLD!");

    EXPECT_EQ(
        result.input_ids,
        std::vector<int64_t>({
            101, 7592, 1010, 2088, 999, 102
        })
    );

    EXPECT_EQ(
        result.attention_mask,
        std::vector<int64_t>({
            1, 1, 1, 1, 1, 1
        })
    );

    EXPECT_EQ(
        result.token_type_ids,
        std::vector<int64_t>({
            0, 0, 0, 0, 0, 0
        })
    );
}

TEST(BertTokenizerTest, HandlesEmailText)
{
    cortex::embedding::BertTokenizer tokenizer(
        "models/bge-small-en-v1.5/vocab.txt"
    );

    const auto result =
        tokenizer.encode(
            "Email: TEST@example.com"
        );

    ASSERT_FALSE(
        result.input_ids.empty()
    );

    EXPECT_EQ(
        result.input_ids.front(),
        101
    );

    EXPECT_EQ(
        result.input_ids.back(),
        102
    );

    EXPECT_EQ(
        result.input_ids.size(),
        result.attention_mask.size()
    );

    EXPECT_EQ(
        result.input_ids.size(),
        result.token_type_ids.size()
    );
}

TEST(BertTokenizerTest, HandlesUnderscoreAndExtension)
{
    cortex::embedding::BertTokenizer tokenizer(
        "models/bge-small-en-v1.5/vocab.txt"
    );

    const auto result =
        tokenizer.encode(
            "This is an email. Please review the attached_document.pdf."
        );

    EXPECT_EQ(
        result.input_ids,
        std::vector<int64_t>({
            101,
            2023,
            2003,
            2019,
            10373,
            1012,
            3531,
            3319,
            1996,
            4987,
            1035,
            6254,
            1012,
            11135,
            1012,
            102
        })
    );
}

TEST(BertTokenizerTest, HandlesContractions)
{
    cortex::embedding::BertTokenizer tokenizer(
        "models/bge-small-en-v1.5/vocab.txt"
    );

    const auto result =
        tokenizer.encode(
            "John's meeting is tomorrow."
        );

    EXPECT_EQ(
        result.input_ids,
        std::vector<int64_t>({
            101,
            2198,
            1005,
            1055,
            3116,
            2003,
            4826,
            1012,
            102
        })
    );
}

TEST(BertTokenizerTest, HandlesHyphen)
{
    cortex::embedding::BertTokenizer tokenizer(
        "models/bge-small-en-v1.5/vocab.txt"
    );

    const auto result =
        tokenizer.encode(
            "technical-interview"
        );

    EXPECT_EQ(
        result.input_ids,
        std::vector<int64_t>({
            101,
            4087,
            1011,
            4357,
            102
        })
    );
}

TEST(BertTokenizerTest, HandlesEmailAddress)
{
    cortex::embedding::BertTokenizer tokenizer(
        "models/bge-small-en-v1.5/vocab.txt"
    );

    const auto result =
        tokenizer.encode(
            "Email: TEST@example.com"
        );

    EXPECT_EQ(
        result.input_ids,
        std::vector<int64_t>({
            101,
            10373,
            1024,
            3231,
            1030,
            2742,
            1012,
            4012,
            102
        })
    );
}
TEST(BertTokenizerTest, HandlesWordPieceContinuation)
{
    cortex::embedding::BertTokenizer tokenizer(
        "models/bge-small-en-v1.5/vocab.txt"
    );

    const auto result =
        tokenizer.encode(
            "The number is 12345."
        );

    EXPECT_EQ(
        result.input_ids,
        std::vector<int64_t>({
            101,
            1996,
            2193,
            2003,
            13138,
            19961,
            1012,
            102
        })
    );
}

TEST(BertTokenizerTest, HandlesMultipleSpaces)
{
    cortex::embedding::BertTokenizer tokenizer(
        "models/bge-small-en-v1.5/vocab.txt"
    );

    const auto result =
        tokenizer.encode(
            "This    has     multiple spaces."
        );

    EXPECT_EQ(
        result.input_ids,
        std::vector<int64_t>({
            101,
            2023,
            2038,
            3674,
            7258,
            1012,
            102
        })
    );
}

TEST(BertTokenizerTest, HandlesNewlines)
{
    cortex::embedding::BertTokenizer tokenizer(
        "models/bge-small-en-v1.5/vocab.txt"
    );

    const auto result =
        tokenizer.encode(
            "Hello\nworld\nthis is a test."
        );

    EXPECT_EQ(
        result.input_ids,
        std::vector<int64_t>({
            101,
            7592,
            2088,
            2023,
            2003,
            1037,
            3231,
            1012,
            102
        })
    );
}