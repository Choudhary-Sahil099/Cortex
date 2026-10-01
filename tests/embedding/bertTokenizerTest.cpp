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