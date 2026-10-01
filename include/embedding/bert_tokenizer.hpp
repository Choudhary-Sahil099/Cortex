#pragma once

#include "embedding/bert_normalizer.hpp"
#include "embedding/basic_tokenizer.hpp"
#include "embedding/wordpiece_tokenizer.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace cortex::embedding {

struct TokenizedInput {
    std::vector<int64_t> input_ids;
    std::vector<int64_t> attention_mask;
    std::vector<int64_t> token_type_ids;
};

class BertTokenizer {
public:

    explicit BertTokenizer(const std::string& vocab_path);
    TokenizedInput encode(const std::string& text) const;

private:

    BertNormalizer normalizer_;
    BasicTokenizer basic_tokenizer_;
    WordPieceTokenizer wordpiece_tokenizer_;
};

}