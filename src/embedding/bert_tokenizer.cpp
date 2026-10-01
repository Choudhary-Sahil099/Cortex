#include "embedding/bert_tokenizer.hpp"

namespace cortex::embedding {

BertTokenizer::BertTokenizer(const std::string& vocab_path): wordpiece_tokenizer_(vocab_path){}

TokenizedInput BertTokenizer::encode(const std::string& text) const
{
    TokenizedInput result;

    const std::string normalized = normalizer_.normalize(text);

    const auto basic_tokens = basic_tokenizer_.tokenizer(normalized);

    std::vector<std::string> wordpiece_tokens;

    for (const auto& word : basic_tokens) {

        const auto pieces = wordpiece_tokenizer_.tokenizeWord(word);

        wordpiece_tokens.insert(
            wordpiece_tokens.end(),
            pieces.begin(),
            pieces.end()
        );
    }

    const int64_t cls_id = wordpiece_tokenizer_.tokenId("[CLS]");

    const int64_t sep_id = wordpiece_tokenizer_.tokenId("[SEP]");

    result.input_ids.push_back(cls_id);

    for (const auto& token : wordpiece_tokens) {

        result.input_ids.push_back(wordpiece_tokenizer_.tokenId(token));
    }

    result.input_ids.push_back(sep_id);
    result.attention_mask.resize(result.input_ids.size(),1);
    result.token_type_ids.resize(result.input_ids.size(),0);

    return result;
}

}