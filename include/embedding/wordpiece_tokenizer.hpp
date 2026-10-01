#pragma once

#include <cstddef>
#include <string>
#include <unordered_map>
#include <vector>

namespace cortex::embedding {

class WordPieceTokenizer {
public:

    explicit WordPieceTokenizer(
        const std::string& vocab_path
    );

    std::vector<std::string> tokenizeWord(
        const std::string& word
    ) const;

    std::vector<int64_t> tokenIds(
        const std::vector<std::string>& tokens
    ) const;

    int64_t tokenId(const std::string& token)const;

private:

    std::unordered_map<std::string, int64_t> vocabulary_;

    std::size_t max_input_chars_per_word_ = 100;

    std::string unknown_token_ = "[UNK]";

    std::string continuation_prefix_ = "##";
};

}