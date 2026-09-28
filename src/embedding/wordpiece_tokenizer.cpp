#include "embedding/wordpiece_tokenizer.hpp"

#include <fstream>
#include <stdexcept>
#include <string>

namespace cortex::embedding
{

    WordPieceTokenizer::WordPieceTokenizer(
        const std::string &vocab_path)
    {
        std::ifstream file(vocab_path);

        if (!file)
        {
            throw std::runtime_error(
                "Failed to open vocabulary file: " +
                vocab_path);
        }

        std::string token;
        std::int64_t id = 0;

        while (std::getline(file, token))
        {

            if (!token.empty() &&
                token.back() == '\r')
            {

                token.pop_back();
            }

            vocabulary_.emplace(
                token,
                id);

            ++id;
        }

        if (vocabulary_.empty())
        {
            throw std::runtime_error(
                "Vocabulary is empty");
        }
    }
    std::vector<std::string>
    WordPieceTokenizer::tokenizeWord(
        const std::string &word) const
    {
        if (word.empty())
        {
            return {};
        }

        if (word.size() > max_input_chars_per_word_)
        {
            return {unknown_token_};
        }

        std::vector<std::string> tokens;

        std::size_t start = 0;

        while (start < word.size())
        {

            std::size_t end = word.size();

            std::string current_token;

            bool found = false;

            while (start < end)
            {

                std::string substring =
                    word.substr(
                        start,
                        end - start);

                if (start > 0)
                {
                    substring =
                        continuation_prefix_ +
                        substring;
                }

                if (vocabulary_.contains(substring))
                {

                    current_token =
                        std::move(substring);

                    found = true;

                    break;
                }

                --end;
            }

            if (!found)
            {
                return {unknown_token_};
            }

            tokens.push_back(
                std::move(current_token));

            start = end;
        }

        return tokens;
    }
    std::vector<int64_t>
    WordPieceTokenizer::tokenIds(
        const std::vector<std::string> &tokens) const
    {
        std::vector<int64_t> ids;

        ids.reserve(tokens.size());

        for (const auto &token : tokens)
        {

            const auto iterator =
                vocabulary_.find(token);

            if (iterator == vocabulary_.end())
            {

                throw std::runtime_error(
                    "Token missing from vocabulary: " +
                    token);
            }

            ids.push_back(
                iterator->second);
        }

        return ids;
    }
}