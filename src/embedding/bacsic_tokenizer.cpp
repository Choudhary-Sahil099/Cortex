#include "embedding/basic_tokenizer.hpp"
#include <cctype>

namespace cortex::embedding
{
    bool BasicTokenizer::isWhiteSpace(char character)
    {
        return std::isspace(static_cast<unsigned char>(character)) != 0;
    }

    bool BasicTokenizer::isPunctuation(char character)
    {
        const unsigned char value = static_cast<unsigned char>(character);
        return std::ispunct(value) != 0; // if char is ! return bool
    }

    std::vector<std::string>BasicTokenizer::tokenizer(const std::string &text) const
    {
        std::vector<std::string> tokens;

        std::string current;

        for (const char character : text)
        {
            if (isWhiteSpace(character))
            {
                if (!current.empty())
                {
                    tokens.push_back(std::move(current));
                    current.clear();
                }
                continue;
            }

            if (isPunctuation(character))
            {
                if (!current.empty())
                {
                    tokens.push_back(std::move(current));
                    current.clear();
                }
                tokens.emplace_back(1,character);
                continue;
            }
            current += character;
        }

        if (!current.empty())
        {
            tokens.push_back(
                std::move(current));
        }

        return tokens;
    }

}