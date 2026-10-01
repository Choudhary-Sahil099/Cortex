#include "embedding/bert_normalizer.hpp"

#include <cctype>

// if u want to remove use namespace std for ur convinenece;
namespace cortex::embedding{
    std::string BertNormalizer::normalize(const std::string& text)const{
        std::string normalized;

        normalized.reserve(text.size()); // assign the space

        for(const char c : text){
            const unsigned char value = static_cast<unsigned char>(c);
            normalized += static_cast<char>(std::tolower(value));
        }
        return normalized;
    }
}