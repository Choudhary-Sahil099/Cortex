#pragma once

#include <string>
#include <vector>

namespace cortex::embedding{
    class BasicTokenizer{
        public:
        std::vector<std::string> tokenizer(const std::string& text)const;
        
        private:
        static bool isWhiteSpace(char character);
        static bool isPunctuation(char character);
    };
}