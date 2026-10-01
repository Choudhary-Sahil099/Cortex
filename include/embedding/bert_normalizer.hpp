#pragma once 

#include<string>
namespace cortex::embedding{
    class BertNormalizer{
        public:
        std::string normalize(const std::string &next)const;

    };
}