#pragma once

#include "retrival/source.hpp"

#include <string>
#include <vector>

namespace cortex::rag
{

struct RAGResult
{
    std::string answer;
    std::vector<retrieval::Source> sources;
};

}