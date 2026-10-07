#pragma once

#include "index/vector_index.hpp"
#include "retrival/source.hpp"
#include <string>
#include <vector>

namespace cortex::retrieval {

class ContextBuilder {
public:
    std::string build(const std::vector<index::VectorSearchResult>& results) const;
    std::vector<Source> buildSources(const std::vector<index::VectorSearchResult>& results) const;
};

}