
#pragma once

#include <string>

//namespace cortex::retrieval
namespace cortex::retrieval {

class QueryTransformer {
public:
    //normalize query before embedding and retrival
    static std::string transform(const std::string& query);
};

} 
