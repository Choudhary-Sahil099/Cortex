#pragma once

#include <string>

namespace cortex::ingestion
{

class EmailFileLoader
{
public:
    std::string load(
        const std::string& path
    ) const;
};

}