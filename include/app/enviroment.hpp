#pragma once

#include <string>

namespace cortex::app
{

void loadEnvironmentFile(
    const std::string& path
);

std::string getEnvironmentVariable(
    const std::string& name
);

}