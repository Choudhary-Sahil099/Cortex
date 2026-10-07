#include "app/enviroment.hpp"

#include <cstdlib>
#include <fstream>
#include <stdexcept>
#include <string>

#ifdef _WIN32
#include <windows.h>
#endif

namespace cortex::app
{

namespace
{

std::string trim(const std::string& value)
{
    const auto first = value.find_first_not_of(" \t\r\n");

    if (first == std::string::npos)
    {
        return {};
    }

    const auto last = value.find_last_not_of(" \t\r\n");

    return value.substr(
        first,
        last - first + 1
    );
}

void setEnvironmentVariable(
    const std::string& name,
    const std::string& value
)
{
#ifdef _WIN32
    if (_putenv_s(name.c_str(), value.c_str()) != 0)
    {
        throw std::runtime_error(
            "Failed to set environment variable: " + name
        );
    }
#else
    if (setenv(name.c_str(), value.c_str(), 1) != 0)
    {
        throw std::runtime_error(
            "Failed to set environment variable: " + name
        );
    }
#endif
}

}

void loadEnvironmentFile(
    const std::string& path
)
{
    std::ifstream file(path);

    if (!file.is_open())
    {
        throw std::runtime_error(
            "Failed to open environment file: " + path
        );
    }

    std::string line;

    while (std::getline(file, line))
    {
        line = trim(line);

        if (line.empty() || line[0] == '#')
        {
            continue;
        }

        const auto separator = line.find('=');

        if (separator == std::string::npos)
        {
            throw std::runtime_error(
                "Invalid environment line: " + line
            );
        }

        const std::string name =
            trim(line.substr(0, separator));

        const std::string value =
            trim(line.substr(separator + 1));

        if (name.empty())
        {
            throw std::runtime_error(
                "Environment variable name cannot be empty"
            );
        }

        setEnvironmentVariable(name, value);
    }
}

std::string getEnvironmentVariable(
    const std::string& name
)
{
    const char* value = std::getenv(name.c_str());

    if (value == nullptr || *value == '\0')
    {
        throw std::runtime_error(
            "Required environment variable is not set: " + name
        );
    }

    return value;
}

}