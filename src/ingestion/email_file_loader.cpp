#include "ingestion/email_file_loader.hpp"

#include <fstream>
#include <sstream>
#include <stdexcept>

namespace cortex::ingestion
{

std::string EmailFileLoader::load(
    const std::string& path
) const
{
    std::ifstream file(path);

    if (!file.is_open())
    {
        throw std::runtime_error(
            "Failed to open email file: " + path
        );
    }

    std::ostringstream buffer;

    buffer << file.rdbuf();

    return buffer.str();
}

}