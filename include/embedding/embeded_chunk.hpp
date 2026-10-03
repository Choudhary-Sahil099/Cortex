#pragma once

#include "email/email_chunk.hpp"

#include <cstddef>
#include <string>
#include <vector>

namespace cortex::embedding
{
    struct EmbeddedChunk
    {
        std::string id;
        std::string email_id;
        std::string thread_id;
        std::string text;

        std::vector<float> embedding;
        std::size_t index;
    };
}