#pragma once

#include <string>

namespace cortex::retrieval
{

struct Source
{
    std::string email_id;
    std::string thread_id;
    std::string text;
};

}