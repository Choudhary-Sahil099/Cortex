#pragma once

#include <string>

namespace cortex::llm {

class LLM {
public:
    virtual ~LLM() = default;

    virtual std::string generate(
        const std::string& prompt
    ) const = 0;
};

}