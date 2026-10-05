#pragma once

#include "llm/llm.hpp"
#include "llm/generation.hpp"

#include <string>

namespace cortex::llm {

class LocalLLM : public LLM {
public:
    LocalLLM(
        std::string server_url,
        GenerationConfig config = {}
    );

    std::string generate(
        const std::string& prompt
    ) const override;

private:
    std::string server_url_;
    GenerationConfig config_;
};

}


// the json structure
/*
{
  "messages": [
    {
      "role": "user",
      "content": "When is my technical interview?"
    }
  ],
  "max_tokens": 512,
  "temperature": 0.2,
  "reasoning_effort": "low"
}
  */