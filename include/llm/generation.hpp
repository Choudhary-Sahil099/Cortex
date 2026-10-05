#pragma once

#include <cstddef>
#include <string>

namespace cortex::llm {

struct GenerationConfig {
    std::size_t max_tokens = 512; // the max token are static as the most token used during the testing
    float temperature = 0.2f; // controls randomness
    std::string reasoning_effort = "low"; // reasoning behaviour
};

}