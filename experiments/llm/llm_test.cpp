#include "llm/local_llm.hpp"

#include <iostream>

int main()
{
    cortex::llm::GenerationConfig config;

    config.max_tokens = 512;
    config.temperature = 0.2f;
    config.reasoning_effort = "low";

    cortex::llm::LocalLLM llm(
        "http://127.0.0.1:8080",
        config
    );

    const std::string prompt =
        "You are answering questions about emails.\n\n"
        "Email context:\n"
        "The technical interview is scheduled for Monday at 10 AM.\n\n"
        "User question:\n"
        "When is my technical interview?";

    const std::string answer =
        llm.generate(prompt);

    std::cout << "LLM Answer:\n";
    std::cout << answer << '\n';

    return 0;
}