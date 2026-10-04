#include "llm/llm.hpp"

#include <gtest/gtest.h>

class TestLLM : public cortex::llm::LLM {
public:

    std::string generate(
        const std::string& prompt
    ) const override
    {
        return "Test answer";
    }
};


TEST(LLMTest, GeneratesResponse)
{
    TestLLM llm;

    const std::string response =
        llm.generate(
            "When is my interview?"
        );

    EXPECT_EQ(
        response,
        "Test answer"
    );
}