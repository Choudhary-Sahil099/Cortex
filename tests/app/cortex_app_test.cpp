#include "app/cortex_app.hpp"
#include "embedding/embedder.hpp"
#include "llm/llm.hpp"

#include <gtest/gtest.h>

#include <cstdio>
#include <string>
#include <vector>
#include <filesystem>
#include <fstream>

namespace
{

class TestEmbedder : public cortex::embedding::Embedder
{
public:
    std::vector<float> embed(
        const std::string&
    ) const override
    {
        return {
            1.0f,
            0.0f,
            0.0f,
            0.0f
        };
    }

    std::size_t dimension() const override
    {
        return 4;
    }
};

class TestLLM : public cortex::llm::LLM
{
public:
    std::string generate(
        const std::string&
    ) const override
    {
        return "The technical interview is on Monday.";
    }
};

}

TEST(CortexAppTest, IndexesEmailAndAnswersQuestion)
{
    const std::string path =
        "test_cortex_app.cortex";

    std::remove(path.c_str());

    TestEmbedder embedder;
    TestLLM llm;

    cortex::app::CortexApp app(
        embedder,
        llm,
        path
    );

    const std::string email =
        "ID: email-001\n"
        "Thread-ID: thread-001\n"
        "From: recruiter@example.com\n"
        "To: sahil@example.com\n"
        "Subject: Technical Interview\n"
        "Date: Mon, 6 Oct 2026 10:00:00\n"
        "\n"
        "Your technical interview is scheduled for Monday.";

    app.indexEmail(email);

    EXPECT_EQ(
        app.indexSize(),
        1
    );

    const std::string answer =
        app.ask(
            "When is my technical interview?",
            1
        );

    EXPECT_EQ(
        answer,
        "The technical interview is on Monday."
    );

    std::remove(path.c_str());
}

TEST(CortexAppTest, IndexesEmailsFromDirectory)
{
    const std::string directory =
        "test_emails";

    const std::string index_path =
        "test_cortex_directory.cortex";

    std::remove(index_path.c_str());

    std::filesystem::remove_all(directory);
    std::filesystem::create_directory(directory);

    {
        std::ofstream file(
            directory + "/email1.eml"
        );

        ASSERT_TRUE(file.is_open());

        file << "ID: email-001\n";
        file << "Thread-ID: thread-001\n";
        file << "From: recruiter@example.com\n";
        file << "To: sahil@example.com\n";
        file << "Subject: Technical Interview\n";
        file << "\n";
        file << "Your technical interview is on Monday.";
    }

    {
        std::ofstream file(
            directory + "/email2.eml"
        );

        ASSERT_TRUE(file.is_open());

        file << "ID: email-002\n";
        file << "Thread-ID: thread-002\n";
        file << "From: hr@example.com\n";
        file << "To: sahil@example.com\n";
        file << "Subject: Internship\n";
        file << "\n";
        file << "Your internship application has been received.";
    }

    // This file should be ignored.
    {
        std::ofstream file(
            directory + "/ignored.txt"
        );

        ASSERT_TRUE(file.is_open());

        file << "This file should not be indexed.";
    }

    TestEmbedder embedder;
    TestLLM llm;

    cortex::app::CortexApp app(
        embedder,
        llm,
        index_path
    );

    app.indexDirectory(directory);

    EXPECT_EQ(
        app.indexSize(),
        2
    );

    std::filesystem::remove_all(directory);
    std::remove(index_path.c_str());
}