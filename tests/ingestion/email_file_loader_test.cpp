#include "ingestion/email_file_loader.hpp"

#include <gtest/gtest.h>

#include <cstdio>
#include <fstream>
#include <string>

TEST(EmailFileLoaderTest, LoadsEmailFile)
{
    const std::string path =
        "test_email_loader.eml";

    {
        std::ofstream file(path);

        ASSERT_TRUE(file.is_open());

        file << "From: recruiter@example.com\n";
        file << "To: sahil@example.com\n";
        file << "Subject: Technical Interview\n";
        file << "\n";
        file << "Your technical interview is on Monday.";
    }

    cortex::ingestion::EmailFileLoader loader;

    const std::string content =
        loader.load(path);

    EXPECT_EQ(
        content,
        "From: recruiter@example.com\n"
        "To: sahil@example.com\n"
        "Subject: Technical Interview\n"
        "\n"
        "Your technical interview is on Monday."
    );

    std::remove(path.c_str());
}

TEST(EmailFileLoaderTest, ThrowsWhenFileDoesNotExist)
{
    const std::string path =
        "file_that_does_not_exist.eml";

    std::remove(path.c_str());

    cortex::ingestion::EmailFileLoader loader;

    EXPECT_THROW(
        loader.load(path),
        std::runtime_error
    );
}