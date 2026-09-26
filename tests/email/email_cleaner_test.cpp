#include "email/email_cleaner.hpp"

#include <gtest/gtest.h>

TEST(EmailCleanerTest, RemovesTrailingWhitespace)
{
    cortex::email::EmailDocument email;

    email.id = "email_001";
    email.thread_id = "thread_001";
    email.sender = "sahil@example.com";
    email.recipients = {
        "sujal@example.com"
    };
    email.subject = "Interview";
    email.body =
        "Hello Sujal.    \n"
        "The interview is tomorrow.   \n";

    cortex::email::EmailCleaner cleaner;

    const auto cleaned =
        cleaner.clean(email);

    EXPECT_EQ(
        cleaned.body,
        "Hello Sujal.\n"
        "The interview is tomorrow.\n"
    );
}

TEST(EmailCleanerTest, PreservesMetadata)
{
    cortex::email::EmailDocument email;

    email.id = "email_002";
    email.thread_id = "thread_002";
    email.sender = "sahil@example.com";
    email.recipients = {
        "sujal@example.com",
        "shivanshu@example.com"
    };
    email.subject = "Meeting";
    email.timestamp = "2026-09-26T10:00:00";
    email.body = "Hello   \n";

    cortex::email::EmailCleaner cleaner;

    const auto cleaned =
        cleaner.clean(email);

    EXPECT_EQ(cleaned.id, email.id);
    EXPECT_EQ(cleaned.thread_id, email.thread_id);
    EXPECT_EQ(cleaned.sender, email.sender);
    EXPECT_EQ(cleaned.recipients, email.recipients);
    EXPECT_EQ(cleaned.subject, email.subject);
    EXPECT_EQ(cleaned.timestamp, email.timestamp);
}
TEST(EmailCleanerTest, RemovesExcessiveBlankLines)
{
    cortex::email::EmailDocument email;

    email.id = "email_003";
    email.thread_id = "thread_003";
    email.sender = "sahil@example.com";
    email.body =
        "Hello sujal.\n"
        "\n"
        "\n"
        "\n"
        "The interview is tomorrow.\n"
        "\n"
        "\n"
        "Regards,\n"
        "sahil\n";

    cortex::email::EmailCleaner cleaner;

    const auto cleaned =
        cleaner.clean(email);

    EXPECT_EQ(
        cleaned.body,
        "Hello sujal.\n"
        "\n"
        "The interview is tomorrow.\n"
        "\n"
        "Regards,\n"
        "sahil\n"
    );
}

TEST(EmailCleanerTest, RemovesLeadingAndTrailingBlankLines)
{
    cortex::email::EmailDocument email;

    email.id = "email_004";
    email.thread_id = "thread_004";
    email.sender = "sahil@example.com";
    email.recipients = { "sujal@example.com" };
    email.body =
        "\n"
        "\n"
        "Hello Sujal.\n"
        "\n"
        "The interview is tomorrow.\n"
        "\n"
        "Regards,\n"
        "Sahil\n"
        "\n"
        "\n";

    cortex::email::EmailCleaner cleaner;

    const auto cleaned =
        cleaner.clean(email);

    EXPECT_EQ(
        cleaned.body,
        "Hello Sujal.\n"
        "\n"
        "The interview is tomorrow.\n"
        "\n"
        "Regards,\n"
        "Sahil\n"
    );
}

TEST(EmailCleanerTest, PreservesAlreadyCleanBody)
{
    cortex::email::EmailDocument email;

    email.id = "email_005";
    email.thread_id = "thread_005";
    email.sender = "sahil@example.com";

    email.body =
        "Hello Sujal.\n"
        "\n"
        "The interview is tomorrow.\n"
        "\n"
        "Regards,\n"
        "Sahil\n";

    cortex::email::EmailCleaner cleaner;

    const auto cleaned =
        cleaner.clean(email);

    EXPECT_EQ(
        cleaned.body,
        email.body
    );
}