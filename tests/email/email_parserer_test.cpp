#include "email/email_parserer.hpp"
#include <gtest/gtest.h>

TEST(EmailParserTest, ParsesBasicEmail) {

    //spelling mistakes can cause error
    const std::string raw_email =
        "ID: email_001\n"
        "From: sahil@example.com\n"
        "To: sujal@example.com\n"
        "Subject: Interview\n"
        "Date: 2026-09-25T10:00:00\n"
        "Thread-ID: thread_123\n"
        "\n"
        "Your interview is scheduled for Monday.\n";

    cortex::email::EmailParser parser;

    const auto email = parser.parse(raw_email);

    EXPECT_EQ(email.sender, "sahil@example.com");

    ASSERT_EQ(email.recipients.size(), 1);
    EXPECT_EQ(
        email.recipients[0],
        "sujal@example.com"
    );

    EXPECT_EQ(email.subject, "Interview");

    EXPECT_EQ(
        email.timestamp,
        "2026-09-25T10:00:00"
    );

    EXPECT_EQ(
        email.thread_id,
        "thread_123"
    );

    EXPECT_EQ(
        email.body,
        "Your interview is scheduled for Monday.\n" 
    );
}

TEST(EmailParserTest, RejectsInvalidEmail)
{
    const std::string raw_email =
        "From:sahil@example.com\n"
        "Subject: Invalid content \n"
        "\n";

    cortex::email::EmailParser parser;

    EXPECT_THROW(
        parser.parse(raw_email),
        std::runtime_error
    );
}
TEST(EmailParserTest, ParsesMultipleRecipients)
{
    const std::string raw_email =
        "ID: email_002\n"
        "From: sahil@example.com\n"
        "To:                   sujal@example.com, shivanshu@example.com\n"
        "Subject: Meeting\n"
        "Date: 2026-09-25T10:00:00\n"
        "Thread-ID: thread_456\n"
        "\n"
        "Meeting is scheduled for tomorrow.\n";

    cortex::email::EmailParser parser;

    const auto email = parser.parse(raw_email);

    ASSERT_EQ(email.recipients.size(), 2);

    EXPECT_EQ(
        email.recipients[0],
        "sujal@example.com"
    );

    EXPECT_EQ(
        email.recipients[1],
        "shivanshu@example.com"
    );
}


TEST(EmailParserTest, ParsesWhiteSpaces) {
    const std::string raw_email =
        "ID: email_002\n"
        "From:         sahil@example.com\n"
        "To:     sujal@example.com, shivanshu@example.com\n"
        "Subject: Meeting\n"
        "Date: 2026-09-25T10:00:00\n"
        "Thread-ID: thread_456\n"
        "\n"
        "Meeting is scheduled for tomorrow.\n";

    cortex::email::EmailParser parser;

    const auto email = parser.parse(raw_email);

    EXPECT_EQ(email.sender, "sahil@example.com");
}

TEST(EmailParserTest, ParsesCRLFEmail)
{
    const std::string raw_email =
        "ID: email_003\r\n"
        "From:    sahil@example.com\r\n"
        "To: sujal@example.com,     shivanshu@example.com\r\n"
        "Subject: Interview\r\n"
        "Date: 2026-09-25T10:00:00\r\n"
        "Thread-ID: thread_789\r\n"
        "\r\n"
        "Your interview is scheduled for Monday.\r\n";

    cortex::email::EmailParser parser;

    const auto email = parser.parse(raw_email);

    EXPECT_EQ(email.id, "email_003");
    EXPECT_EQ(email.sender, "sahil@example.com");

    ASSERT_EQ(email.recipients.size(), 2);
    EXPECT_EQ(email.recipients[0], "sujal@example.com");
    EXPECT_EQ(email.recipients[1], "shivanshu@example.com");

    EXPECT_EQ(email.subject, "Interview");
    EXPECT_EQ(email.timestamp, "2026-09-25T10:00:00");
    EXPECT_EQ(email.thread_id, "thread_789");

    EXPECT_EQ(
        email.body,
        "Your interview is scheduled for Monday.\n"
    );
}

TEST(EmailParserTest, ParsesFoldedSubject)
{
    const std::string raw_email =
        "ID: email_004\r\n"
        "From: sahil@example.com\r\n"
        "To: sujal@example.com\r\n"
        "Subject: Interview discussion\r\n"
        " tomorrow at 10 AM\r\n"
        "Date: 2026-09-25T10:00:00\r\n"
        "Thread-ID: thread_004\r\n"
        "\r\n"
        "The interview is scheduled.\r\n";

    cortex::email::EmailParser parser;

    const auto email = parser.parse(raw_email);

    EXPECT_EQ(
        email.subject,
        "Interview discussion tomorrow at 10 AM"
    );
}