#include "email/email_document.hpp"

#include <gtest/gtest.h>

namespace {


    // created a valid email for testing th e document acceptance
    cortex::email::EmailDocument makeValidEmail()
    {
        cortex::email::EmailDocument email;

        email.id = "email_001";
        email.thread_id = "thread_001";
        email.sender = "sahil@example.com";
        email.recipients = { "sujal@example.com" };
        email.subject = "Interview";
        email.body = "Your interview is scheduled for Monday.";
        email.timestamp = "2026-09-25T10:00:00";

        return email;
    }

}

TEST(EmailDocumentTest, ValidEmailReturnsTrue)
{
    const auto email = makeValidEmail();

    EXPECT_TRUE(email.valid());
}

TEST(EmailDocumentTest, MissingIdReturnsFalse)
{
    auto email = makeValidEmail();
    email.id.clear();

    EXPECT_FALSE(email.valid());
}

TEST(EmailDocumentTest, MissingThreadIdReturnsFalse)
{
    auto email = makeValidEmail();
    email.thread_id.clear();

    EXPECT_FALSE(email.valid());
}

TEST(EmailDocumentTest, MissingSenderReturnsFalse)
{
    auto email = makeValidEmail();
    email.sender.clear();

    EXPECT_FALSE(email.valid());
}

TEST(EmailDocumentTest, MissingBodyReturnsFalse)
{
    auto email = makeValidEmail();
    email.body.clear();

    EXPECT_FALSE(email.valid());
}