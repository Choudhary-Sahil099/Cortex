#include "email/email_chunker.hpp"

#include <gtest/gtest.h>
#include <stdexcept>
TEST(EmailChunkerTest, SmallEmailProducesSingleChunk)
{
    cortex::email::EmailDocument email;

    email.id = "email_001";
    email.thread_id = "thread_001";
    email.sender = "Sahil@example.com";

    email.body =
        "Hello Sujal.\n"
        "The interview is tomorrow.\n";

    cortex::email::EmailChunker chunker(500);

    const auto chunks =
        chunker.chunk(email);

    ASSERT_EQ(
        chunks.size(),
        std::size_t{ 1 }
    );

    EXPECT_EQ(
        chunks[0].email_id,
        "email_001"
    );

    EXPECT_EQ(
        chunks[0].thread_id,
        "thread_001"
    );

    EXPECT_EQ(
        chunks[0].index,
        std::size_t{ 0 }
    );

    EXPECT_EQ(
        chunks[0].text,
        email.body
    );

    EXPECT_EQ(
        chunks[0].id,
        "email_001_0"
    );
}

TEST(EmailChunkerTest, LargeEmailProducesMultipleChunks)
{
    cortex::email::EmailDocument email;

    email.id = "email_002";
    email.thread_id = "thread_002";
    email.sender = "Sahil@example.com";

    email.body = "12345678901234567890";

    cortex::email::EmailChunker chunker(10);

    const auto chunks =
        chunker.chunk(email);

    ASSERT_EQ(
        chunks.size(),
        std::size_t{ 2 }
    );

    EXPECT_EQ(
        chunks[0].index,
        std::size_t{ 0 }
    );

    EXPECT_EQ(
        chunks[0].id,
        "email_002_0"
    );

    EXPECT_EQ(
        chunks[0].text,
        "1234567890"
    );

    EXPECT_EQ(
        chunks[1].index,
        std::size_t{ 1 }
    );

    EXPECT_EQ(
        chunks[1].id,
        "email_002_1"
    );

    EXPECT_EQ(
        chunks[1].text,
        "1234567890"
    );
}

TEST(EmailChunkerTest, RejectsZeroChunkSize)
{
    EXPECT_THROW(
        cortex::email::EmailChunker chunker(0),
        std::invalid_argument
    );
}