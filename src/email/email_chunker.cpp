#include "email/email_chunker.hpp"
#include <algorithm>
namespace cortex::email {

    EmailChunker::EmailChunker(
        std::size_t chunk_size
    )
        : chunk_size_(chunk_size)
    {
    }

    std::vector<emailChunk> EmailChunker::chunk(
        const EmailDocument& email
    ) const
    {
        std::vector<emailChunk> chunks;

        if (email.body.empty()) {
            return chunks;
        }

        std::size_t index = 0;
        std::size_t position = 0;

        while (position < email.body.size()) {

            const std::size_t length = std::min( chunk_size_, email.body.size() - position );

            emailChunk chunk;

            chunk.index = index;
            chunk.email_id = email.id;
            chunk.thread_id = email.thread_id;

            chunk.id = email.id + "_" + std::to_string(index);

            chunk.text = email.body.substr(position, length);

            chunks.push_back(
                std::move(chunk)
            );

            position += length;
            ++index;
        }

        return chunks;
    }

}