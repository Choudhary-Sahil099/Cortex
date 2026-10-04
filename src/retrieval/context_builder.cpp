#include "retrival/context_builder.hpp"

#include <sstream>

namespace cortex::retrieval {

std::string ContextBuilder::build(
    const std::vector<index::VectorSearchResult>& results
) const
{
    std::ostringstream context;

    std::size_t context_index = 0; // this is to avoid the unecessary counting of the chunks that do not make it to the context

    for (const auto& result : results) {

        const auto* record = result.record;

        if (record == nullptr) {
            continue;
        }
        // find if present else we will skip it 
        const auto email_it =
            record->metadata.find("email_id");

        const auto thread_it =
            record->metadata.find("thread_id");

        const auto text_it =
            record->metadata.find("text");

        if (
            email_it == record->metadata.end() ||
            thread_it == record->metadata.end() ||
            text_it == record->metadata.end()
        ) {
            continue;
        }

        ++context_index;

        context << "Email "<< context_index<< "\n";

        context << "Email ID: "<< email_it->second<< "\n";

        context << "Thread ID: "<< thread_it->second<< "\n";

        context << "Content:\n";

        context << text_it->second<< "\n";

        context << "\n---\n\n";
    }

    return context.str();
}

}