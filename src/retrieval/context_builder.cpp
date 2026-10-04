#include "retrival/context_builder.hpp"

#include <sstream>

namespace cortex::retrieval {

std::string ContextBuilder::build(
    const std::vector<index::VectorSearchResult>& results
) const
{
    std::ostringstream context;

    for (std::size_t i = 0; i < results.size(); ++i) {

        const auto* record = results[i].record; // records in the vector store 

        if (record == nullptr) continue;

        context << "Email " << (i + 1) << "\n";

        context << "Email ID: "<< record->metadata.at("email_id")<< "\n";

        context << "Thread ID: "<< record->metadata.at("thread_id")<< "\n";
        context << "Content:\n";
        context << record->metadata.at("text")<< "\n";

        if (i + 1 < results.size()) {
            context << "\n---\n\n";
        }
    }

    return context.str();
}

}