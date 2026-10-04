#pragma once

#include "llm/llm.hpp"
#include "retrival/context_builder.hpp"
#include "retrival/retriever.hpp"

#include <string>

namespace cortex::rag {

    class RAGPipeline {
        public:
            RAGPipeline(
                const retrieval::Retriever& retriever,
                const retrieval::ContextBuilder& context_builder,
                const llm::LLM& llm
            );

            std::string ask(
                const std::string& question,
                std::size_t k
            ) const;

        private:
            const retrieval::Retriever& retriever_;
            const retrieval::ContextBuilder& context_builder_;
            const llm::LLM& llm_;
    };

}