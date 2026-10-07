#include "rag/rag_pipeline.hpp"

#include <stdexcept>

namespace cortex::rag {

    RAGPipeline::RAGPipeline(const retrieval::Retriever& retriever,const retrieval::ContextBuilder& context_builder,const llm::LLM& llm): retriever_(retriever),context_builder_(context_builder),llm_(llm){}

    std::string RAGPipeline::ask(const std::string& question,std::size_t k) const
    {
        return askWithSources(question, k).answer;
    }

    cortex::rag::RAGResult RAGPipeline::askWithSources(const std::string& question,std::size_t k) const
    {
        const auto results =
            retriever_.search(question, k);

        const std::string context =
            context_builder_.build(results);

        const std::string prompt =
            "Answer the user's question using only the provided email context.\n\n"
            "Email context:\n" +
            context +
            "\n\n"
            "User question:\n" +
            question;

        const std::string answer =
            llm_.generate(prompt);

        return RAGResult{
            answer,
            context_builder_.buildSources(results)
        };
    }

}