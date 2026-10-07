#include "app/cortex_app.hpp"
#include <utility>
#include <filesystem>
#include <stdexcept>
namespace cortex::app
{

    CortexApp::CortexApp(const embedding::Embedder &embedder, const llm::LLM &llm, std::string index_path, std::size_t chunk_size) : embedder_(embedder), llm_(llm), parser_(), cleaner_(), chunker_(chunk_size), embedding_pipeline_(embedder_), index_manager_(std::move(index_path)), index_(index_manager_.loadOrCreate(embedder_.dimension())), email_indexer_(parser_, cleaner_, chunker_,embedding_pipeline_, index_),retriever_(embedder_,index_),context_builder_(), rag_pipeline_(retriever_,context_builder_,llm_){}

    void CortexApp::indexEmail(const std::string &raw_email)
    {
        email_indexer_.addEmail(raw_email);
    }

    std::string CortexApp::ask(const std::string &question,std::size_t k) const
    {
        return rag_pipeline_.ask(question,k);
    }

    cortex::rag::RAGResult CortexApp::askWithSources(const std::string& question,std::size_t k) const{
        return rag_pipeline_.askWithSources(
            question,
            k
        );
    }

    std::size_t CortexApp::indexSize() const
    {
        return index_.size();
    }

    void CortexApp::indexDirectory(const std::string &directory)
    {
        if (!std::filesystem::exists(directory))
        {
            throw std::runtime_error(
                "Email directory does not exist: " + directory);
        }

        if (!std::filesystem::is_directory(directory))
        {
            throw std::runtime_error(
                "Path is not a directory: " + directory);
        }

        for (const auto &entry :
             std::filesystem::directory_iterator(directory))
        {
            if (!entry.is_regular_file())
            {
                continue;
            }

            if (entry.path().extension() != ".eml")
            {
                continue;
            }

            const auto raw_email =
                file_loader_.load(
                    entry.path().string());

            indexEmail(raw_email);
        }
        index_manager_.save(index_);
    }
}