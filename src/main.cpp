#include "app/cortex_app.hpp"
#include "embedding/bgeEmbedder.hpp"
#include "llm/local_llm.hpp"

#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>

namespace
{

    std::string getEnvironmentVariable(
        const char *name)
    {
        const char *value = std::getenv(name);

        if (value == nullptr || *value == '\0')
        {
            throw std::runtime_error(
                std::string("Required environment variable is not set: ") +
                name);
        }

        return value;
    }

    void printUsage()
    {
        std::cout
            << "Cortex - Local Email RAG Engine\n\n"
            << "Usage:\n"
            << "  cortex index <email-directory>\n"
            << "  cortex ask <question>\n"
            << "  cortex help\n";
    }

}

int main(
    int argc,
    char *argv[])
{
    try
    {
        if (argc < 2)
        {
            printUsage();
            return 1;
        }

        const std::string command = argv[1];

        if (command == "help")
        {
            printUsage();
            return 0;
        }

        if (command != "index" && command != "ask")
        {
            std::cerr << "Unknown command: " << command << "\n\n";
            printUsage();
            return 1;
        }

        if (argc < 3)
        {
            std::cerr << "Missing argument for command: " << command << "\n\n";
            printUsage();
            return 1;
        }

        const std::string model_path = getEnvironmentVariable("CORTEX_MODEL_PATH");

        const std::string vocab_path = getEnvironmentVariable("CORTEX_VOCAB_PATH");

        const std::string index_path = getEnvironmentVariable("CORTEX_INDEX_PATH");

        const std::string llm_url = getEnvironmentVariable("CORTEX_LLM_URL");

        cortex::embedding::BGEEmbedder embedder(model_path, vocab_path);

        cortex::llm::LocalLLM llm(llm_url);

        cortex::app::CortexApp app(embedder, llm, index_path);

        if (command == "index")
        {
            const std::string directory = argv[2];

            std::cout << "Indexing emails from: " << directory << "\n";
            app.indexDirectory(directory);

            std::cout << "Indexing complete.\n"<< "Indexed vectors: " << app.indexSize() << "\n";
            return 0;
        }

        if (command == "ask")
        {
            std::string question = std::string(argv[2]);

            for (int i = 3; i < argc; ++i)
            {
                question += " ";
                question += std::string(argv[i]);
            }

            std::cout<< "Question: "<< question<< "\n\n";

            const auto result =app.askWithSources(question, 5);

            std::cout << "Answer:\n"<< result.answer << "\n\n";

            std::cout << "Sources:\n";

            for (const auto& source : result.sources)
            {
                std::cout<< "- Email: "<< source.email_id<< "\n"<< "  Thread: "<< source.thread_id<< "\n"<< "  Content: "<< source.text<< "\n\n";
            }
            return 0;
        }
    }
    catch (const std::exception &error)
    {
        std::cerr << "Cortex error: " << error.what() << "\n";
        return 1;
    }

    return 0;
}