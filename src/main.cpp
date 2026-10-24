#include "app/cortex_app.hpp"
#include "embedding/bgeEmbedder.hpp"
#include "llm/local_llm.hpp"
#include "app/enviroment.hpp"
#include "evaluation/retrieval_metric.hpp"
#include <fstream>
#include <unordered_map>
#include <unordered_set>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <algorithm>
#include <vector>
#include <utility>

namespace
{
    void printUsage()
    {
        std::cout
            << "Cortex - Local Email RAG Engine\n\n"
            << "Usage:\n"
            << "  cortex index <email-directory>\n"
            << "  cortex ask <question>\n"
            << "  cortex evaluate <dataset-path>\n"
            << "  cortex inspect\n"
            << "  cortex help\n";
    }

    struct EvaluationQuery
    {
        std::string query;
        std::unordered_set<std::string> relevant_chunk_ids;
    };

    std::vector<EvaluationQuery> loadEvaluationDataset(
        const std::string &path)
    {
        std::ifstream file(path);

        if (!file)
        {
            throw std::runtime_error(
                "Could not open evaluation dataset: " + path);
        }

        std::vector<EvaluationQuery> queries;
        std::string line;
        std::size_t line_number = 0;

        while (std::getline(file, line))
        {
            ++line_number;

            if (line.empty())
            {
                continue;
            }

            const std::size_t tab = line.find('\t');

            if (tab == std::string::npos || tab == 0)
            {
                throw std::runtime_error(
                    "Invalid dataset format at line " +
                    std::to_string(line_number) +
                    ". Expected query<TAB>chunk_id[,chunk_id...]");
            }

            EvaluationQuery item;
            item.query = line.substr(0, tab);

            const std::string ids = line.substr(tab + 1);
            std::size_t start = 0;

            while (start < ids.size())
            {
                const std::size_t comma = ids.find(',', start);

                const std::string id = ids.substr(
                    start,
                    comma == std::string::npos
                        ? std::string::npos
                        : comma - start);

                if (!id.empty())
                {
                    item.relevant_chunk_ids.insert(id);
                }

                if (comma == std::string::npos)
                {
                    break;
                }

                start = comma + 1;
            }

            if (item.relevant_chunk_ids.empty())
            {
                throw std::runtime_error(
                    "No relevant chunk IDs at line " +
                    std::to_string(line_number));
            }

            queries.push_back(std::move(item));
        }

        if (queries.empty())
        {
            throw std::runtime_error(
                "Evaluation dataset contains no queries.");
        }

        return queries;
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

        if (command != "index" && command != "ask" && command != "inspect" && command != "evaluate")
        {
            std::cerr << "Unknown command: " << command << "\n\n";
            printUsage();
            return 1;
        }

        if (command != "inspect" && argc < 3)
        {
            std::cerr << "Missing argument for command: " << command << "\n\n";
            printUsage();
            return 1;
        }
        cortex::app::loadEnvironmentFile(".env");
        const std::string model_path = cortex::app::getEnvironmentVariable("CORTEX_MODEL_PATH");

        const std::string vocab_path = cortex::app::getEnvironmentVariable("CORTEX_VOCAB_PATH");

        const std::string index_path = cortex::app::getEnvironmentVariable("CORTEX_INDEX_PATH");

        const std::string llm_url = cortex::app::getEnvironmentVariable("CORTEX_LLM_URL");

        cortex::embedding::BGEEmbedder embedder(model_path, vocab_path);

        cortex::llm::LocalLLM llm(llm_url);

        cortex::app::CortexApp app(embedder, llm, index_path);

        if (command == "evaluate")
        {
            const auto dataset = loadEvaluationDataset(argv[2]);
            constexpr std::size_t k = 5;

            struct Metrics
            {
                double recall = 0.0;
                double rr = 0.0;
                double ndcg = 0.0;
            };

            Metrics semantic_total;
            Metrics hybrid_total;

            const auto evaluate_results = [&](const auto &results,
                                              const EvaluationQuery &item)
            {
                std::vector<std::string> ranked_ids;
                ranked_ids.reserve(results.size());

                for (const auto &result : results)
                {
                    if (result.record == nullptr)
                    {
                        continue;
                    }

                    const auto it = result.record->metadata.find("chunk_id");
                    if (it != result.record->metadata.end())
                    {
                        ranked_ids.push_back(it->second);
                    }
                }

                std::unordered_map<std::string, double> relevance;
                for (const auto &id : item.relevant_chunk_ids)
                {
                    relevance.emplace(id, 1.0);
                }

                Metrics metrics;
                metrics.recall =
                    cortex::evaluation::RetrievalMetrics::recallAtK(
                        ranked_ids, item.relevant_chunk_ids, k);
                metrics.rr =
                    cortex::evaluation::RetrievalMetrics::reciprocalRank(
                        ranked_ids, item.relevant_chunk_ids);
                metrics.ndcg =
                    cortex::evaluation::RetrievalMetrics::ndcgAtK(
                        ranked_ids, relevance, k);

                return std::pair{metrics, ranked_ids};
            };

            std::cout << "Cortex Retrieval Comparison\n"
                      << "Queries: " << dataset.size() << "\n"
                      << "K: " << k << "\n\n";

            for (const auto &item : dataset)
            {
                const auto semantic_results = app.search(item.query, k);
                const auto hybrid_results = app.hybridSearch(item.query, k);

                const auto [semantic, semantic_ids] =
                    evaluate_results(semantic_results, item);
                const auto [hybrid, hybrid_ids] =
                    evaluate_results(hybrid_results, item);

                semantic_total.recall += semantic.recall;
                semantic_total.rr += semantic.rr;
                semantic_total.ndcg += semantic.ndcg;

                hybrid_total.recall += hybrid.recall;
                hybrid_total.rr += hybrid.rr;
                hybrid_total.ndcg += hybrid.ndcg;

                std::cout << "Query: " << item.query << "\n";

                std::cout << "  Semantic-only: ";
                for (const auto &id : semantic_ids)
                {
                    std::cout << id << " ";
                }
                std::cout << "\n"
                          << "    Recall@5: " << semantic.recall
                          << ", RR: " << semantic.rr
                          << ", nDCG@5: " << semantic.ndcg << "\n";

                std::cout << "  Hybrid:        ";
                for (const auto &id : hybrid_ids)
                {
                    std::cout << id << " ";
                }
                std::cout << "\n"
                          << "    Recall@5: " << hybrid.recall
                          << ", RR: " << hybrid.rr
                          << ", nDCG@5: " << hybrid.ndcg << "\n\n";
            }

            const double count = static_cast<double>(dataset.size());

            const double semantic_recall = semantic_total.recall / count;
            const double hybrid_recall = hybrid_total.recall / count;

            std::cout << "=== Aggregate Results ===\n"
                      << "Semantic-only Mean Recall@5: "
                      << semantic_recall << "\n"
                      << "Hybrid Mean Recall@5:        "
                      << hybrid_recall << "\n"
                      << "Semantic-only MRR:            "
                      << semantic_total.rr / count << "\n"
                      << "Hybrid MRR:                   "
                      << hybrid_total.rr / count << "\n"
                      << "Semantic-only Mean nDCG@5:    "
                      << semantic_total.ndcg / count << "\n"
                      << "Hybrid Mean nDCG@5:           "
                      << hybrid_total.ndcg / count << "\n";

            if (semantic_recall > 0.0)
            {
                std::cout << "Relative Recall@5 improvement: "
                          << ((hybrid_recall - semantic_recall) / semantic_recall) * 100.0
                          << "%\n";
            }
            else
            {
                std::cout << "Relative Recall@5 improvement: undefined "
                             "(semantic-only Recall@5 is zero)\n";
            }

            return 0;
        }

        
        if (command == "inspect")
        {
            if (argc < 3)
            {
                std::cerr << "Usage: cortex inspect <query>\n";
                return 1;
            }

            std::string query = argv[2];

            for (int i = 3; i < argc; ++i)
            {
                query += " ";
                query += argv[i];
            }

            const std::size_t k =
                app.indexSize() > 0 ? app.indexSize() : 1;

            const auto print_results =
                [](const std::string &title,
                   const auto &results)
            {
                std::cout << "\n=== " << title << " ===\n";

                std::size_t rank = 1;

                for (const auto &result : results)
                {
                    if (result.record == nullptr)
                    {
                        continue;
                    }

                    const auto &metadata = result.record->metadata;

                    const auto chunk_it = metadata.find("chunk_id");
                    const auto email_it = metadata.find("email_id");
                    const auto text_it = metadata.find("text");

                    std::cout << "Rank: " << rank++ << '\n';
                    std::cout << "Vector ID: " << result.id << '\n';

                    if (chunk_it != metadata.end())
                    {
                        std::cout << "Chunk ID: "
                                  << chunk_it->second << '\n';
                    }

                    if (email_it != metadata.end())
                    {
                        std::cout << "Email ID: "
                                  << email_it->second << '\n';
                    }

                    std::cout << "Distance/score: "
                              << result.distance << '\n';

                    if (text_it != metadata.end())
                    {
                        std::cout << "Text: " << text_it->second << '\n';
                    }

                    std::cout << "-----------------------------\n";
                }
            };

            const auto semantic_results = app.search(query, k);
            const auto hybrid_results = app.hybridSearch(query, k);

            print_results("SEMANTIC-ONLY RESULTS", semantic_results);
            print_results("HYBRID RESULTS", hybrid_results);

            return 0;
        }


        if (command == "index")
        {
            const std::string directory = argv[2];

            std::cout << "Indexing emails from: " << directory << "\n";
            app.indexDirectory(directory);

            std::cout << "Indexing complete.\n"
                      << "Indexed vectors: " << app.indexSize() << "\n";
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

            std::cout << "Question: " << question << "\n\n";

            const auto result = app.askWithSources(question, 5);

            std::cout << "Answer:\n"
                      << result.answer << "\n\n";

            std::cout << "Sources:\n";

            for (const auto &source : result.sources)
            {
                std::cout << "- Email: " << source.email_id << "\n"
                          << "  Thread: " << source.thread_id << "\n"
                          << "  Content: " << source.text << "\n\n";
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