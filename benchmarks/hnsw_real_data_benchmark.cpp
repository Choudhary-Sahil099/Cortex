
#include "embedding/bgeEmbedder.hpp"
#include "retrival/query_transformer.hpp"
#include "persistence/serializer.hpp"
#include "index/hnsw.hpp"
#include "vector/backend.hpp"

#include <benchmark/benchmark.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cctype>
#include <cstdlib>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace
{

    using Clock = std::chrono::steady_clock;

    constexpr std::size_t K = 10;
    constexpr std::uint64_t RANDOM_SEED = 12345;

    struct Config
    {
        std::size_t M;
        std::size_t ef_construction;
        std::size_t ef_search;
    };

    struct SourceRecord
    {
        std::size_t original_id;
        std::vector<float> values;
        cortex::core::metaData metadata;
    };

    struct Query
    {
        std::string text;
        std::vector<std::string> relevant_chunk_ids;
        std::vector<float> embedding;
        std::vector<std::string> exact_top_chunk_ids;
    };

    struct Metrics
    {
        double exact_recall_at_10 = 0.0;
        double relevant_recall_at_10 = 0.0;
        double qps = 0.0;
        double p50_us = 0.0;
        double p95_us = 0.0;
        double p99_us = 0.0;
        double build_ms = 0.0;
        double vector_payload_mb = 0.0;
    };

    std::string envOr(const char *name, const char *fallback)
    {
        const char *value = std::getenv(name);
        return value && *value ? value : fallback;
    }

    std::string trim(const std::string &value)
    {
        const auto first = value.find_first_not_of(" \t\r\n");
        if (first == std::string::npos)
        {
            return {};
        }

        const auto last = value.find_last_not_of(" \t\r\n");
        return value.substr(first, last - first + 1);
    }

    std::string chunkIdFromMetadata(
        const cortex::core::metaData &metadata,
        std::size_t fallback_id)
    {
        const auto it = metadata.find("chunk_id");

        if (it != metadata.end() && !it->second.empty())
        {
            return it->second;
        }

        return "__vector_id_" + std::to_string(fallback_id);
    }

    std::vector<std::string> splitRelevantIds(const std::string &field)
    {
        std::vector<std::string> ids;
        std::stringstream stream(field);
        std::string item;

        while (std::getline(stream, item, ','))
        {
            item = trim(item);
            if (!item.empty())
            {
                ids.push_back(item);
            }
        }

        return ids;
    }

    std::vector<Query> loadQueries(const std::string &path)
    {
        std::ifstream file(path);

        if (!file.is_open())
        {
            throw std::runtime_error("Cannot open evaluation TSV: " + path);
        }

        std::vector<Query> queries;
        std::string line;
        std::size_t line_number = 0;

        while (std::getline(file, line))
        {
            ++line_number;
            line = trim(line);

            if (line.empty() || line[0] == '#')
            {
                continue;
            }

            const auto tab = line.find('\t');

            if (tab == std::string::npos)
            {
                throw std::runtime_error(
                    "Expected query<TAB>relevant_chunk_id at TSV line " +
                    std::to_string(line_number));
            }

            Query query;
            query.text = trim(line.substr(0, tab));
            query.relevant_chunk_ids =
                splitRelevantIds(line.substr(tab + 1));

            if (query.text.empty() || query.relevant_chunk_ids.empty())
            {
                throw std::runtime_error(
                    "Empty query or relevant ID at TSV line " +
                    std::to_string(line_number));
            }

            queries.push_back(std::move(query));
        }

        if (queries.empty())
        {
            throw std::runtime_error("The evaluation TSV contains no queries.");
        }

        return queries;
    }

    std::vector<SourceRecord> copySourceRecords(
        const cortex::index::HNSWIndex &source)
    {
        std::vector<SourceRecord> records;
        const auto &stored_records = source.vector_store().records();

        records.reserve(stored_records.size());

        for (const auto &record : stored_records)
        {
            SourceRecord copied;
            copied.original_id = record.id;
            copied.values.assign(
                record.vector.data(),
                record.vector.data() + record.vector.dimension());
            copied.metadata = record.metadata;

            records.push_back(std::move(copied));
        }

        return records;
    }

    cortex::vector::Vector toVector(const std::vector<float> &values)
    {
        cortex::vector::Vector result(values.size());

        std::copy(values.begin(), values.end(), result.data());

        return result;
    }

    double percentile(
        std::vector<double> values,
        double percentile_value)
    {
        if (values.empty())
        {
            return 0.0;
        }

        std::sort(values.begin(), values.end());

        const auto index = static_cast<std::size_t>(
            std::ceil(percentile_value * values.size()));

        return values[std::max<std::size_t>(
                          1, index) -
                      1];
    }

    void prepareQueries(
        std::vector<Query> &queries,
        const cortex::embedding::BGEEmbedder &embedder)
    {
        for (auto &query : queries)
        {
            // Match the production retrieval pipeline: transform once, then embed.
            const std::string transformed =
                cortex::retrieval::QueryTransformer::transform(query.text);

            query.embedding = embedder.embed(transformed);
        }
    }

    std::string expandQueryForExperiment(const std::string &transformed)
    {
        std::string expanded = transformed;

        const auto contains = [&](const std::string &term)
        {
            return transformed.find(term) != std::string::npos;
        };

        if (contains("lend") || contains("borrow") ||
            contains("loan"))
        {
            expanded += " available availability when receive return";
        }

        if (contains("when") || contains("what time") ||
            contains("what date"))
        {
            expanded += " date time schedule";
        }

        return expanded;
    }

    void computeExactGroundTruth(
        std::vector<Query> &queries,
        const std::vector<SourceRecord> &records,
        std::size_t dimension)
    {
        const auto &backend = cortex::vector::get_vector_backend();

        for (auto &query : queries)
        {
            std::vector<std::pair<float, std::string>> distances;
            distances.reserve(records.size());

            for (const auto &record : records)
            {
                if (record.values.size() != dimension)
                {
                    throw std::runtime_error(
                        "Stored vector dimension does not match index dimension.");
                }

                const float distance = backend.raw_l2_distance(
                    query.embedding.data(),
                    record.values.data(),
                    dimension);

                distances.emplace_back(
                    distance,
                    chunkIdFromMetadata(record.metadata, record.original_id));
            }

            std::sort(
                distances.begin(),
                distances.end(),
                [](const auto &a, const auto &b)
                {
                    if (a.first != b.first)
                    {
                        return a.first < b.first;
                    }
                    return a.second < b.second;
                });

            if (query.text == "When can Neha lend me the book?")
            {
                std::cout << "\n[EXACT RANK DIAGNOSTIC]\n";
                std::cout << "Query: " << query.text << '\n';

                std::size_t rank = 0;
                bool found = false;

                for (std::size_t i = 0; i < distances.size(); ++i)
                {
                    if (distances[i].second == "email-011_0")
                    {
                        rank = i + 1;
                        found = true;
                    }

                    if (i < 15 || distances[i].second == "email-011_0")
                    {
                        std::cout << "Rank " << (i + 1)
                                  << ": " << distances[i].second
                                  << ", distance=" << distances[i].first
                                  << '\n';
                    }
                }

                if (found)
                {
                    std::cout << "Expected chunk exact rank: " << rank << '\n';
                }
                else
                {
                    std::cout << "Expected chunk is missing from exact ranking.\n";
                }

                std::cout << "---------------------------\n";
            }

            query.exact_top_chunk_ids.clear();

            const std::size_t count = std::min(K, distances.size());

            for (std::size_t i = 0; i < count; ++i)
            {
                query.exact_top_chunk_ids.push_back(distances[i].second);
            }
        }
    }

    double calculateExactRecall(
        const std::vector<std::string> &approximate_chunk_ids,
        const std::vector<std::string> &exact_chunk_ids)
    {
        if (exact_chunk_ids.empty())
        {
            return 0.0;
        }

        const std::unordered_set<std::string> exact_set(
            exact_chunk_ids.begin(), exact_chunk_ids.end());

        std::size_t hits = 0;

        for (const auto &id : approximate_chunk_ids)
        {
            if (exact_set.contains(id))
            {
                ++hits;
            }
        }

        return static_cast<double>(hits) / exact_chunk_ids.size();
    }

    double calculateRelevantRecall(
        const std::vector<std::string> &retrieved_chunk_ids,
        const std::vector<std::string> &relevant_chunk_ids)
    {
        if (relevant_chunk_ids.empty())
        {
            return 0.0;
        }

        std::unordered_set<std::string> retrieved(
            retrieved_chunk_ids.begin(), retrieved_chunk_ids.end());

        std::size_t hits = 0;

        for (const auto &relevant_id : relevant_chunk_ids)
        {
            if (retrieved.contains(relevant_id))
            {
                ++hits;
            }
        }

        return static_cast<double>(hits) / relevant_chunk_ids.size();
    }

    Metrics runConfiguration(
        const Config &config,
        const std::vector<SourceRecord> &source_records,
        const std::vector<Query> &queries,
        std::size_t dimension)
    {
        Metrics metrics;

        const auto build_start = Clock::now();

        cortex::index::HNSWIndex index(
            dimension,
            config.M,
            config.ef_construction,
            config.ef_search,
            RANDOM_SEED);

        // New graph in memory. The persistent source index is never modified.
        std::unordered_map<std::size_t, std::size_t> rebuilt_to_original_id;
        std::unordered_map<std::size_t, std::string> rebuilt_to_chunk_id;

        rebuilt_to_original_id.reserve(source_records.size());
        rebuilt_to_chunk_id.reserve(source_records.size());

        for (const auto &record : source_records)
        {
            auto vector = toVector(record.values);

            const std::size_t rebuilt_id = index.insert(
                std::move(vector),
                record.metadata);

            rebuilt_to_original_id.emplace(rebuilt_id, record.original_id);
            rebuilt_to_chunk_id.emplace(
                rebuilt_id,
                chunkIdFromMetadata(record.metadata, record.original_id));
        }

        const auto build_end = Clock::now();

        metrics.build_ms =
            std::chrono::duration<double, std::milli>(
                build_end - build_start)
                .count();

        // This is the vector payload lower bound, not total process RSS.
        metrics.vector_payload_mb =
            static_cast<double>(index.size()) *
            static_cast<double>(dimension) *
            sizeof(float) / (1024.0 * 1024.0);

        // Warm up search without including it in the measured results.
        const std::size_t warmup_count = std::min<std::size_t>(10, queries.size());

        for (std::size_t i = 0; i < warmup_count; ++i)
        {
            const auto query_vector = toVector(queries[i].embedding);
            (void)index.search(query_vector, K, config.ef_search);
        }

        std::vector<double> latencies_us;
        latencies_us.reserve(queries.size());

        double exact_recall_sum = 0.0;
        double relevant_recall_sum = 0.0;

        const auto search_start = Clock::now();

        for (const auto &query : queries)
        {
            const auto query_vector = toVector(query.embedding);

            const auto query_start = Clock::now();
            const auto results = index.search(
                query_vector, K, config.ef_search);
            const auto query_end = Clock::now();

            latencies_us.push_back(
                std::chrono::duration<double, std::micro>(
                    query_end - query_start)
                    .count());

            std::vector<std::string> approximate_chunk_ids;
            std::vector<std::string> retrieved_chunk_ids;

            approximate_chunk_ids.reserve(results.size());
            retrieved_chunk_ids.reserve(results.size());

            // Extract the chunk IDs returned by HNSW.
            for (const auto &result : results)
            {
                const auto it = rebuilt_to_chunk_id.find(result.id);

                if (it == rebuilt_to_chunk_id.end())
                {
                    throw std::runtime_error(
                        "HNSW returned an ID missing from rebuilt_to_chunk_id: " +
                        std::to_string(result.id));
                }

                approximate_chunk_ids.push_back(it->second);
                retrieved_chunk_ids.push_back(it->second);
            }

            // Measure ANN recall against brute-force ground truth.
            const double query_exact_recall = calculateExactRecall(
                approximate_chunk_ids,
                query.exact_top_chunk_ids);

            exact_recall_sum += query_exact_recall;

            // Measure recall against the labelled relevant chunks.
            const double query_recall = calculateRelevantRecall(
                retrieved_chunk_ids,
                query.relevant_chunk_ids);

            relevant_recall_sum += query_recall;

            // Print the relevance miss only for the selected configuration.
            if (query_recall < 1.0 &&
                config.M == 16 &&
                config.ef_construction == 200)
            {

                std::cout << "\n[RELEVANCE MISS]\n";
                std::cout << "Query: " << query.text << '\n';

                std::cout << "Expected chunk(s): ";
                for (const auto &id : query.relevant_chunk_ids)
                {
                    std::cout << id << ' ';
                }

                std::cout << "\nRetrieved top " << K << ":\n";

                for (std::size_t i = 0; i < results.size(); ++i)
                {
                    std::cout << "  Rank " << (i + 1)
                              << ": " << retrieved_chunk_ids[i]
                              << ", distance=" << results[i].distance
                              << '\n';
                }
            }
        }

        const auto search_end = Clock::now();

        const double elapsed_seconds =
            std::chrono::duration<double>(
                search_end - search_start)
                .count();

        metrics.qps = elapsed_seconds > 0.0
                          ? static_cast<double>(queries.size()) / elapsed_seconds
                          : 0.0;

        metrics.p50_us = percentile(latencies_us, 0.50);
        metrics.p95_us = percentile(latencies_us, 0.95);
        metrics.p99_us = percentile(latencies_us, 0.99);

        metrics.exact_recall_at_10 =
            exact_recall_sum / queries.size();

        metrics.relevant_recall_at_10 =
            relevant_recall_sum / queries.size();

        return metrics;
    }

    struct LexicalDocument
    {
        std::string chunk_id;
        std::vector<std::string> tokens;
    };

    struct RankedChunk
    {
        std::string chunk_id;
        double score;
    };

    struct RetrievalMetrics
    {
        double recall_at_10 = 0.0;
        double mrr_at_10 = 0.0;
    };

    std::vector<std::string> tokenize(const std::string &text)
    {
        std::vector<std::string> tokens;
        std::string token;

        for (unsigned char ch : text)
        {
            if (std::isalnum(ch))
            {
                token.push_back(static_cast<char>(std::tolower(ch)));
            }
            else if (!token.empty())
            {
                tokens.push_back(std::move(token));
                token.clear();
            }
        }

        if (!token.empty())
        {
            tokens.push_back(std::move(token));
        }

        return tokens;
    }

    std::vector<LexicalDocument> buildLexicalDocuments(
        const std::vector<SourceRecord> &records)
    {
        std::vector<LexicalDocument> documents;
        documents.reserve(records.size());

        for (const auto &record : records)
        {
            const auto text_it = record.metadata.find("text");

            const std::string text =
                text_it == record.metadata.end() ? "" : text_it->second;

            documents.push_back({chunkIdFromMetadata(record.metadata, record.original_id),
                                 tokenize(text)});
        }

        return documents;
    }

    std::vector<RankedChunk> rankSemantic(
        const Query &query,
        const std::vector<SourceRecord> &records,
        std::size_t dimension)
    {
        const auto &backend = cortex::vector::get_vector_backend();

        std::vector<RankedChunk> ranked;
        ranked.reserve(records.size());

        for (const auto &record : records)
        {
            const float distance = backend.raw_l2_distance(
                query.embedding.data(),
                record.values.data(),
                dimension);

            ranked.push_back({chunkIdFromMetadata(record.metadata, record.original_id),
                              -static_cast<double>(distance)});
        }

        std::sort(
            ranked.begin(), ranked.end(),
            [](const auto &a, const auto &b)
            {
                if (a.score != b.score)
                {
                    return a.score > b.score;
                }
                return a.chunk_id < b.chunk_id;
            });

        return ranked;
    }

    std::vector<RankedChunk> rankBM25(
        const std::string &query_text,
        const std::vector<LexicalDocument> &documents)
    {
        constexpr double k1 = 1.5;
        constexpr double b = 0.75;

        const std::size_t N = documents.size();

        std::unordered_map<std::string, std::size_t> document_frequency;
        std::vector<std::unordered_map<std::string, std::size_t>> term_frequencies;
        std::vector<std::size_t> lengths;

        term_frequencies.reserve(N);
        lengths.reserve(N);

        std::size_t total_length = 0;

        for (const auto &document : documents)
        {
            std::unordered_map<std::string, std::size_t> frequencies;

            for (const auto &token : document.tokens)
            {
                ++frequencies[token];
            }

            for (const auto &[term, frequency] : frequencies)
            {
                (void)frequency;
                ++document_frequency[term];
            }

            lengths.push_back(document.tokens.size());
            total_length += document.tokens.size();
            term_frequencies.push_back(std::move(frequencies));
        }

        const double avgdl = N == 0
                                 ? 0.0
                                 : static_cast<double>(total_length) / N;

        const auto query_tokens = tokenize(query_text);
        const std::unordered_set<std::string> unique_query_terms(
            query_tokens.begin(), query_tokens.end());

        std::vector<RankedChunk> ranked;
        ranked.reserve(N);

        for (std::size_t i = 0; i < N; ++i)
        {
            double score = 0.0;

            for (const auto &term : unique_query_terms)
            {
                const auto df_it = document_frequency.find(term);
                const auto tf_it = term_frequencies[i].find(term);

                if (df_it == document_frequency.end() ||
                    tf_it == term_frequencies[i].end())
                {
                    continue;
                }

                const double df = static_cast<double>(df_it->second);
                const double idf = std::log(
                    1.0 + (static_cast<double>(N) - df + 0.5) / (df + 0.5));

                const double tf = static_cast<double>(tf_it->second);
                const double dl = static_cast<double>(lengths[i]);

                const double denominator =
                    tf + k1 * (1.0 - b + b * dl / (avgdl > 0.0 ? avgdl : 1.0));

                score += idf * (tf * (k1 + 1.0)) / denominator;
            }

            ranked.push_back({documents[i].chunk_id, score});
        }

        std::sort(
            ranked.begin(), ranked.end(),
            [](const auto &a, const auto &b)
            {
                if (a.score != b.score)
                {
                    return a.score > b.score;
                }
                return a.chunk_id < b.chunk_id;
            });

        return ranked;
    }

    std::vector<RankedChunk> rankHybridRRF(
        const std::vector<RankedChunk> &semantic,
        const std::vector<RankedChunk> &lexical)
    {
        constexpr double rrf_k = 60.0;

        std::unordered_map<std::string, double> fused_scores;

        for (std::size_t i = 0; i < semantic.size(); ++i)
        {
            fused_scores[semantic[i].chunk_id] +=
                1.0 / (rrf_k + static_cast<double>(i + 1));
        }

        for (std::size_t i = 0; i < lexical.size(); ++i)
        {
            fused_scores[lexical[i].chunk_id] +=
                1.0 / (rrf_k + static_cast<double>(i + 1));
        }

        std::vector<RankedChunk> ranked;
        ranked.reserve(fused_scores.size());

        for (const auto &[chunk_id, score] : fused_scores)
        {
            ranked.push_back({chunk_id, score});
        }

        std::sort(
            ranked.begin(), ranked.end(),
            [](const auto &a, const auto &b)
            {
                if (a.score != b.score)
                {
                    return a.score > b.score;
                }
                return a.chunk_id < b.chunk_id;
            });

        return ranked;
    }

    RetrievalMetrics evaluateRanking(
        const std::vector<RankedChunk> &ranking,
        const std::vector<std::string> &relevant_ids)
    {
        RetrievalMetrics metrics;

        if (relevant_ids.empty())
        {
            return metrics;
        }

        const std::unordered_set<std::string> relevant(
            relevant_ids.begin(), relevant_ids.end());

        std::size_t hits = 0;

        for (std::size_t i = 0;
             i < std::min<std::size_t>(K, ranking.size());
             ++i)
        {
            if (relevant.contains(ranking[i].chunk_id))
            {
                ++hits;

                if (metrics.mrr_at_10 == 0.0)
                {
                    metrics.mrr_at_10 = 1.0 / static_cast<double>(i + 1);
                }
            }
        }

        metrics.recall_at_10 =
            static_cast<double>(hits) / relevant.size();

        return metrics;
    }

    void evaluateRetrievalMethods(
        const std::vector<Query> &queries,
        const std::vector<SourceRecord> &records,
        std::size_t dimension)
    {
        const auto lexical_documents = buildLexicalDocuments(records);

        RetrievalMetrics semantic_total;
        RetrievalMetrics lexical_total;
        RetrievalMetrics hybrid_total;

        std::size_t hybrid_improved = 0;
        std::size_t hybrid_unchanged = 0;
        std::size_t hybrid_regressed = 0;

        for (const auto &query : queries)
        {
            const auto semantic = rankSemantic(query, records, dimension);

            const std::string transformed =
                cortex::retrieval::QueryTransformer::transform(query.text);

            const auto lexical = rankBM25(transformed, lexical_documents);
            const auto hybrid = rankHybridRRF(semantic, lexical);

            const auto semantic_metrics =
                evaluateRanking(semantic, query.relevant_chunk_ids);

            const auto lexical_metrics =
                evaluateRanking(lexical, query.relevant_chunk_ids);

            const auto hybrid_metrics =
                evaluateRanking(hybrid, query.relevant_chunk_ids);

            if (query.text == "When can Neha lend me the book?")
            {
                auto printRank = [](const std::string &name,
                                    const std::vector<RankedChunk> &ranking)
                {
                    std::size_t target_rank = 0;

                    for (std::size_t i = 0; i < ranking.size(); ++i)
                    {
                        if (ranking[i].chunk_id == "email-011_0")
                        {
                            target_rank = i + 1;
                            break;
                        }
                    }

                    std::cout << name << " target rank: "
                              << (target_rank ? std::to_string(target_rank) : "NOT FOUND")
                              << '\n';

                    std::cout << "Top 5:\n";
                    for (std::size_t i = 0;
                         i < std::min<std::size_t>(5, ranking.size()); ++i)
                    {
                        std::cout << "  " << (i + 1) << ". "
                                  << ranking[i].chunk_id
                                  << " score=" << ranking[i].score << '\n';
                    }
                };

                std::cout << "\n[RETRIEVAL METHOD DIAGNOSTIC]\n";
                printRank("Semantic", semantic);
                printRank("BM25", lexical);
                printRank("Hybrid RRF", hybrid);
            }

            semantic_total.recall_at_10 += semantic_metrics.recall_at_10;
            semantic_total.mrr_at_10 += semantic_metrics.mrr_at_10;

            lexical_total.recall_at_10 += lexical_metrics.recall_at_10;
            lexical_total.mrr_at_10 += lexical_metrics.mrr_at_10;

            hybrid_total.recall_at_10 += hybrid_metrics.recall_at_10;
            hybrid_total.mrr_at_10 += hybrid_metrics.mrr_at_10;

            if (hybrid_metrics.recall_at_10 >
                semantic_metrics.recall_at_10)
            {
                ++hybrid_improved;
            }
            else if (hybrid_metrics.recall_at_10 <
                     semantic_metrics.recall_at_10)
            {
                ++hybrid_regressed;
            }
            else
            {
                ++hybrid_unchanged;
            }
        }

        const double count = static_cast<double>(queries.size());

        std::cout << "\n========================================\n";
        std::cout << " Semantic vs Lexical vs Hybrid Retrieval\n";
        std::cout << "========================================\n";

        std::cout << std::left
                  << std::setw(18) << "Method"
                  << std::setw(18) << "Recall@10"
                  << std::setw(18) << "MRR@10"
                  << '\n';

        std::cout << std::string(54, '-') << '\n';
        std::cout << std::fixed << std::setprecision(4);

        auto print_metrics = [&](const std::string &name,
                                 const RetrievalMetrics &metrics)
        {
            std::cout << std::left
                      << std::setw(18) << name
                      << std::setw(18) << metrics.recall_at_10 / count
                      << std::setw(18) << metrics.mrr_at_10 / count
                      << '\n';
        };

        print_metrics("Semantic-only", semantic_total);
        print_metrics("BM25 lexical", lexical_total);
        print_metrics("Hybrid RRF", hybrid_total);

        std::cout << "\nHybrid vs semantic, per-query Recall@10:\n"
                  << "  Improved:  " << hybrid_improved << '\n'
                  << "  Unchanged: " << hybrid_unchanged << '\n'
                  << "  Regressed: " << hybrid_regressed << '\n';

        std::cout << "Queries evaluated: " << queries.size() << '\n';
    }

    void BM_HNSWRealDataConfigSweep(benchmark::State &state)
    {
        for (auto _ : state)
        {
            (void)_;

            try
            {
                const std::string index_path = envOr(
                    "CORTEX_INDEX_PATH",
                    "data/email_index_50.cortex");

                const std::string model_path = envOr(
                    "CORTEX_MODEL_PATH",
                    "models/bge-small-en-v1.5/onnx/model.onnx");

                const std::string vocab_path = envOr(
                    "CORTEX_VOCAB_PATH",
                    "models/bge-small-en-v1.5/vocab.txt");

                const std::string tsv_path = envOr(
                    "CORTEX_EVAL_TSV",
                    "tests/evaluation/data/retrieval_dataset_100.tsv");

                std::cout << "\n========================================\n";
                std::cout << " Cortex HNSW Real-Data Benchmark\n";
                std::cout << "========================================\n";
                std::cout << "Index:   " << index_path << '\n';
                std::cout << "Dataset: " << tsv_path << "\n\n";

                const auto source = cortex::persistence::Serializer::load(
                    index_path);

                const std::size_t dimension = source.dimension();

                const auto source_records = copySourceRecords(source);

                if (source_records.empty())
                {
                    throw std::runtime_error(
                        "The source index contains no vectors.");
                }

                std::cout << "Loaded vectors: " << source_records.size() << '\n';
                std::cout << "Dimension:      " << dimension << '\n';

                auto queries = loadQueries(tsv_path);

                std::cout << "Evaluation queries: " << queries.size() << '\n';
                std::cout << "Loading BGE embedder and embedding queries...\n";

                cortex::embedding::BGEEmbedder embedder(
                    model_path,
                    vocab_path);

                if (embedder.dimension() != dimension)
                {
                    throw std::runtime_error(
                        "BGE embedding dimension does not match index dimension.");
                }

                prepareQueries(queries, embedder);

                const std::vector<std::string> probe_queries = {
                    "When can Neha lend me the book?",
                    "When can Neha lend me her book?",
                    "When is Neha available to lend me her copy of Atomic Habits?"};

                std::cout << "\n[QUERY FORMULATION EXPERIMENT]\n";

                const auto &backend = cortex::vector::get_vector_backend();

                for (const auto &text : probe_queries)
                {
                    const std::string transformed =
                        cortex::retrieval::QueryTransformer::transform(text);

                    const std::string expanded =
                        expandQueryForExperiment(transformed);

                    const auto baseline_embedding = embedder.embed(transformed);
                    const auto expanded_embedding = embedder.embed(expanded);

                    auto rankTarget = [&](const std::vector<float> &embedding)
                    {
                        std::vector<std::pair<float, std::string>> distances;
                        distances.reserve(source_records.size());

                        for (const auto &record : source_records)
                        {
                            const float distance = backend.raw_l2_distance(
                                embedding.data(),
                                record.values.data(),
                                dimension);

                            distances.emplace_back(
                                distance,
                                chunkIdFromMetadata(
                                    record.metadata,
                                    record.original_id));
                        }

                        std::sort(
                            distances.begin(),
                            distances.end(),
                            [](const auto &a, const auto &b)
                            {
                                return a.first != b.first
                                           ? a.first < b.first
                                           : a.second < b.second;
                            });

                        std::size_t target_rank = 0;

                        for (std::size_t i = 0; i < distances.size(); ++i)
                        {
                            if (distances[i].second == "email-011_0")
                            {
                                target_rank = i + 1;
                                break;
                            }
                        }

                        return std::make_pair(target_rank, distances);
                    };

                    const auto [baseline_rank, baseline_distances] =
                        rankTarget(baseline_embedding);

                    const auto [expanded_rank, expanded_distances] =
                        rankTarget(expanded_embedding);

                    std::cout << "\nQuery: " << text
                              << "\nBaseline target rank: " << baseline_rank
                              << "\nExpanded target rank: " << expanded_rank
                              << "\nExpanded top 5:\n";

                    for (std::size_t i = 0;
                         i < std::min<std::size_t>(5, expanded_distances.size());
                         ++i)
                    {
                        std::cout << "  " << (i + 1)
                                  << ". " << expanded_distances[i].second
                                  << " distance=" << expanded_distances[i].first
                                  << '\n';
                    }
                }

                computeExactGroundTruth(queries, source_records, dimension);

                evaluateRetrievalMethods(
                    queries,
                    source_records,
                    dimension);

                const std::vector<Config> configs = {
                    {8, 100, 100},
                    {16, 100, 100},
                    {16, 200, 100},
                    {24, 200, 100},
                    {32, 200, 100}};

                std::cout << "\n"
                          << std::left
                          << std::setw(7) << "M"
                          << std::setw(10) << "efC"
                          << std::setw(10) << "efS"
                          << std::setw(12) << "ANN R@10"
                          << std::setw(14) << "Relevant R@10"
                          << std::setw(12) << "QPS"
                          << std::setw(12) << "p50 (us)"
                          << std::setw(12) << "p95 (us)"
                          << std::setw(12) << "p99 (us)"
                          << std::setw(13) << "Build (ms)"
                          << "Vector MB\n";

                std::cout << std::string(116, '-') << '\n';
                std::cout << std::fixed << std::setprecision(4);

                for (const auto &config : configs)
                {
                    const Metrics metrics = runConfiguration(
                        config,
                        source_records,
                        queries,
                        dimension);

                    std::cout
                        << std::left
                        << std::setw(7) << config.M
                        << std::setw(10) << config.ef_construction
                        << std::setw(10) << config.ef_search
                        << std::setw(12) << metrics.exact_recall_at_10
                        << std::setw(14) << metrics.relevant_recall_at_10
                        << std::setw(12) << metrics.qps
                        << std::setw(12) << metrics.p50_us
                        << std::setw(12) << metrics.p95_us
                        << std::setw(12) << metrics.p99_us
                        << std::setw(13) << metrics.build_ms
                        << metrics.vector_payload_mb
                        << '\n';
                }

                std::cout << "\nNotes:\n"
                          << "  ANN R@10 compares HNSW results against exact "
                             "brute-force L2 top-10.\n"
                          << "  Relevant R@10 measures retrieval of TSV-labelled "
                             "relevant chunks.\n"
                          << "  Vector MB estimates vector payload only; it is "
                             "not total RAM usage.\n"
                          << "  Query embedding time is excluded from search "
                             "latency and QPS.\n"
                          << "  The persistent index was loaded read-only; "
                             "test graphs were built in memory.\n\n";

                state.SetLabel("Cortex real email dataset HNSW sweep");
            }
            catch (const std::exception &ex)
            {
                state.SkipWithError(ex.what());
                return;
            }
        }
    }

    BENCHMARK(BM_HNSWRealDataConfigSweep)->Iterations(1);

} // namespace
