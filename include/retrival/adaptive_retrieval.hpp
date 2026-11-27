#pragma once

#include <algorithm>
#include <cstddef>
#include <regex>
#include <string>
#include <unordered_set>
#include <vector>

namespace cortex::retrieval {

// Experimental, conservative text-pattern router. No relevance labels are used.
enum class AdaptiveRoute { Hybrid, Identifier, Contrast, Compound, Temporal };

struct AdaptivePlan {
    AdaptiveRoute route;
    std::string first;
    std::string second;
};

inline const char* routeName(AdaptiveRoute route) {
    switch (route) {
        case AdaptiveRoute::Identifier: return "identifier";
        case AdaptiveRoute::Contrast: return "contrast";
        case AdaptiveRoute::Compound: return "compound";
        case AdaptiveRoute::Temporal: return "temporal";
        default: return "fallback";
    }
}

inline AdaptivePlan planAdaptiveRetrieval(const std::string& query) {
    using namespace std::regex_constants;
    static const std::regex compound(R"(^Find both the ([^.!?]+?) and the ([^.!?]+?) for ([^.!?]+)[.]$)", icase);
    static const std::regex contrast(R"(^(For [^.!?]+, find [^.!?]+?),\s+rather than\s+[^.!?]+[.]$)", icase);
    static const std::regex identifier(R"(\breference\s+[A-Z0-9]+(?:-[A-Z0-9]+)+\b)", icase);
    static const std::regex temporal(R"(^When did .+[?]$)", icase);
    std::smatch match;
    if (std::regex_match(query, match, compound)) {
        const auto scope = match[3].str();
        // Reject more complex coordination rather than silently dropping intents.
        static const std::regex extra(R"(\b(?:and|or|rather than|except)\b)", icase);
        if (!std::regex_search(match[1].str(), extra) &&
            !std::regex_search(match[2].str(), extra)) {
            return {AdaptiveRoute::Compound,
                    "Find the " + match[1].str() + " for " + scope + ".",
                    "Find the " + match[2].str() + " for " + scope + "."};
        }
    }
    if (std::regex_match(query, match, contrast))
        return {AdaptiveRoute::Contrast, match[1].str() + ".", {}};
    if (std::regex_search(query, identifier))
        return {AdaptiveRoute::Identifier, query, {}};
    if (std::regex_match(query, temporal))
        return {AdaptiveRoute::Temporal, query, {}};
    return {AdaptiveRoute::Hybrid, query, {}};
}

// Fixed first-list-first round robin, preserving original IDs and budget.
template<class Result>
std::vector<Result> mergeAdaptiveRankings(const std::vector<Result>& a,
                                         const std::vector<Result>& b,
                                         std::size_t k) {
    std::vector<Result> out;
    std::unordered_set<std::size_t> seen;
    for (std::size_t i = 0; i < std::max(a.size(), b.size()); ++i) {
        for (const auto* list : {&a, &b}) {
            if (out.size() == k) return out;
            if (i < list->size() && seen.insert((*list)[i].id).second)
                out.push_back((*list)[i]);
        }
    }
    return out;
}

} // namespace cortex::retrieval
