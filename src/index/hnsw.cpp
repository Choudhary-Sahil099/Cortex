#include "index/hnsw.hpp"
#include "vector/backend.hpp"
#include "index/neighbour_selection.hpp"

#include <algorithm>
#include <limits>
#include <queue>
#include <stdexcept>
#include <unordered_set>
#include <utility>


// for the persistence state saftey the design includes states are -> next_id ,, entry->point ,, max_level 
namespace cortex::index {

    HNSWIndex::HNSWIndex(
        std::size_t dimension,
        std::size_t M,
        std::size_t ef_construction,
        std::size_t ef_search,
        std::uint64_t seed
    )
        : dimension_(dimension),
        M_(M),
        ef_construction_(ef_construction),
        ef_search_(ef_search),
        max_level_(0),
        entry_point_(std::numeric_limits<std::size_t>::max()),
        vector_store_(dimension),
        level_generator_(1.0, seed)
    {
        if (dimension == 0) {
            throw std::invalid_argument(
                "HNSW dimension must be greater than zero"
            );
        }

        if (M == 0) {
            throw std::invalid_argument(
                "HNSW M must be greater than zero"
            );
        }

        if (ef_construction == 0) {
            throw std::invalid_argument(
                "ef_construction must be greater than zero"
            );
        }

        if (ef_search == 0) {
            throw std::invalid_argument(
                "ef_search must be greater than zero"
            );
        }
    }

    std::size_t HNSWIndex::dimension() const {
        return dimension_;
    }

    std::size_t HNSWIndex::M() const {
        return M_;
    }

    std::size_t HNSWIndex::ef_construction() const {
        return ef_construction_;
    }

    std::size_t HNSWIndex::ef_search() const {
        return ef_search_;
    }

    std::size_t HNSWIndex::size() const {
        return nodes_.size();
    }

    bool HNSWIndex::empty() const {
        return nodes_.empty();
    }


    // states persistence

    //highest layer
    std::size_t HNSWIndex::max_level() const {
        return max_level_;
    }

    // where the search starts
    std::size_t HNSWIndex::entry_point() const
    {
        return entry_point_;
    }

    // prevents id reuse 
    std::size_t HNSWIndex::next_id() const
    {
        return vector_store_.next_id();
    }


    bool HNSWIndex::has_entry_point() const {
        return !nodes_.empty();
    }

    const HNSWNode& HNSWIndex::node(std::size_t id) const {
        const auto it = nodes_.find(id);

        if (it == nodes_.end()) {
            throw std::out_of_range(
                "HNSW node ID does not exist"
            );
        }

        return *it->second;
    }

    const float* HNSWIndex::vector_data(std::size_t id) const {
        return vector_store_.vector_data(id);
    }

    std::size_t HNSWIndex::insert(vector::Vector vector) {
        if (vector.dimension() != dimension_) {
            throw std::invalid_argument(
                "Vector dimension does not match HNSW index dimension"
            );
        }

        const std::size_t id =
            vector_store_.add(
                std::move(vector)
            );

        const float* inserted_data =
            vector_store_.vector_data(id);

        const std::size_t level =
            level_generator_.generate();

        insertNode(
            id,
            inserted_data,
            level
        );

        return id;
    }


    //insertNode 
    void HNSWIndex::insertNode(
        std::size_t id,
        const float* vector_data,
        std::size_t level
    ) {
        if (vector_data == nullptr) {
            throw std::invalid_argument(
                "Vector data must not be null"
            );
        }

        if (!vector_store_.contains(id)) {
            throw std::out_of_range(
                "VectorStore ID does not exist"
            );
        }

        if (nodes_.contains(id)) {
            throw std::invalid_argument(
                "HNSW node ID already exists"
            );
        }

        auto node =
            std::make_unique<HNSWNode>(
                id,
                level
            );

        nodes_.emplace(
            id,
            std::move(node)
        );

        if (nodes_.size() == 1) {
            entry_point_ = id;
            max_level_ = level;
            return;
        }

        std::size_t current_entry =
            entry_point_;

        for (
            std::size_t current_level = max_level_;
            current_level > level;
            --current_level
            ) {
            current_entry =
                greedySearch(
                    vector_data,
                    current_entry,
                    current_level
                );
        }

        const std::size_t lowest_level =
            std::min(level, max_level_);

        for (
            std::size_t current_level = lowest_level;
            ;
            --current_level
            ) {
            const auto candidates =
                search_layer(
                    vector_data,
                    { current_entry },
                    ef_construction_,
                    current_level
                );

            connectSelectedNeighbours(
                id,
                candidates,
                current_level
            );

            if (!candidates.empty()) {
                current_entry =
                    candidates.front().id;
            }

            if (current_level == 0) {
                break;
            }
        }

        if (level > max_level_) {
            entry_point_ = id;
            max_level_ = level;
        }
    }

    // remove implementation
    bool HNSWIndex::remove(std::size_t id) {
        const auto node_it = nodes_.find(id);

        if (node_it == nodes_.end()) {
            return false;
        }

        HNSWNode& node =
            *node_it->second;

        const std::size_t node_level =
            node.level();

        std::vector<std::pair<std::size_t, std::size_t>>
            edges_to_remove;

        for (
            std::size_t level = 0;
            level <= node_level;
            ++level
            ) {
            const auto& neighbours =
                node.neighbors(level);

            for (
                const std::size_t neighbour_id :
            neighbours
                ) {
                edges_to_remove.emplace_back(
                    neighbour_id,
                    level
                );
            }
        }

        for (
            const auto& [neighbour_id, level] :
            edges_to_remove
            ) {
            if (nodes_.contains(neighbour_id)) {
                disconnectNodes(
                    id,
                    neighbour_id,
                    level
                );
            }
        }
        nodes_.erase(node_it);

        if (id == entry_point_) {
            if (nodes_.empty()) {
                entry_point_ =
                    std::numeric_limits<std::size_t>::max();

                max_level_ = 0;
            }
            else {
                std::size_t new_entry_point =
                    std::numeric_limits<std::size_t>::max();

                std::size_t new_max_level = 0;

                for (const auto& [node_id, node_ptr] : nodes_) {
                    const std::size_t level =
                        node_ptr->level();

                    if (
                        new_entry_point ==
                        std::numeric_limits<std::size_t>::max()
                        ||
                        level > new_max_level
                        ) {
                        new_entry_point = node_id;
                        new_max_level = level;
                    }
                }

                entry_point_ = new_entry_point;
                max_level_ = new_max_level;
            }
        }
        const bool removed =
            vector_store_.remove(id);

        if (!removed) {
            throw std::logic_error(
                "HNSW and VectorStore are inconsistent"
            );
        }

        return true;
    }

    //updated implementation
    bool HNSWIndex::update(
        std::size_t id,
        vector::Vector vector,
        core::metaData metadata
    ) {
        if (!nodes_.contains(id)) {
            return false;
        }

        if (vector.dimension() != dimension_) {
            throw std::invalid_argument(
                "Vector dimension does not match HNSW index dimension"
            );
        }

        HNSWNode& old_node =
            *nodes_.at(id);

        const std::size_t old_level =
            old_node.level();

        // Remove the old graph.
        std::vector<std::pair<std::size_t, std::size_t>>
            edges_to_remove;

        for (
            std::size_t level = 0;
            level <= old_level;
            ++level
            ) {
            const auto& neighbours =
                old_node.neighbors(level);

            for (
                const std::size_t neighbour_id :
            neighbours
                ) {
                edges_to_remove.emplace_back(
                    neighbour_id,
                    level
                );
            }
        }

        for (
            const auto& [neighbour_id, level] :
            edges_to_remove
            ) {
            if (nodes_.contains(neighbour_id)) {
                disconnectNodes(
                    id,
                    neighbour_id,
                    level
                );
            }
        }

        nodes_.erase(id);
        const bool updated =
            vector_store_.update(
                id,
                std::move(vector),
                std::move(metadata)
            );

        if (!updated) {
            throw std::logic_error(
                "HNSW and VectorStore are inconsistent"
            );
        }

        const float* updated_data =
            vector_store_.vector_data(id);

        insertNode(
            id,
            updated_data,
            old_level
        );

        return true;
    }

    void HNSWIndex::connect_nodes(
        std::size_t first,
        std::size_t second,
        std::size_t level
    ) {
        const auto first_it =
            nodes_.find(first);

        const auto second_it =
            nodes_.find(second);

        if (
            first_it == nodes_.end() ||
            second_it == nodes_.end()
            ) {
            throw std::out_of_range(
                "HNSW node ID does not exist"
            );
        }

        if (first == second) {
            throw std::invalid_argument(
                "HNSW node cannot connect to itself"
            );
        }

        HNSWNode& first_node =
            *first_it->second;

        HNSWNode& second_node =
            *second_it->second;

        if (
            level > first_node.level() ||
            level > second_node.level()
            ) {
            throw std::invalid_argument(
                "Level exceeds node level"
            );
        }

        auto& first_neighbours =
            first_node.neighbors(level);

        auto& second_neighbours =
            second_node.neighbors(level);

        
        if (
            std::find(
                first_neighbours.begin(),
                first_neighbours.end(),
                second
            ) == first_neighbours.end()
            ) {
            first_neighbours.push_back(second);
        }

        if (
            std::find(
                second_neighbours.begin(),
                second_neighbours.end(),
                first
            ) == second_neighbours.end()
            ) {
            second_neighbours.push_back(first);
        }
    }

  
    std::vector<HNSWSearchResult> HNSWIndex::search_layer(
        const vector::Vector& query,
        const std::vector<std::size_t>& entry_points,
        std::size_t ef,
        std::size_t level
    ) const {
        if (query.dimension() != dimension_) {
            throw std::invalid_argument(
                "Query dimension does not match HNSW index dimension"
            );
        }

        return search_layer(
            query.data(),
            entry_points,
            ef,
            level
        );
    }

    
    std::vector<HNSWSearchResult> HNSWIndex::search_layer(
        const float* query_data,
        const std::vector<std::size_t>& entry_points,
        std::size_t ef,
        std::size_t level
    ) const {
        if (query_data == nullptr) {
            throw std::invalid_argument(
                "Query data must not be null"
            );
        }

        if (ef == 0) {
            throw std::invalid_argument(
                "ef must be greater than zero"
            );
        }

        if (level > max_level_) {
            throw std::invalid_argument(
                "Search level exceeds current maximum level"
            );
        }

        if (entry_points.empty()) {
            return {};
        }

        using Result = HNSWSearchResult;
        auto compare_min =
            [](const Result& a, const Result& b) {
            return a.distance > b.distance;
            };

        
        auto compare_max =
            [](const Result& a, const Result& b) {
            return a.distance < b.distance;
            };

        std::priority_queue<
            Result,
            std::vector<Result>,
            decltype(compare_min)
        > candidates(compare_min);

        std::priority_queue<
            Result,
            std::vector<Result>,
            decltype(compare_max)
        > results(compare_max);

        std::unordered_set<std::size_t> visited;

        const auto& backend =
            cortex::vector::get_vector_backend();

        for (const std::size_t entry_point : entry_points) {
            if (!nodes_.contains(entry_point)) {
                throw std::out_of_range(
                    "HNSW entry point does not exist"
                );
            }

            const HNSWNode& node =
                *nodes_.at(entry_point);

            if (level > node.level()) {
                continue;
            }

            const float distance =
                backend.raw_l2_distance(
                    query_data,
                    vector_store_.vector_data(entry_point),
                    dimension_
                );

            const Result result{
                entry_point,
                distance
            };

            candidates.push(result);
            results.push(result);

            visited.insert(entry_point);
        }

        while (!candidates.empty()) {
            const Result current =
                candidates.top();

            candidates.pop();

     
            if (results.size() >= ef) {
                const float worst_distance =
                    results.top().distance;

                if (current.distance > worst_distance) {
                    break;
                }
            }

            const HNSWNode& current_node =
                *nodes_.at(current.id);

            const auto& neighbours =
                current_node.neighbors(level);

            for (const std::size_t neighbour_id : neighbours) {
                if (visited.contains(neighbour_id)) {
                    continue;
                }

                visited.insert(neighbour_id);

                const float distance =
                    backend.raw_l2_distance(
                        query_data,
                        vector_store_.vector_data(neighbour_id),
                        dimension_
                    );

                const Result result{
                    neighbour_id,
                    distance
                };

                if (results.size() < ef) {
                    candidates.push(result);
                    results.push(result);
                }
                else if (
                    distance < results.top().distance
                    ) {
                    candidates.push(result);
                    results.push(result);
                    results.pop();
                }
            }
        }

        std::vector<Result> output;

        output.reserve(results.size());

        while (!results.empty()) {
            output.push_back(results.top());
            results.pop();
        }

        std::sort(
            output.begin(),
            output.end(),
            [](const Result& a, const Result& b) {
                return a.distance < b.distance;
            }
        );

        return output;
    }

    std::size_t HNSWIndex::greedySearch(
        const vector::Vector& query,
        std::size_t entry_point,
        std::size_t level
    ) const {
        if (query.dimension() != dimension_) {
            throw std::invalid_argument(
                "Query dimension does not match HNSW index dimension"
            );
        }

        return greedySearch(
            query.data(),
            entry_point,
            level
        );
    }

    std::size_t HNSWIndex::greedySearch(
        const float* query_data,
        std::size_t entry_point,
        std::size_t level
    ) const {
        if (query_data == nullptr) {
            throw std::invalid_argument(
                "Query data must not be null"
            );
        }

        if (!nodes_.contains(entry_point)) {
            throw std::out_of_range(
                "Entry point does not exist"
            );
        }

        if (level > nodes_.at(entry_point)->level()) {
            throw std::invalid_argument(
                "Search level exceeds entry point level"
            );
        }

        const auto& backend =
            cortex::vector::get_vector_backend();

        std::size_t current =
            entry_point;

        float current_distance =
            backend.raw_l2_distance(
                query_data,
                vector_store_.vector_data(current),
                dimension_
            );

        bool improved = true;

        while (improved) {
            improved = false;

            const auto& neighbours =
                nodes_.at(current)->neighbors(level);

            for (const std::size_t neighbour : neighbours) {
                const float distance =
                    backend.raw_l2_distance(
                        query_data,
                        vector_store_.vector_data(neighbour),
                        dimension_
                    );

                if (distance < current_distance) {
                    current = neighbour;
                    current_distance = distance;
                    improved = true;
                }
            }
        }

        return current;
    }

    void HNSWIndex::connectSelectedNeighbours(
        std::size_t node_id,
        const std::vector<HNSWSearchResult>& candidates,
        std::size_t level
    ) {
        const auto& backend =
            cortex::vector::get_vector_backend();

        const neighbourDistanceFunction distanceFunction =
            [this, &backend](
                std::size_t first,
                std::size_t second
                ) {
                    return backend.raw_l2_distance(
                        this->vector_data(first),
                        this->vector_data(second),
                        this->dimension_
                    );
            };

        const std::size_t max_neighbours =
            (level == 0)
            ? 2 * M_
            : M_;

        const auto selected =
            select_neighbors(
                candidates,
                max_neighbours,
                distanceFunction
            );

        std::vector<std::size_t> affected_nodes;

        affected_nodes.reserve(
            selected.size()
        );

        for (const auto& candidate : selected) {
            if (candidate.id == node_id) {
                continue;
            }

            connect_nodes(
                node_id,
                candidate.id,
                level
            );

            affected_nodes.push_back(
                candidate.id
            );
        }

        for (
            const std::size_t neighbour_id :
        affected_nodes
            ) {
            pruneNeighbours(
                neighbour_id,
                level
            );
        }
        pruneNeighbours(
            node_id,
            level
        );
    }

    void HNSWIndex::pruneNeighbours(
        std::size_t node_id,
        std::size_t level
    ) {
        if (!nodes_.contains(node_id)) {
            throw std::out_of_range(
                "HNSW node ID does not exist"
            );
        }

        const auto& backend =
            cortex::vector::get_vector_backend();

        const neighbourDistanceFunction distanceFunction =
            [this, &backend](
                std::size_t first,
                std::size_t second
                ) {
                    return backend.raw_l2_distance(
                        this->vector_data(first),
                        this->vector_data(second),
                        this->dimension_
                    );
            };

        HNSWNode& node =
            *nodes_.at(node_id);

        if (level > node.level()) {
            throw std::invalid_argument(
                "Level exceeds node level"
            );
        }

        auto& neighbours =
            node.neighbors(level);

        const std::size_t max_neighbours =
            (level == 0)
            ? 2 * M_
            : M_;

        if (neighbours.size() <= max_neighbours) {
            return;
        }

        const float* node_data =
            vector_store_.vector_data(node_id);

        std::vector<HNSWSearchResult> candidates;

        candidates.reserve(
            neighbours.size()
        );

        for (
            const std::size_t neighbour_id :
        neighbours
            ) {
            const float distance =
                backend.raw_l2_distance(
                    node_data,
                    vector_store_.vector_data(neighbour_id),
                    dimension_
                );

            candidates.push_back({
                neighbour_id,
                distance
                });
        }

        const auto selected =
            select_neighbors(
                candidates,
                max_neighbours,
                distanceFunction
            );

        std::vector<std::size_t> selected_ids;

        selected_ids.reserve(
            selected.size()
        );

        for (const auto& candidate : selected) {
            selected_ids.push_back(
                candidate.id
            );
        }

        std::vector<std::size_t> removed;

        for (
            const std::size_t neighbour_id :
        neighbours
            ) {
            if (
                std::find(
                    selected_ids.begin(),
                    selected_ids.end(),
                    neighbour_id
                ) == selected_ids.end()
                ) {
                removed.push_back(
                    neighbour_id
                );
            }
        }

        for (
            const std::size_t neighbour_id :
        removed
            ) {
            disconnectNodes(
                node_id,
                neighbour_id,
                level
            );
        }
    }

    void HNSWIndex::disconnectNodes(
        std::size_t first,
        std::size_t second,
        std::size_t level
    ) {
        if (
            !nodes_.contains(first) ||
            !nodes_.contains(second)
            ) {
            throw std::out_of_range(
                "HNSW node ID does not exist"
            );
        }

        HNSWNode& first_node =
            *nodes_.at(first);

        HNSWNode& second_node =
            *nodes_.at(second);

        if (
            level > first_node.level() ||
            level > second_node.level()
            ) {
            throw std::invalid_argument(
                "Level exceeds node level"
            );
        }

        auto& first_neighbours =
            first_node.neighbors(level);

        first_neighbours.erase(
            std::remove(
                first_neighbours.begin(),
                first_neighbours.end(),
                second
            ),
            first_neighbours.end()
        );

        auto& second_neighbours =
            second_node.neighbors(level);

        second_neighbours.erase(
            std::remove(
                second_neighbours.begin(),
                second_neighbours.end(),
                first
            ),
            second_neighbours.end()
        );
    }

    std::vector<HNSWSearchResult> HNSWIndex::search(
        const vector::Vector& query,
        std::size_t k,
        std::size_t ef_search
    ) const {
        if (query.dimension() != dimension_) {
            throw std::invalid_argument(
                "Query dimension does not match index dimension"
            );
        }

        if (k == 0) {
            throw std::invalid_argument(
                "k must be greater than zero"
            );
        }

        if (ef_search == 0) {
            throw std::invalid_argument(
                "ef_search must be greater than zero"
            );
        }

        if (nodes_.empty()) {
            return {};
        }

        const std::size_t search_ef =
            std::max(
                ef_search,
                k
            );

        std::size_t current_entry =
            entry_point_;

        for (
            std::size_t level = max_level_;
            level > 0;
            --level
            ) {
            current_entry =
                greedySearch(
                    query,
                    current_entry,
                    level
                );
        }
        const auto candidates =
            search_layer(
                query,
                { current_entry },
                search_ef,
                0
            );

        const std::size_t result_count =
            std::min(
                k,
                candidates.size()
            );

        return std::vector<HNSWSearchResult>(
            candidates.begin(),
            candidates.begin() + result_count
        );
    }

    std::vector<HNSWSearchResult> HNSWIndex::search(
        const vector::Vector& query,
        std::size_t k
    ) const {
        return search(
            query,
            k,
            ef_search_
        );
    }

     // read the existing nodes 
    const std::unordered_map<std::size_t,std::unique_ptr<HNSWNode>>& HNSWIndex::nodes() const
    {
        return nodes_;
    }


    //vector store 
    const core::VectorStore& HNSWIndex::vector_store() const
    {
        return vector_store_;
    }
}