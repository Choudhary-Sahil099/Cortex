#include "index/hnsw.hpp"
#include "vector/backend.hpp"
#include "index/neighbour_selection.hpp"
#include <limits>
#include <stdexcept>
#include <algorithm>
#include<queue> // for priority que
#include <unordered_set>

namespace cortex::index {

    HNSWIndex::HNSWIndex(
        std::size_t dimension,
        std::size_t M,
        std::size_t ef_construction, // explore while building a graph
		std::size_t ef_search, // explore while quering a graph
        std::uint64_t seed
    )
        : dimension_(dimension),// check the dimension
        M_(M), // no of neighbours during construction
        ef_construction_(ef_construction), // this gives the number if neighbours to be considered  
        ef_search_(ef_search), // number dof neighbours to be considered durng searching
		max_level_(0), // the maximum level of the index
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

    // return the number of nodes in that particular index
    std::size_t HNSWIndex::size() const {
        return nodes_.size();
    }

    bool HNSWIndex::empty() const {
        return nodes_.empty();
    }
    

    // get the highest level of the index
    std::size_t HNSWIndex::max_level() const {
        return max_level_;
    }

	//check if the index has an entry point
    bool HNSWIndex::has_entry_point() const {
        return !nodes_.empty();
    }

	// get the node with the given id is not out of the range
    const HNSWNode& HNSWIndex::node(std::size_t id) const {
        if (id >= nodes_.size()) {
            throw std::out_of_range(
                "HNSW node ID out of range"
            );
        }

        return *nodes_[id];
    }

    // return the vector from the store
    const float* HNSWIndex::vector_data(std::size_t id) const {
        if (id >= vector_store_.size()) {
            throw std::out_of_range("Vector ID is not in the required range");
        }
        return vector_store_.vector_data(id);
    }
	
    std::size_t HNSWIndex::insert(
        const vector::Vector& vector
    ) {
        if (vector.dimension() != dimension_) {
            throw std::invalid_argument(
                "Vector dimension does not match HNSW index dimension"
            );
        }

        // vector store first
        vector_store_.add(vector);

        const std::size_t id =
            vector_store_.size() - 1;

        // Highest node level
        const std::size_t level =
            level_generator_.generate();

        auto node =
            std::make_unique<HNSWNode>(
                id,
                level
            );

        nodes_.push_back(
            std::move(node)
        );

        // case for first node
        if (nodes_.size() == 1) {
            entry_point_ = id;
            max_level_ = level;

            return id;
        }

        std::size_t current_entry =
            entry_point_;

         //decend through levels
        for (
            std::size_t current_level = max_level_;
            current_level > level;
            --current_level
            ) {
            current_entry =
                greedySearch(
                    vector,
                    current_entry,
                    current_level
                );
        }    
        //Search and connect at every level
         
        const std::size_t lowest_level =
            std::min(level, max_level_);

        //cleaner loop than prev
        for (
            std::size_t current_level = lowest_level;
            ;
            --current_level
            ) {
            const auto candidates =
                search_layer(
                    vector,
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

        // if the nodes reaches a new leevel it becomes its entry point
        if (level > max_level_) {
            entry_point_ = id;
            max_level_ = level;
        }

        return id;
    }

    void HNSWIndex::connect_nodes(std::size_t first, std::size_t second, std::size_t level) {
		if (first >= nodes_.size() || second >= nodes_.size()) {
			throw std::out_of_range(
				"node id is not in the required range"
			);
		}

        if (first == second) {
			throw std::invalid_argument(
				"Same node"
			);
        }
        HNSWNode& first_Node = *nodes_[first];
        HNSWNode& second_Node = *nodes_[second];

        if (level > first_Node.level() || level > second_Node.level()) {
            throw std::invalid_argument("Level excceds");
        }

        // Find the First and the second neighbour of the first and the second node
		auto& first_Neighbour = first_Node.neighbors(level);
        auto& second_Neighbour = second_Node.neighbors(level);


		// CHECK THE MAP IF THE NEIGHBOUR IS ALREADY PRESENT OR NOT IN BOTH THE FIRST AND SECOND NEIGHBOUR LISTS
		if (std::find(first_Neighbour.begin(), first_Neighbour.end(), second) == first_Neighbour.end()) {
			first_Neighbour.push_back(second);
		}
		if (std::find(second_Neighbour.begin(), second_Neighbour.end(), first) == second_Neighbour.end()) {
			second_Neighbour.push_back(first);
		}
    }

	// search for the nearest neighbors in a specific layer of the HNSW index
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
            vector::get_vector_backend();

        for (const std::size_t entry_point : entry_points) {

            if (entry_point >= nodes_.size()) {
                throw std::out_of_range(
                    "HNSW entry point out of range"
                );
            }

            const HNSWNode& node =
                *nodes_[entry_point];

            if (level > node.level()) {
                continue;
            }

            const float distance =
                backend.raw_l2_distance(
                    query.data(),
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
        
		//check while the que is not empty and the results size is less than ef
        while (!candidates.empty()) {

            const Result current = candidates.top();
            candidates.pop();

            if (results.size() >= ef) {
                const float worst_distance =
                    results.top().distance;

                if (current.distance > worst_distance) {
                    break;
                }
            }

            const HNSWNode& current_node =
                *nodes_[current.id];

            const auto& neighbors =
                current_node.neighbors(level);

            for (const std::size_t neighbor_id : neighbors) {

                if (visited.contains(neighbor_id)) {
                    continue;
                }

                visited.insert(neighbor_id);

                const float distance =
                    backend.raw_l2_distance(
                        query.data(),
                        vector_store_.vector_data(neighbor_id),
                        dimension_
                    );

                const Result result{
                    neighbor_id,
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


    //greedy search implementation to find the closest node in a level
    std::size_t HNSWIndex::greedySearch(const vector::Vector& query, std::size_t entry_point, std::size_t level) const {
        if (query.dimension() != dimension_) {
            throw std::invalid_argument("Does not match the Index dimension");
        }
        if (entry_point >= nodes_.size()) {
            throw std::out_of_range("Entry pint not in the range");
        }
        if (level > nodes_[entry_point]->level()) {
            throw std::invalid_argument(
                "Search level exceeds entry point level"
            );
        }

        const auto& backend =
            cortex::vector::get_vector_backend();

        std::size_t current = entry_point;

        float current_distance =
            backend.raw_l2_distance(
                query.data(),
                vector_store_.vector_data(current),
                dimension_
            );

        bool improved = true;

        while (improved) {
            improved = false;

            const auto& neighbors =
                nodes_[current]->neighbors(level);

            for (const std::size_t neighbor : neighbors) {

                const float distance =
                    backend.raw_l2_distance(
                        query.data(),
                        vector_store_.vector_data(neighbor),
                        dimension_
                    );

                if (distance < current_distance) {
                    current = neighbor;
                    current_distance = distance;
                    improved = true;
                }
            }
        }
        return current;
    }

	// connect the selected neighbors to the new node at a specific level
    void HNSWIndex::connectSelectedNeighbours(
        std::size_t node_id,
        const std::vector<HNSWSearchResult>& candidates,
        std::size_t level
    ) {

        const auto& backend = cortex::vector::get_vector_backend();

        const neighbourDistanceFunction distanceFunction =
            [this, &backend](std::size_t first, std::size_t second) {

            return backend.raw_l2_distance(
                this->vector_data(first),
                this->vector_data(second),
                this->dimension_);
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
        affected_nodes.reserve(selected.size());
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

        for (const std::size_t neighbour_id :
        affected_nodes) {

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

    // prune helper implementation --> second imp
    void HNSWIndex::pruneNeighbours(
        std::size_t node_id,
        std::size_t level
    ) {

        const auto& backend = cortex::vector::get_vector_backend();

        const neighbourDistanceFunction distanceFunction =
            [this, &backend](std::size_t first, std::size_t second) {

            return backend.raw_l2_distance(
                this->vector_data(first),
                this->vector_data(second),
                this->dimension_);
            };
        HNSWNode& node = *nodes_[node_id];

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

        candidates.reserve(neighbours.size());

        for (const std::size_t neighbor_id : neighbours) {

            const float distance =
                backend.raw_l2_distance(
                    node_data,
                    vector_store_.vector_data(neighbor_id),
                    dimension_
                );

            candidates.push_back({
                neighbor_id,
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

        selected_ids.reserve(selected.size());

        for (const auto& candidate : selected) {
            selected_ids.push_back(candidate.id);
        }

        std::vector<std::size_t> removed;

        for (const std::size_t neighbor_id : neighbours) {

            if (
                std::find(
                    selected_ids.begin(),
                    selected_ids.end(),
                    neighbor_id
                ) == selected_ids.end()
                ) {
                removed.push_back(neighbor_id);
            }
        }

        for (const std::size_t neighbor_id : removed) {
            disconnectNodes(
                node_id,
                neighbor_id,
                level
            );
        }
    }

    // node remove code
    void HNSWIndex::disconnectNodes(
        std::size_t first,
        std::size_t second,
        std::size_t level
    ) {
        if (first >= nodes_.size() ||
            second >= nodes_.size()) {
            throw std::out_of_range(
                "HNSW node ID out of range"
            );
        }

        auto& firstNeighbours =
            nodes_[first]->neighbors(level);

        firstNeighbours.erase(
            std::remove(
                firstNeighbours.begin(),
                firstNeighbours.end(),
                second
            ),
            firstNeighbours.end()
        );

        auto& secondNeighbours =
            nodes_[second]->neighbors(level);

        secondNeighbours.erase(
            std::remove(
                secondNeighbours.begin(),
                secondNeighbours.end(),
                first
            ),
            secondNeighbours.end()
        );
    }

    std::vector<HNSWSearchResult> HNSWIndex::search(
        const vector::Vector& query,
        std::size_t k,
        std::size_t ef_search
    ) const {
        if (query.dimension() != dimension_)
            throw std::invalid_argument(
                "Query dimension does not match index dimension"
            );

        if (k == 0)
            throw std::invalid_argument(
                "k must be greater than zero"
            );

        if (ef_search == 0)
            throw std::invalid_argument(
                "ef_search must be greater than zero"
            );

        if (nodes_.empty())
            return {};

        const std::size_t search_ef =
            std::max(ef_search, k);

        std::size_t current_entry = entry_point_;

        for (std::size_t level = max_level_;
            level > 0;
            --level) {

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
            std::min(k, candidates.size());

        return std::vector<HNSWSearchResult>(
            candidates.begin(),
            candidates.begin() + result_count
        );
    }

    std::vector<HNSWSearchResult> HNSWIndex::search(
        const vector::Vector& query,
        std::size_t k
    ) const {
        return search(query, k, ef_search_);
    }
}
