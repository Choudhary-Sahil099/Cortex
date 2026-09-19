#include "index/hnsw.hpp"
#include "vector/backend.hpp"
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
		std::size_t ef_search // explore while quering a graph
    )
        : dimension_(dimension),// check the dimension
        M_(M), // no of neighbours during construction
        ef_construction_(ef_construction), // this gives the number if neighbours to be considered  
        ef_search_(ef_search), // number dof neighbours to be considered durng searching
		max_level_(0), // the maximum level of the index
        entry_point_(std::numeric_limits<std::size_t>::max()),
        vector_store_(dimension),
        level_generator_()
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

    std::size_t HNSWIndex::insert(const vector::Vector& vector) {
        // check the dimension of the vector
        if (vector.dimension() != dimension_) {
			throw::std::invalid_argument("dimensions not mathch");
        }
        vector_store_.add(vector);
        const std::size_t id = vector_store_.size() - 1; // we need the vector store size not the node size anymore 
		const std::size_t level = level_generator_.generate(); // the number of levels for a new node 
        auto node = std::make_unique<HNSWNode>(id, level);
		nodes_.push_back(std::move(node)); // add the nodde to the list of the nodes in the index

        if (nodes_.size() == 1) {
            entry_point_ = id;
            max_level_ = level;
        }
        return id; // return the id of the newly inserted node
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
}