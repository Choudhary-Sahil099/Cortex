#pragma once

//hnsw includes
#include "index/hnsw_node.hpp"
#include "index/hnsw_level_generator.hpp"
#include "index/hnsw_search.hpp" // seach struct

//vector includes
#include "vector/vector.hpp"
#include "vector/vector_store.hpp"

// libraries includes
#include <cstddef>
#include <memory>
#include <vector>

namespace cortex::index {

    class HNSWIndex {
    public:
        HNSWIndex(
            std::size_t dimension,
            std::size_t M = 16,
            std::size_t ef_construction = 200,
            std::size_t ef_search = 50
        );

        std::size_t dimension() const;

        std::size_t M() const;

        std::size_t ef_construction() const;

        std::size_t ef_search() const;

        std::size_t size() const;

        bool empty() const;

        std::size_t max_level() const;

        bool has_entry_point() const;

        const HNSWNode& node(std::size_t id) const;
        const float* vector_data(std::size_t id) const; // access to the stored vector data

        std::size_t insert(const vector::Vector& vector);
		//parameter for search layer -> query, entry points, ef, level
        std::vector<HNSWSearchResult> search_layer(
            const vector::Vector& query,
            const std::vector<std::size_t>& entry_points,
            std::size_t ef,
            std::size_t level
        ) const;

    private:
        
        std::size_t dimension_;
        std::size_t M_;
        std::size_t ef_construction_;
        std::size_t ef_search_;
        std::size_t max_level_;
        std::size_t entry_point_;


        // adding the vector store 
        vector::VectorStore vector_store_;
        std::vector<std::unique_ptr<HNSWNode>> nodes_;
        HNSWLevelGenerator level_generator_; // level genrator

    };

}