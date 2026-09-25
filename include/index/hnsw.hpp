#pragma once

//hnsw includes
#include "index/hnsw_node.hpp"
#include "index/hnsw_level_generator.hpp"
#include "index/hnsw_search.hpp" // seach struct

//vector includes
#include "vector/vector.hpp"
//#include "vector/vector_store.hpp"
#include "core/vector_store.hpp"

// libraries includes
#include <cstddef>
#include <memory>
#include <vector>
#include <cstdint>
#include <random>
#include <unordered_map> //--> for unordered_map
namespace cortex::index {

    class HNSWIndex {
    public:
        HNSWIndex(
            std::size_t dimension,
            std::size_t M = 16,
            std::size_t ef_construction = 200,
            std::size_t ef_search = 50,
            std::uint64_t seed = std::random_device{}()
        );

        std::size_t dimension() const;

        std::size_t M() const;

        std::size_t ef_construction() const;

        std::size_t ef_search() const;

        std::size_t size() const;

        bool empty() const;
        

        //persistence states 
        std::size_t max_level() const;
        std::size_t entry_point() const;
        std::size_t next_id() const;

        bool has_entry_point() const;

        const HNSWNode& node(std::size_t id) const;
        const float* vector_data(std::size_t id) const; // access to the stored vector data

        std::size_t insert(vector::Vector vector);
        bool remove(std::size_t id); //remove a node
        bool update(
            std::size_t id,
            vector::Vector vector,
            core::metaData metadata = {}
        ); //--> update a node
		//parameter for search layer -> query, entry points, ef, level
        
        //Connect two nodes in the graph at a specific level
        void connect_nodes(std::size_t first, std::size_t second, std::size_t level);
        std::vector<HNSWSearchResult> search_layer(
            const vector::Vector& query,
            const std::vector<std::size_t>& entry_points,
            std::size_t ef,
            std::size_t level
        ) const;

		// search a layer but returns only the closest node id
        std::size_t greedySearch(const vector::Vector& query, std::size_t entry_point, std::size_t level) const;
        //search
        std::vector<HNSWSearchResult> search(const vector::Vector& query,std::size_t k ) const;
        std::vector<HNSWSearchResult> search(const vector::Vector& query, std::size_t k, std::size_t ef_search) const;
        //nodes read only
        const std::unordered_map< std::size_t, std::unique_ptr<HNSWNode>>& nodes() const;
        const core::VectorStore& vector_store() const; // keep it read only else the peristance can change the index
        bool restore_vector(core::VectorId id, vector::Vector vector,core::metaData metadata = {}); //reconstuct the vector

        bool restore_node(std::size_t id,std::size_t level); // restore the node to create the HNSW nodes
        bool restore_edge(std::size_t first, std::size_t second, std::size_t level); // restore edges -> dont use again restore Connected nodes i have used the existing connect node func

        void restore_state(std::size_t entry_point,std::size_t max_level);
        void restore_next_id(core::VectorId next_id);
    private:
        
        std::size_t dimension_;
        std::size_t M_;
        std::size_t ef_construction_;
        std::size_t ef_search_;
        std::size_t max_level_;
        std::size_t entry_point_;


        // adding the vector store 
        core::VectorStore vector_store_;
        //std::vector<std::unique_ptr<HNSWNode>> nodes_;  //--> not compatible for the delete operation 


        std::unordered_map<std::size_t, std::unique_ptr<HNSWNode>> nodes_; // --> updated node_
        HNSWLevelGenerator level_generator_; // level genrator
        
        //
        void connectSelectedNeighbours(std::size_t node_id, const std::vector<HNSWSearchResult>& candidates, std::size_t level);
        // prune helper
        void pruneNeighbours(std::size_t node_id, std::size_t level);

        // node remover
        void disconnectNodes(std::size_t first, std::size_t second, std::size_t level);

        void insertNode(std::size_t id,
            const float* vector_data,
            std::size_t level
        );
        //internal overloader
        std::size_t greedySearch(
            const float* query_data,
            std::size_t entry_point,
            std::size_t level
        ) const;

        std::vector<HNSWSearchResult> search_layer(
            const float* query_data,
            const std::vector<std::size_t>& entry_points,
            std::size_t ef,
            std::size_t level
        ) const;


    };
    
}