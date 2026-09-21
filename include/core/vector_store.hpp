#pragma once

#include "core/metadata.hpp"
#include "core/vector_record.hpp"

#include <cstddef>
#include <vector>
#include <unordered_map>


namespace cortex::core {

    class VectorStore {
    public:
        explicit VectorStore(std::size_t dimension);

        std::size_t dimension() const;
        std::size_t size() const;

        bool empty() const;
        VectorId add(vector::Vector vector, metaData metadata = {}); // add func
        const VectorRecord& get(VectorId id) const; // get func
        bool remove(VectorId id); // remove func --> only the id required
           
        // check id exists or not
        bool contains(VectorId id) const;
        // updae
        bool update(VectorId id,vector::Vector vector,metaData metadata = {});

        //Hide layer
        const float* vector_data(VectorId id) const;
    private:
        std::size_t dimension_;
        VectorId next_id_;
        std::vector<VectorRecord> records_;
        std::unordered_map<VectorId, std::size_t> id_to_position_; // maping to the pos
    };

}