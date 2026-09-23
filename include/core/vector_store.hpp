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

        // dont confuse add and restore
        VectorId add(vector::Vector vector, metaData metadata = {}); // add func

        const VectorRecord& get(VectorId id) const; // get func
        bool remove(VectorId id); // remove func --> only the id required
           
        // check id exists or not
        bool contains(VectorId id) const;
        // updae
        bool update(VectorId id,vector::Vector vector,metaData metadata = {});

        //Hide layer
        const float* vector_data(VectorId id) const;
        //record APi
        const std::vector<VectorRecord>& records() const;
        VectorId next_id() const; //  store the nxt_id

        // restore -> insteadf of creating a new id we use the existiing / orignal one that is given by the persistence layer
        bool restore( VectorId id, vector::Vector vector, metaData metadata);
    private:
        std::size_t dimension_;
        VectorId next_id_;
        std::vector<VectorRecord> records_;
        std::unordered_map<VectorId, std::size_t> id_to_position_; // maping to the pos
    };

}