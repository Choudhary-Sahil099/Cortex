#include "core/vector_store.hpp"

#include <limits>
#include <stdexcept>
#include <utility>

namespace cortex::core {

    //vector_store
    VectorStore::VectorStore(std::size_t dimension)
        : dimension_(dimension),
          next_id_(0)
    {
        if (dimension == 0) {
            throw std::invalid_argument(
                "VectorStore dimension must be greater than zero"
            );
        }
    }

    std::size_t VectorStore::dimension() const
    {
        return dimension_;
    }

    std::size_t VectorStore::size() const
    {
        return records_.size();
    }

    bool VectorStore::empty() const
    {
        return records_.empty();
    }

    // adding a vector to the new_vector store
    VectorId VectorStore::add(
        vector::Vector vector,
        metaData metadata)
    {
        if (vector.dimension() != dimension_) {
            throw std::invalid_argument(
                "Vector dimension does not match VectorStore dimension"
            );
        }

        if (records_.size() >=
            static_cast<std::size_t>(
                std::numeric_limits<VectorId>::max())) {
            throw std::overflow_error(
                "VectorStore VectorId space exhausted"
            );
        }

        const VectorId id = next_id_;
        ++next_id_;

        const std::size_t position = records_.size();

        records_.push_back(
            VectorRecord{
                id,
                std::move(vector),
                std::move(metadata)
            }
        );
        id_to_position_[id] = position;
        return id;
    }


    // the get implementation
    const VectorRecord& VectorStore::get(VectorId id) const
    {
        const auto it = id_to_position_.find(id);

        if (it == id_to_position_.end()) {
            throw std::out_of_range(
                "VectorStore VectorId does not exist"
            );
        }

        return records_[it->second];
    }

    // remove implementation
    bool VectorStore::remove(VectorId id)
    {
        const auto it = id_to_position_.find(id);

        if (it == id_to_position_.end()) {
            return false;
        }

        const std::size_t position = it->second;

        const std::size_t last_position =
            records_.size() - 1;

        if (position != last_position) {
            records_[position] =
                std::move(records_[last_position]);

            const VectorId moved_id =
                records_[position].id;

            id_to_position_[moved_id] = position;
        }

        records_.pop_back();
        id_to_position_.erase(it);

        return true;
    }


    // contains implemenrataion
    bool VectorStore::contains(VectorId id) const
    {
        return id_to_position_.find(id) != id_to_position_.end(); // check present or not
    }

    //update implementation
    bool VectorStore::update(
        VectorId id,
        vector::Vector vector,
        metaData metadata)
    {
        if (vector.dimension() != dimension_) {
            throw std::invalid_argument(
                "Vector dimension does not match VectorStore dimension"
            );
        }

        const auto it = id_to_position_.find(id);

        if (it == id_to_position_.end()) {
            return false;
        }

        const std::size_t position = it->second;

        records_[position].vector = std::move(vector);
        records_[position].metadata = std::move(metadata);

        return true;
    }

    const float* VectorStore::vector_data(VectorId id) const
    {
        const auto it = id_to_position_.find(id);

        if (it == id_to_position_.end()) {
            throw std::out_of_range(
                "VectorStore VectorId does not exist"
            );
        }

        return records_[it->second].vector.data();
    }

    // records implementation
    const std::vector<VectorRecord>& VectorStore::records() const
    {
        return records_;
    }
    VectorId VectorStore::next_id() const
    {
        return next_id_;
    }

    // restore implementation

    bool VectorStore::restore(
        VectorId id,
        vector::Vector vector,
        metaData metadata
    )
    {
        if (vector.dimension() != dimension_) {
            throw std::invalid_argument(
                "Vector dimension does not match VectorStore dimension"
            );
        }

        if (id_to_position_.contains(id)) {
            throw std::invalid_argument(
                "VectorStore VectorId already exists"
            );
        }

        const std::size_t position =
            records_.size();

        records_.push_back(
            VectorRecord{
                id,
                std::move(vector),
                std::move(metadata)
            }
        );

        id_to_position_[id] = position;

        if (id >= next_id_) {
            next_id_ = id + 1;
        }

        return true;
    }

    void VectorStore::restore_next_id(VectorId next_id)
    {
        if (next_id < next_id_) {
            throw std::invalid_argument(
                "Persisted next_id is smaller than restored state"
            );
        }

        next_id_ = next_id;
    }
}