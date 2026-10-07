#include "persistence/index_manager.hpp"

#include "persistence/serializer.hpp"

#include <filesystem>
#include <stdexcept>
#include <utility>

namespace cortex::persistence
{

    IndexManager::IndexManager(std::string path): path_(std::move(path)){}

    index::VectorIndex IndexManager::loadOrCreate(std::size_t dimension) const{
        if (dimension == 0)
        {
            throw std::invalid_argument(
                "Index dimension must be greater than zero"
            );
        }

        if (!std::filesystem::exists(path_))
        {
            return index::VectorIndex(dimension);
        }

        auto hnsw =
            Serializer::load(path_);

        if (hnsw.dimension() != dimension)
        {
            throw std::invalid_argument(
                "Persisted index dimension does not match requested dimension"
            );
        }

        return index::VectorIndex(
            std::move(hnsw)
        );
    }

    void IndexManager::save(const index::VectorIndex& index) const{
        Serializer::save(
            index.hnsw(),
            path_
        );
    }

}
