#include "persistence/serializer.hpp"

#include <fstream>
#include <stdexcept>
#include <vector>
#include <filesystem>
#include <random>
#include <limits>
#include <system_error>
#include <cmath>
#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#else
#include <cstdlib>
#endif

namespace {
namespace fs = std::filesystem;
struct StagingDirectory {
    fs::path path;
    explicit StagingDirectory(const fs::path& parent) {
#ifdef _WIN32
        std::random_device random;
        for (int i = 0; i < 32; ++i) {
            auto candidate = parent / (L".cortex-save-" + std::to_wstring(random()) + L"-" + std::to_wstring(random()));
            if (fs::create_directory(candidate)) { path = std::move(candidate); return; }
        }
        throw std::runtime_error("Cannot create staging directory");
#else
        auto pattern = (parent / ".cortex-save-XXXXXX").string();
        if (!::mkdtemp(pattern.data())) throw std::runtime_error("Cannot create staging directory");
        path = pattern;
#endif
    }
    ~StagingDirectory() { std::error_code ec; fs::remove_all(path, ec); }
    StagingDirectory(const StagingDirectory&) = delete;
    StagingDirectory& operator=(const StagingDirectory&) = delete;
};
void replace_file(const fs::path& source, const fs::path& destination) {
#ifdef _WIN32
    if (!MoveFileExW(source.c_str(), destination.c_str(), MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH))
        throw std::system_error(static_cast<int>(GetLastError()), std::system_category(), "Replace Cortex index");
#else
    fs::rename(source, destination);
#endif
}
}

namespace cortex::persistence {

    //binary file helper
    void Serializer::save(
        const index::HNSWIndex& index,
        const std::string& path
    ) {

        const auto destination = fs::absolute(fs::path(path));
        StagingDirectory staging(destination.parent_path());
        const auto temporary = staging.path / "index.tmp";
        std::ofstream file(
            temporary,
            std::ios::binary
        );

        if (!file) {
            throw std::runtime_error(
                "Failed to open file for writing"
            );
        }

        file.write(
            reinterpret_cast<const char*>(&MAGIC),
            sizeof(MAGIC)
        );

        file.write(
            reinterpret_cast<const char*>(&VERSION),
            sizeof(VERSION)
        );

        const std::uint64_t dimension =
            static_cast<std::uint64_t>(
                index.dimension()
                );

        const std::uint64_t M =
            static_cast<std::uint64_t>(
                index.M()
                );

        const std::uint64_t ef_construction =static_cast<std::uint64_t>(index.ef_construction());

        const std::uint64_t ef_search = static_cast<std::uint64_t>(index.ef_search());

        file.write(
            reinterpret_cast<const char*>(&dimension),
            sizeof(dimension)
        );

        file.write(
            reinterpret_cast<const char*>(&M),
            sizeof(M)
        );

        file.write(
            reinterpret_cast<const char*>(&ef_construction),
            sizeof(ef_construction)
        );


        //elf consttruction
        file.write(
            reinterpret_cast<const char*>(&ef_search),
            sizeof(ef_search)
        );

        //index states

        // the id preservence
        const std::uint64_t next_id = static_cast<std::uint64_t>(index.next_id());

        // entry state point
        const std::uint64_t entry_point =static_cast<std::uint64_t>(index.entry_point());
        // max level
        const std::uint64_t max_level =static_cast<std::uint64_t>(index.max_level());
      

        file.write(reinterpret_cast<const char*>(&next_id),sizeof(next_id));

        file.write(reinterpret_cast<const char*>(&entry_point),sizeof(entry_point));

        file.write(reinterpret_cast<const char*>(&max_level),sizeof(max_level));
        // serialize the vector values
        const auto& records = index.vector_store().records();

        const std::uint64_t record_count =static_cast<std::uint64_t>(records.size());

        // keepp this above the loop else error is going to occur 
        file.write(reinterpret_cast<const char*>(&record_count),sizeof(record_count) );
        for (const auto& record : records) {

            const std::uint64_t id =
                static_cast<std::uint64_t>(
                    record.id
                    );

            file.write(
                reinterpret_cast<const char*>(&id),
                sizeof(id)
            );

            const std::uint64_t dimension =
                static_cast<std::uint64_t>(
                    record.vector.dimension()
                    );

            file.write(
                reinterpret_cast<const char*>(&dimension),
                sizeof(dimension)
            );

            file.write(
                reinterpret_cast<const char*>(
                    record.vector.data()
                    ),
                static_cast<std::streamsize>(
                    dimension * sizeof(float)
                    )
            );

            const std::uint64_t metadata_count =
                static_cast<std::uint64_t>(
                    record.metadata.size()
                    );

            file.write(
                reinterpret_cast<const char*>(&metadata_count),
                sizeof(metadata_count)
            );

            for (const auto& [key, value] : record.metadata) {

                const std::uint64_t key_size =
                    static_cast<std::uint64_t>(
                        key.size()
                        );

                const std::uint64_t value_size =
                    static_cast<std::uint64_t>(
                        value.size()
                        );

                file.write(
                    reinterpret_cast<const char*>(&key_size),
                    sizeof(key_size)
                );

                file.write(
                    key.data(),
                    static_cast<std::streamsize>(key_size)
                );

                file.write(
                    reinterpret_cast<const char*>(&value_size),
                    sizeof(value_size)
                );

                file.write(
                    value.data(),
                    static_cast<std::streamsize>(value_size)
                );
            }
        }

        const auto& nodes =
            index.nodes();

        const std::uint64_t node_count =
            static_cast<std::uint64_t>(
                nodes.size()
                );

        file.write(
            reinterpret_cast<const char*>(&node_count),
            sizeof(node_count)
        );

        for (const auto& [id, node_ptr] : nodes) {

            const std::uint64_t node_id =static_cast<std::uint64_t>(id);

            const std::uint64_t level = static_cast<std::uint64_t>(node_ptr->level());

            file.write(
                reinterpret_cast<const char*>(&node_id),
                sizeof(node_id)
            );

            file.write(
                reinterpret_cast<const char*>(&level),
                sizeof(level)
            );
            for (std::size_t current_level = 0;current_level <= node_ptr->level();++current_level) {

                const auto& neighbours =
                    node_ptr->neighbors(current_level);

                const std::uint64_t neighbour_count =
                    static_cast<std::uint64_t>(
                        neighbours.size()
                        );

                file.write(
                    reinterpret_cast<const char*>(
                        &neighbour_count
                        ),
                    sizeof(neighbour_count)
                );

                for (const std::size_t neighbour_id : neighbours) {
                    const std::uint64_t stored_neighbour_id =static_cast<std::uint64_t>(neighbour_id);
                    file.write(reinterpret_cast<const char*>(&stored_neighbour_id),sizeof(stored_neighbour_id));
                }
            }
        }
        if (!file) {
            throw std::runtime_error(
                "Failed while writing Cortex file"
            );
        }
        file.flush();
        if (!file) throw std::runtime_error("Failed to flush Cortex index");
        file.close();
        if (!file) throw std::runtime_error("Failed to close Cortex index");
        replace_file(temporary, destination);
    }

    index::HNSWIndex Serializer::load(
        const std::string& path
    ) {
        std::ifstream file(
            path,
            std::ios::binary
        );

        if (!file) {
            throw std::runtime_error(
                "Failed to open Cortex file for reading"
            );
        }

        // Bound allocations by the actual file length, and cap configuration
        // parameters before constructing any graph objects.
        file.seekg(0, std::ios::end);
        const auto end = file.tellg();
        if (end < 0) throw std::runtime_error("Cannot determine Cortex file size");
        const auto file_size = static_cast<std::uint64_t>(end);
        file.seekg(0, std::ios::beg);
        auto require_bytes = [&](std::uint64_t count, std::uint64_t width) {
            const auto position = file.tellg();
            if (position < 0 || static_cast<std::uint64_t>(position) > file_size ||
                count > (file_size - static_cast<std::uint64_t>(position)) / width ||
                count > std::numeric_limits<std::size_t>::max() / width)
                throw std::runtime_error("Invalid or truncated Cortex payload");
        };
        std::uint64_t magic = 0;

        file.read(
            reinterpret_cast<char*>(&magic),
            sizeof(magic)
        );

        if (!file) {
            throw std::runtime_error(
                "Failed to read Cortex file header"
            );
        }

        if (magic != MAGIC) {
            throw std::runtime_error(
                "Invalid Cortex file magic"
            );
        }

        std::uint32_t version = 0;

        file.read(
            reinterpret_cast<char*>(&version),
            sizeof(version)
        );

        if (!file) {
            throw std::runtime_error(
                "Failed to read Cortex file version"
            );
        }

        if (version != VERSION) {
            throw std::runtime_error(
                "Unsupported Cortex file version"
            );
        }

        std::uint64_t dimension = 0;
        std::uint64_t M = 0;
        std::uint64_t ef_construction = 0;
        std::uint64_t ef_search = 0;

        file.read(
            reinterpret_cast<char*>(&dimension),
            sizeof(dimension)
        );

        file.read(
            reinterpret_cast<char*>(&M),
            sizeof(M)
        );

        file.read(
            reinterpret_cast<char*>(&ef_construction),
            sizeof(ef_construction)
        );

        file.read(
            reinterpret_cast<char*>(&ef_search),
            sizeof(ef_search)
        );
        if (!file) {
            throw std::runtime_error(
                "Failed to read Cortex configuration"
            );
        }

        if (dimension == 0 || dimension > 65536) {
            throw std::runtime_error(
                "Invalid Cortex dimension"
            );
        }

        if (M == 0 || M > 65536) {
            throw std::runtime_error(
                "Invalid Cortex M parameter"
            );
        }

        if (ef_construction == 0 || ef_construction > 10000000 || ef_search == 0 || ef_search > 10000000) {
            throw std::runtime_error(
                "Invalid Cortex ef_construction parameter"
            );
        }
        std::uint64_t next_id = 0;
        std::uint64_t entry_point = 0;
        std::uint64_t max_level = 0;

        file.read(
            reinterpret_cast<char*>(&next_id),
            sizeof(next_id)
        );

        file.read(
            reinterpret_cast<char*>(&entry_point),
            sizeof(entry_point)
        );

        file.read(
            reinterpret_cast<char*>(&max_level),
            sizeof(max_level)
        );

        if (!file) {
            throw std::runtime_error(
                "Failed to read Cortex index state"
            );
        }
        if (max_level > 64) throw std::runtime_error("Invalid Cortex max level");
        index::HNSWIndex index(
            static_cast<std::size_t>(dimension),
            static_cast<std::size_t>(M),
            static_cast<std::size_t>(ef_construction),
            static_cast<std::size_t>(ef_search)
        );
        std::uint64_t record_count = 0;

        file.read(
            reinterpret_cast<char*>(&record_count),
            sizeof(record_count)
        );
        if (!file) {
            throw std::runtime_error(
                "Failed to read Cortex record count"
            );
        }
        require_bytes(record_count, 24 + dimension * sizeof(float));
        for (std::uint64_t i = 0; i < record_count; ++i) {

            std::uint64_t id = 0;
            std::uint64_t record_dimension = 0;

            file.read(
                reinterpret_cast<char*>(&id),
                sizeof(id)
            );

            file.read(
                reinterpret_cast<char*>(&record_dimension),
                sizeof(record_dimension)
            );

            if (!file) {
                throw std::runtime_error(
                    "Failed to read Cortex vector record"
                );
            }

            if (record_dimension != dimension) {
                throw std::runtime_error(
                    "Vector dimension does not match Cortex index dimension"
                );
            }

            require_bytes(record_dimension, sizeof(float));
            cortex::vector::Vector vector(
                static_cast<std::size_t>(record_dimension)
            );

            file.read(
                reinterpret_cast<char*>(vector.data()),
                static_cast<std::streamsize>(
                    record_dimension * sizeof(float)
                    )
            );

            if (!file) {
                throw std::runtime_error(
                    "Failed to read Cortex vector data"
                );
            }
            for (std::size_t j = 0; j < vector.dimension(); ++j)
                if (!std::isfinite(vector[j])) throw std::runtime_error("Non-finite persisted vector");
            std::uint64_t metadata_count = 0;

            file.read(
                reinterpret_cast<char*>(&metadata_count),
                sizeof(metadata_count)
            );

            if (!file) {
                throw std::runtime_error(
                    "Failed to read Cortex metadata count"
                );
            }

            require_bytes(metadata_count, 16);
            cortex::core::metaData metadata;

            for (std::uint64_t j = 0; j < metadata_count; ++j) {
                std::uint64_t key_size = 0;
                std::uint64_t value_size = 0;

                file.read(
                    reinterpret_cast<char*>(&key_size),
                    sizeof(key_size)
                );

                if (!file) {
                    throw std::runtime_error(
                        "Failed to read Cortex metadata key size"
                    );
                }

                require_bytes(key_size, 1);
                std::string key(
                    static_cast<std::size_t>(key_size),
                    '\0'
                );

                file.read(
                    key.data(),
                    static_cast<std::streamsize>(key_size)
                );

                if (!file) {
                    throw std::runtime_error(
                        "Failed to read Cortex metadata key"
                    );
                }

                file.read(
                    reinterpret_cast<char*>(&value_size),
                    sizeof(value_size)
                );

                if (!file) {
                    throw std::runtime_error(
                        "Failed to read Cortex metadata value size"
                    );
                }

                require_bytes(value_size, 1);
                std::string value(
                    static_cast<std::size_t>(value_size),
                    '\0'
                );

                file.read(
                    value.data(),
                    static_cast<std::streamsize>(value_size)
                );

                if (!file) {
                    throw std::runtime_error(
                        "Failed to read Cortex metadata value"
                    );
                }

                metadata.emplace(
                    std::move(key),
                    std::move(value)
                );
            }
            if (id >= next_id || !index.restore_vector(
                static_cast<cortex::core::VectorId>(id),
                std::move(vector),
                std::move(metadata)
            )) throw std::runtime_error("Duplicate or invalid Cortex vector ID");
        }

        std::uint64_t node_count = 0;

        file.read(
            reinterpret_cast<char*>(&node_count),
            sizeof(node_count)
        );
        if (!file) {
            throw std::runtime_error(
                "Failed to read Cortex record count"
            );
        }
        if (node_count != record_count) throw std::runtime_error("Cortex node/record count mismatch");
        require_bytes(node_count, 24);
        // updated version
        struct PendingEdge {
            std::size_t first;
            std::size_t second;
            std::size_t level;
        };

        std::vector<PendingEdge> pending_edges;

        for (std::uint64_t i = 0; i < node_count; ++i) {

            std::uint64_t node_id = 0;
            std::uint64_t level = 0;

            file.read(
                reinterpret_cast<char*>(&node_id),
                sizeof(node_id)
            );

            file.read(
                reinterpret_cast<char*>(&level),
                sizeof(level)
            );

            if (!file) {
                throw std::runtime_error(
                    "Failed to read Cortex HNSW node"
                );
            }

            if (level > max_level) {
                throw std::runtime_error(
                    "HNSW node level exceeds persisted max level"
                );
            }

            index.restore_node(
                static_cast<std::size_t>(node_id),
                static_cast<std::size_t>(level)
            );

            for (
                std::size_t current_level = 0;
                current_level <= static_cast<std::size_t>(level);
                ++current_level
                ) {
                std::uint64_t neighbour_count = 0;

                file.read(
                    reinterpret_cast<char*>(&neighbour_count),
                    sizeof(neighbour_count)
                );

                if (!file) {
                    throw std::runtime_error(
                        "Failed to read Cortex neighbour count"
                    );
                }

                require_bytes(neighbour_count, 8);
                if (neighbour_count > node_count) throw std::runtime_error("Invalid neighbour count");
                for (
                    std::uint64_t j = 0;
                    j < neighbour_count;
                    ++j
                    ) {
                    std::uint64_t neighbour_id = 0;

                    file.read(
                        reinterpret_cast<char*>(&neighbour_id),
                        sizeof(neighbour_id)
                    );

                    if (!file) {
                        throw std::runtime_error(
                            "Failed to read Cortex neighbour ID"
                        );
                    }

                    pending_edges.push_back({
                        static_cast<std::size_t>(node_id),
                        static_cast<std::size_t>(neighbour_id),
                        current_level
                        });
                }
            }
        }

        for (const auto& edge : pending_edges) {
            index.restore_edge(
                edge.first,
                edge.second,
                edge.level
            );
        }

        index.restore_next_id(
            static_cast<cortex::core::VectorId>(next_id)
        );

        index.restore_state(
            static_cast<std::size_t>(entry_point),
            static_cast<std::size_t>(max_level)
        );

        if (!file) {
            throw std::runtime_error(
                "Failed to read Cortex index state"
            );
        }
        if (file.peek() != std::char_traits<char>::eof()) throw std::runtime_error("Unexpected trailing Cortex data");
        return index;
    }

}