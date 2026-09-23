#include "persistence/serializer.hpp"

#include <fstream>
#include <stdexcept>

namespace cortex::persistence {

    //binary file helper
    void Serializer::save(
        const index::HNSWIndex& index,
        const std::string& path
    ) {

        // header states
        std::ofstream file(
            path,
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

        const std::uint64_t ef_construction =
            static_cast<std::uint64_t>(
                index.ef_construction()
                );

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


        //index states

        // the id preservence
        const std::uint64_t next_id =
            static_cast<std::uint64_t>(
                index.next_id()
                );

        // entry state point
        const std::uint64_t entry_point =
            static_cast<std::uint64_t>(
                index.entry_point()
                );
        // max level
        const std::uint64_t max_level =
            static_cast<std::uint64_t>(
                index.max_level()
                );
      

        file.write(
            reinterpret_cast<const char*>(&next_id),
            sizeof(next_id)
        );

        file.write(
            reinterpret_cast<const char*>(&entry_point),
            sizeof(entry_point)
        );

        file.write(
            reinterpret_cast<const char*>(&max_level),
            sizeof(max_level)
        );
        // serialize the vector values
        const auto& records = index.vector_store().records();

        const std::uint64_t record_count =
            static_cast<std::uint64_t>(
                records.size()
                );

        // keepp this above the loop else error is going to occur 
        file.write(
            reinterpret_cast<const char*>(&record_count),
            sizeof(record_count)
        );


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

            const std::uint64_t node_id =
                static_cast<std::uint64_t>(
                    id
                    );

            const std::uint64_t level =
                static_cast<std::uint64_t>(
                    node_ptr->level()
                    );

            file.write(
                reinterpret_cast<const char*>(&node_id),
                sizeof(node_id)
            );

            file.write(
                reinterpret_cast<const char*>(&level),
                sizeof(level)
            );
            for (std::size_t current_level = 0;
                current_level <= node_ptr->level();
                ++current_level) {

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

                    const std::uint64_t stored_neighbour_id =
                        static_cast<std::uint64_t>(
                            neighbour_id
                            );

                    file.write(
                        reinterpret_cast<const char*>(
                            &stored_neighbour_id
                            ),
                        sizeof(stored_neighbour_id)
                    );
                }
            }
        }
        if (!file) {
            throw std::runtime_error(
                "Failed while writing Cortex file"
            );
        }


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

        if (!file) {
            throw std::runtime_error(
                "Failed to read Cortex configuration"
            );
        }

        if (dimension == 0) {
            throw std::runtime_error(
                "Invalid Cortex dimension"
            );
        }

        if (M == 0) {
            throw std::runtime_error(
                "Invalid Cortex M parameter"
            );
        }

        if (ef_construction == 0) {
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
        index::HNSWIndex index(
            static_cast<std::size_t>(dimension),
            static_cast<std::size_t>(M),
            static_cast<std::size_t>(ef_construction)
        );
    }

}