#include "core/vector_store.hpp"
#include "persistence/serializer.hpp"


#include <cstdio>
#include <gtest/gtest.h>

#include <algorithm>
#include <string>
#include <cstdint>
#include <stdexcept>
#include <utility>
#include <fstream> // for files 
namespace {

    cortex::vector::Vector make_vector(
        std::initializer_list<float> values)
    {
        cortex::vector::Vector vector(values.size());

        std::copy(
            values.begin(),
            values.end(),
            vector.data()
        );

        return vector;
    }




    TEST(VectorStoreTest, RestoresExplicitId)
    {
        cortex::core::VectorStore store(4);

        const cortex::core::VectorId id = 42;

        auto vector = make_vector({
            1.0f,
            2.0f,
            3.0f,
            4.0f
            });

        cortex::core::metaData metadata{
            {"source", "test"}
        };

        EXPECT_TRUE(
            store.restore(
                id,
                std::move(vector),
                metadata
            )
        );

        EXPECT_TRUE(
            store.contains(id)
        );

        const auto& record =
            store.get(id);

        EXPECT_EQ(
            record.id,
            id
        );

        EXPECT_FLOAT_EQ(
            record.vector[0],
            1.0f
        );

        EXPECT_EQ(
            record.metadata.at("source"),
            "test"
        );
    }

    TEST(VectorStoreTest, RestoreRejectsDuplicateId)
    {
        cortex::core::VectorStore store(4);

        store.add(
            make_vector({
                1.0f,
                2.0f,
                3.0f,
                4.0f
                })
        );

        EXPECT_THROW(
            store.restore(
                0,
                make_vector({
                    5.0f,
                    6.0f,
                    7.0f,
                    8.0f
                    }),
                {}
            ),
            std::invalid_argument
        );
    }

    TEST(VectorStoreTest, RestoreRejectsWrongDimension)
    {
        cortex::core::VectorStore store(4);

        auto wrong_vector =
            make_vector({
                1.0f,
                2.0f,
                3.0f
                });

        EXPECT_THROW(
            store.restore(
                10,
                std::move(wrong_vector),
                {}
            ),
            std::invalid_argument
        );
    }

    TEST(VectorStoreTest, RestoreAdvancesNextId)
    {
        cortex::core::VectorStore store(4);

        store.restore(
            10,
            make_vector({
                1.0f,
                2.0f,
                3.0f,
                4.0f
                }),
            {}
        );

        EXPECT_EQ(
            store.next_id(),
            11
        );
    }

    TEST(VectorStoreTest, AddAfterRestoreUsesNextId)
    {
        cortex::core::VectorStore store(4);

        store.restore(
            10,
            make_vector({
                1.0f,
                2.0f,
                3.0f,
                4.0f
                }),
            {}
        );

        const cortex::core::VectorId id =
            store.add(
                make_vector({
                    5.0f,
                    6.0f,
                    7.0f,
                    8.0f
                    })
            );

        EXPECT_EQ(id, 11);
    }


    TEST(SerializerTest, SaveAndLoadRoundTrip)
    {
        cortex::index::HNSWIndex index(4, 4, 50, 20, 42);

        auto v1 = make_vector({ 1.0f, 2.0f, 3.0f, 4.0f });
        auto v2 = make_vector({ 5.0f, 6.0f, 7.0f, 8.0f });
        auto v3 = make_vector({ 9.0f, 10.0f, 11.0f, 12.0f });

        cortex::core::metaData metadata1{
            {"email", "first"},
            {"sender", "alice"}
        };

        cortex::core::metaData metadata2{
            {"email", "second"},
            {"sender", "bob"}
        };

        const auto id1 = index.insert(std::move(v1));
        const auto id2 = index.insert(std::move(v2));
        const auto id3 = index.insert(std::move(v3));

        index.update(id1, make_vector({ 10.0f, 20.0f, 30.0f, 40.0f }), metadata1);
        index.update(id2, make_vector({ 50.0f, 60.0f, 70.0f, 80.0f }), metadata2);

        const std::string path = "test_roundtrip.cortex";

        cortex::persistence::Serializer::save(index, path);

        auto loaded =
            cortex::persistence::Serializer::load(path);

        EXPECT_EQ(loaded.dimension(), index.dimension());
        EXPECT_EQ(loaded.M(), index.M());
        EXPECT_EQ(
            loaded.ef_construction(),
            index.ef_construction()
        );

        EXPECT_EQ(loaded.size(), index.size());
        EXPECT_EQ(loaded.next_id(), index.next_id());
        EXPECT_EQ(loaded.entry_point(), index.entry_point());
        EXPECT_EQ(loaded.max_level(), index.max_level());

        EXPECT_EQ(
            loaded.vector_store().records().size(),
            index.vector_store().records().size()
        );

        for (const auto& record : index.vector_store().records()) {
            const auto& restored =
                loaded.vector_store().get(record.id);

            EXPECT_EQ(
                restored.vector.dimension(),
                record.vector.dimension()
            );

            for (std::size_t i = 0; i < record.vector.dimension(); ++i) {
                EXPECT_FLOAT_EQ(
                    restored.vector[i],
                    record.vector[i]
                );
            }

            EXPECT_EQ(restored.metadata, record.metadata);
        }

        std::remove(path.c_str());
    }

    // rejects if not a cortex file

    TEST(SerializerTest, RejectsInvalidMagic)
    {
        const std::string path = "invalid_magic.cortex";

        {
            std::ofstream file(path, std::ios::binary);

            std::uint64_t invalid_magic = 12345;

            file.write(
                reinterpret_cast<const char*>(&invalid_magic),
                sizeof(invalid_magic)
            );
        }

        EXPECT_THROW(
            cortex::persistence::Serializer::load(path),
            std::runtime_error
        );

        std::remove(path.c_str());
    }
    // not the version  / invalid version
    TEST(SerializerTest, RejectsUnsupportedVersion)
    {
        const std::string path = "invalid_version.cortex";

        {
            std::ofstream file(path, std::ios::binary);

            const std::uint64_t magic =
                0x434F525445585F31;

            const std::uint32_t invalid_version = 999;

            file.write(
                reinterpret_cast<const char*>(&magic),
                sizeof(magic)
            );

            file.write(
                reinterpret_cast<const char*>(&invalid_version),
                sizeof(invalid_version)
            );
        }

        EXPECT_THROW(
            cortex::persistence::Serializer::load(path),
            std::runtime_error
        );

        std::remove(path.c_str());
    }

    // corrupt cases --> 
    TEST(SerializerTest, RejectsTruncatedFile)
    {
        const std::string path = "truncated.cortex";

        {
            std::ofstream file(path, std::ios::binary);

            const std::uint64_t magic =
                0x434F525445585F31;

            const std::uint32_t version = 1;

            file.write(
                reinterpret_cast<const char*>(&magic),
                sizeof(magic)
            );

            file.write(
                reinterpret_cast<const char*>(&version),
                sizeof(version)
            );

            // File intentionally ends here.
        }

        EXPECT_THROW(
            cortex::persistence::Serializer::load(path),
            std::runtime_error
        );

        std::remove(path.c_str());
    }

    TEST(SerializerTest, RejectsInvalidVectorDimension)
    {
        const std::string path = "invalid_dimension.cortex";

        {
            std::ofstream file(path, std::ios::binary);

            const std::uint64_t magic =
                0x434F525445585F31;

            const std::uint32_t version = 1;

            const std::uint64_t dimension = 4;
            const std::uint64_t M = 4;
            const std::uint64_t ef_construction = 50;

            const std::uint64_t next_id = 1;
            const std::uint64_t entry_point = 0;
            const std::uint64_t max_level = 0;

            const std::uint64_t record_count = 1;

            const std::uint64_t id = 0;
            const std::uint64_t invalid_dimension = 3;

            file.write(
                reinterpret_cast<const char*>(&magic),
                sizeof(magic)
            );

            file.write(
                reinterpret_cast<const char*>(&version),
                sizeof(version)
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

            file.write(
                reinterpret_cast<const char*>(&record_count),
                sizeof(record_count)
            );

            file.write(
                reinterpret_cast<const char*>(&id),
                sizeof(id)
            );

            file.write(
                reinterpret_cast<const char*>(&invalid_dimension),
                sizeof(invalid_dimension)
            );
        }

        EXPECT_THROW(
            cortex::persistence::Serializer::load(path),
            std::runtime_error
        );

        std::remove(path.c_str());
    }
    //invlid level
    TEST(SerializerTest, RejectsInvalidNodeLevel)
    {
        const std::string path = "invalid_node_level.cortex";

        {
            std::ofstream file(path, std::ios::binary);

            const std::uint64_t magic =
                0x434F525445585F31;

            const std::uint32_t version = 1;

            const std::uint64_t dimension = 4;
            const std::uint64_t M = 4;
            const std::uint64_t ef_construction = 50;

            const std::uint64_t next_id = 1;
            const std::uint64_t entry_point = 0;
            const std::uint64_t max_level = 0;

            const std::uint64_t record_count = 1;

            const std::uint64_t id = 0;
            const std::uint64_t record_dimension = 4;

            const float values[] = {
                1.0f, 2.0f, 3.0f, 4.0f
            };

            const std::uint64_t metadata_count = 0;

            const std::uint64_t node_count = 1;
            const std::uint64_t node_id = 0;

            // Invalid: node level = 1, but max_level = 0.
            const std::uint64_t invalid_level = 1;

            file.write(
                reinterpret_cast<const char*>(&magic),
                sizeof(magic)
            );

            file.write(
                reinterpret_cast<const char*>(&version),
                sizeof(version)
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

            file.write(
                reinterpret_cast<const char*>(&record_count),
                sizeof(record_count)
            );

            file.write(
                reinterpret_cast<const char*>(&id),
                sizeof(id)
            );

            file.write(
                reinterpret_cast<const char*>(&record_dimension),
                sizeof(record_dimension)
            );

            file.write(
                reinterpret_cast<const char*>(values),
                sizeof(values)
            );

            file.write(
                reinterpret_cast<const char*>(&metadata_count),
                sizeof(metadata_count)
            );

            file.write(
                reinterpret_cast<const char*>(&node_count),
                sizeof(node_count)
            );

            file.write(
                reinterpret_cast<const char*>(&node_id),
                sizeof(node_id)
            );

            file.write(
                reinterpret_cast<const char*>(&invalid_level),
                sizeof(invalid_level)
            );
        }

        EXPECT_THROW(
            cortex::persistence::Serializer::load(path),
            std::runtime_error
        );

        std::remove(path.c_str());
    }

    // invalid neighbours
    TEST(SerializerTest, RejectsInvalidNeighbour)
    {
        const std::string path = "invalid_neighbour.cortex";

        {
            std::ofstream file(path, std::ios::binary);

            const std::uint64_t magic = 0x434F525445585F31;
            const std::uint32_t version = 1;

            const std::uint64_t dimension = 4;
            const std::uint64_t M = 4;
            const std::uint64_t ef_construction = 50;

            const std::uint64_t next_id = 1;
            const std::uint64_t entry_point = 0;
            const std::uint64_t max_level = 0;

            const std::uint64_t record_count = 1;
            const std::uint64_t id = 0;
            const std::uint64_t record_dimension = 4;

            const float values[] = {
                1.0f, 2.0f, 3.0f, 4.0f
            };

            const std::uint64_t metadata_count = 0;

            const std::uint64_t node_count = 1;
            const std::uint64_t node_id = 0;
            const std::uint64_t level = 0;

            const std::uint64_t neighbour_count = 1;

            // Invalid: node 999 does not exist.
            const std::uint64_t invalid_neighbour = 999;

            file.write(
                reinterpret_cast<const char*>(&magic),
                sizeof(magic)
            );

            file.write(
                reinterpret_cast<const char*>(&version),
                sizeof(version)
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

            file.write(
                reinterpret_cast<const char*>(&record_count),
                sizeof(record_count)
            );

            file.write(
                reinterpret_cast<const char*>(&id),
                sizeof(id)
            );

            file.write(
                reinterpret_cast<const char*>(&record_dimension),
                sizeof(record_dimension)
            );

            file.write(
                reinterpret_cast<const char*>(values),
                sizeof(values)
            );

            file.write(
                reinterpret_cast<const char*>(&metadata_count),
                sizeof(metadata_count)
            );

            file.write(
                reinterpret_cast<const char*>(&node_count),
                sizeof(node_count)
            );

            file.write(
                reinterpret_cast<const char*>(&node_id),
                sizeof(node_id)
            );

            file.write(
                reinterpret_cast<const char*>(&level),
                sizeof(level)
            );

            file.write(
                reinterpret_cast<const char*>(&neighbour_count),
                sizeof(neighbour_count)
            );

            file.write(
                reinterpret_cast<const char*>(&invalid_neighbour),
                sizeof(invalid_neighbour)
            );
        }

        EXPECT_THROW(
            cortex::persistence::Serializer::load(path),
            std::out_of_range
        );

        std::remove(path.c_str());
    }
}