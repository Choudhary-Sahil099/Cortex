#include "core/vector_store.hpp"
#include <gtest/gtest.h>

#include <algorithm>
#include <string>
#include <cstdint>
#include <stdexcept>

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

}

TEST(VectorStoreTest, GeneratesStableIds)
{
    cortex::core::VectorStore store(3);

    const auto id0 =
        store.add(make_vector({ 1.0f, 2.0f, 3.0f }));

    const auto id1 =
        store.add(make_vector({ 4.0f, 5.0f, 6.0f }));

    EXPECT_EQ(id0, 0);
    EXPECT_EQ(id1, 1);
    EXPECT_EQ(store.size(), 2);
}

TEST(VectorStoreTest, RetrievesVectorRecord)
{
    cortex::core::VectorStore store(3);

    const auto id =
        store.add(make_vector({ 1.0f, 2.0f, 3.0f }));

    const auto& record = store.get(id);

    EXPECT_EQ(record.id, id);
    EXPECT_FLOAT_EQ(record.vector.data()[0], 1.0f);
    EXPECT_FLOAT_EQ(record.vector.data()[1], 2.0f);
    EXPECT_FLOAT_EQ(record.vector.data()[2], 3.0f);
}

TEST(VectorStoreTest, StoresMetadata)
{
    cortex::core::VectorStore store(3);

    cortex::core::metaData metadata{
        {"source", "email"},
        {"subject", "Meeting tomorrow"}
    };

    const auto id =
        store.add(
            make_vector({ 1.0f, 2.0f, 3.0f }),
            metadata
        );

    const auto& record = store.get(id);

    EXPECT_EQ(record.metadata.at("source"), "email");
    EXPECT_EQ(
        record.metadata.at("subject"),
        "Meeting tomorrow"
    );
}

TEST(VectorStoreTest, RejectsWrongDimension)
{
    cortex::core::VectorStore store(3);

    EXPECT_THROW(
        store.add(make_vector({ 1.0f, 2.0f })),
        std::invalid_argument
    );
}

TEST(VectorStoreTest, RejectsUnknownId)
{
    cortex::core::VectorStore store(3);

    EXPECT_THROW(
        store.get(999),
        std::out_of_range
    );
}

TEST(VectorStoreTest, EmptyStoreIsEmpty)
{
    cortex::core::VectorStore store(3);

    EXPECT_TRUE(store.empty());
    EXPECT_EQ(store.size(), 0);
}

TEST(VectorStoreTest, IDsIncreaseMonotonically)
{
    cortex::core::VectorStore store(3);

    const auto id0 =
        store.add(make_vector({ 1.0f, 2.0f, 3.0f }));

    const auto id1 =
        store.add(make_vector({ 4.0f, 5.0f, 6.0f }));

    const auto id2 =
        store.add(make_vector({ 7.0f, 8.0f, 9.0f }));

    EXPECT_EQ(id0, 0);
    EXPECT_EQ(id1, 1);
    EXPECT_EQ(id2, 2);
}

TEST(VectorStoreTest, RemovesExistingVector)
{
    cortex::core::VectorStore store(3);

    const auto id0 =
        store.add(make_vector({ 1.0f, 2.0f, 3.0f }));

    const auto id1 =
        store.add(make_vector({ 4.0f, 5.0f, 6.0f }));

    const auto id2 =
        store.add(make_vector({ 7.0f, 8.0f, 9.0f }));

    EXPECT_TRUE(store.remove(id1));

    EXPECT_EQ(store.size(), 2);

    EXPECT_THROW(
        store.get(id1),
        std::out_of_range
    );

    EXPECT_EQ(store.get(id0).id, id0);
    EXPECT_EQ(store.get(id2).id, id2);
}

TEST(VectorStoreTest, RemovingUnknownIdReturnsFalse)
{
    cortex::core::VectorStore store(3);

    EXPECT_FALSE(store.remove(999));
}

TEST(VectorStoreTest, RemovedIdIsNotReused)
{
    cortex::core::VectorStore store(3);

    const auto id0 =
        store.add(make_vector({ 1.0f, 2.0f, 3.0f }));

    const auto id1 =
        store.add(make_vector({ 4.0f, 5.0f, 6.0f }));

    EXPECT_TRUE(store.remove(id0));

    const auto id2 =
        store.add(make_vector({ 7.0f, 8.0f, 9.0f }));

    EXPECT_EQ(id0, 0);
    EXPECT_EQ(id1, 1);
    EXPECT_EQ(id2, 2);

    EXPECT_THROW(
        store.get(id0),
        std::out_of_range
    );

    EXPECT_EQ(store.get(id1).id, id1);
    EXPECT_EQ(store.get(id2).id, id2);
}

TEST(VectorStoreTest, ContainsExistingId)
{
    cortex::core::VectorStore store(3);

    const auto id =
        store.add(make_vector({ 1.0f, 2.0f, 3.0f }));

    EXPECT_TRUE(store.contains(id));
}

TEST(VectorStoreTest, DoesNotContainRemovedId)
{
    cortex::core::VectorStore store(3);

    const auto id =
        store.add(make_vector({ 1.0f, 2.0f, 3.0f }));

    EXPECT_TRUE(store.contains(id));

    EXPECT_TRUE(store.remove(id));

    EXPECT_FALSE(store.contains(id));
}

TEST(VectorStoreTest, UpdatesVectorAndMetadata)
{
    cortex::core::VectorStore store(3);

    const auto id =
        store.add(
            make_vector({ 1.0f, 2.0f, 3.0f }),
            {
                {"source", "old"},
                {"subject", "Old subject"}
            }
        );

    const bool updated =
        store.update(
            id,
            make_vector({ 10.0f, 20.0f, 30.0f }),
            {
                {"source", "new"},
                {"subject", "New subject"}
            }
        );

    EXPECT_TRUE(updated);

    const auto& record = store.get(id);

    EXPECT_EQ(record.id, id);

    EXPECT_FLOAT_EQ(
        record.vector.data()[0],
        10.0f
    );

    EXPECT_FLOAT_EQ(
        record.vector.data()[1],
        20.0f
    );

    EXPECT_FLOAT_EQ(
        record.vector.data()[2],
        30.0f
    );

    EXPECT_EQ(
        record.metadata.at("source"),
        "new"
    );

    EXPECT_EQ(
        record.metadata.at("subject"),
        "New subject"
    );
}

TEST(VectorStoreTest, UpdatingUnknownIdReturnsFalse)
{
    cortex::core::VectorStore store(3);

    EXPECT_FALSE(
        store.update(
            999,
            make_vector({ 1.0f, 2.0f, 3.0f })
        )
    );
}

TEST(VectorStoreTest, UpdateRejectsWrongDimension)
{
    cortex::core::VectorStore store(3);

    const auto id =
        store.add(
            make_vector({ 1.0f, 2.0f, 3.0f })
        );

    EXPECT_THROW(
        store.update(
            id,
            make_vector({ 1.0f, 2.0f })
        ),
        std::invalid_argument
    );
}

TEST(VectorStoreTest, ReturnsRawVectorData)
{
    cortex::core::VectorStore store(3);

    const auto id =
        store.add(
            make_vector({ 10.0f, 20.0f, 30.0f })
        );

    const float* data =
        store.vector_data(id);

    ASSERT_NE(data, nullptr);

    EXPECT_FLOAT_EQ(data[0], 10.0f);
    EXPECT_FLOAT_EQ(data[1], 20.0f);
    EXPECT_FLOAT_EQ(data[2], 30.0f);
}

TEST(VectorStoreTest, VectorDataRejectsUnknownId)
{
    cortex::core::VectorStore store(3);

    EXPECT_THROW(
        store.vector_data(999),
        std::out_of_range
    );
}
