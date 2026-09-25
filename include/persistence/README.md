## Persistence

Cortex supports binary persistence for the HNSW index through the
`cortex::persistence::Serializer` class.

### Save

```cpp
cortex::persistence::Serializer::save(
    index,
    "index.cortex"
);

Load
auto index =
    cortex::persistence::Serializer::load(
        "index.cortex"
    );


File Format
The .cortex file contains:
CORTEX INDEX FILE
│
├── Header
│   ├── Magic
│   └── Version
│
├── Configuration
│   ├── Dimension
│   ├── M
│   └── ef_construction
│
├── Index State
│   ├── next_id
│   ├── entry_point
│   └── max_level
│
├── Vector Records
│   ├── Vector ID
│   ├── Dimension
│   ├── Vector data
│   └── Metadata
│
└── HNSW Graph
    ├── Node ID
    ├── Node level
    └── Neighbor lists

Persistence Guarantees
The persistence layer preserves:
- Vector IDs
- Vector values
- Metadata
- HNSW node levels
- HNSW graph edges
- Entry point
- Maximum HNSW level
- Next available vector ID
ef_search is not persisted because it is a runtime search parameter.
The loader validates the file header, version, dimensions, node levels,
and graph references before restoring the index.
The persistence implementation currently provides index save/load
functionality. Crash-safe durability and WAL-based recovery are planned
for a later stage.


Important note -> don't use format document or else have trouble