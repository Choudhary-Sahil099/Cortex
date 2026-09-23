For the design of the persistence layer we are going to use a binary file

Reason --> lets suppose that our vector contains 1,000,000 X 1536 floats . This size will cause the problem if we have to use a json file as it decreases the read and the write spedd


The structure of the file is as such

┌──────────────────────────────┐
│          HEADER              │
├──────────────────────────────┤
│ Magic                        │
│ Format version               │
│                              │
│ Dimension                    │
│ M                            │
│ ef_construction              │
├──────────────────────────────┤
│        INDEX STATE           │
├──────────────────────────────┤
│ next_id                      │
│ entry_point                  │
│ max_level                    │
├──────────────────────────────┤
│       VECTOR RECORDS         │
├──────────────────────────────┤
│ Record count                 │
│                              │
│ Record 0                     │
│   ID                         │
│   vector dimension floats    │
│   metadata                   │
│                              │
│ Record 1                     │
│   ID                         │
│   vector dimension floats    │
│   metadata                   │
│                              │
│ ...                          │
├──────────────────────────────┤
│          HNSW GRAPH          │
├──────────────────────────────┤
│ Node count                   │
│                              │
│ Node 0                       │
│   ID                         │
│   level                      │
│   neighbors at level 0       │
│   neighbors at level 1       │
│   ...                        │
│                              │
│ Node 1                       │
│   ...                        │
└──────────────────────────────┘