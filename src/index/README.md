## HNSW Index

Cortex implements a hierarchical navigable small world (HNSW) graph for
approximate nearest-neighbor search.

### Configuration

| Parameter | Description |
|---|---|
| `M` | Maximum number of neighbors for upper HNSW layers |
| `2M` | Maximum number of neighbors at layer 0 |
| `ef_construction` | Number of candidates explored while constructing the graph |
| `ef_search` | Number of candidates explored during search |

### Search Process

HNSW search operates in two stages:

1. **Greedy traversal**
   - Start from the index entry point at the highest layer.
   - Move toward increasingly closer nodes.
   - Descend through the hierarchy until reaching layer 0.

2. **Layer-0 search**
   - Perform candidate-based graph exploration.
   - Maintain a candidate queue and result set.
   - Explore up to `ef_search` candidates.
   - Return the closest `k` vectors.

### Graph Construction

During insertion:

1. Generate a random level for the new node.
2. Start from the current entry point.
3. Greedily descend through higher layers.
4. Search for candidate neighbors at each applicable layer.
5. Select diverse neighbors.
6. Create bidirectional connections.
7. Prune connections exceeding the layer's degree limit.
8. Update the entry point if the new node reaches a higher level.

Layer 0 uses `2M` connections while upper layers use `M` connections.

### Benchmark

Benchmark configuration:

- Dataset: 10,000 vectors
- Dimension: 128
- Queries: 100
- `M`: 32
- Layer-0 maximum degree: 64
- `ef_construction`: 100
- `ef_search`: 400
- `k`: 10

Results:

| Metric | HNSW | Brute Force |
|---|---:|---:|
| Recall@10 | 100% | 100% |
| Average query latency | ~1.42 ms | ~8.75 ms |
| Build time | ~6.06 s | — |

These measurements are from the current Cortex HNSW benchmark configuration and are intended as a baseline rather than a universal performance claim.