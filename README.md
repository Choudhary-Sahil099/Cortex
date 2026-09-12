# Cortex

## High-Performance C++ Vector Engine

Cortex is a high-performance vector computation engine written in modern C++20.

The project focuses on building the low-level numerical infrastructure used by vector search and similarity-search systems, with an emphasis on:

* SIMD acceleration
* CPU feature detection
* Cache-friendly memory layouts
* Vectorized distance computation
* Batch processing
* Top-K search
* Performance benchmarking

Cortex is being developed as a foundation for a future approximate nearest neighbor (ANN) indexing layer.

## Phase 1 — Vector Engine

Phase 1 implements the core vector computation engine.

### Implemented

* Aligned memory allocation
* Dynamic vector representation
* Dimension validation
* Scalar L2 distance
* AVX2 L2 distance
* AVX2 + FMA L2 distance
* Runtime CPU feature detection
* Automatic SIMD backend selection
* Scalar / AVX2 / AVX2+FMA dot product
* Cosine similarity
* Batch L2 distance
* Contiguous vector storage
* Top-K vector search
* GoogleTest correctness tests
* Google Benchmark performance tests

## Architecture

**Plaintext**

```
                           Cortex
                             │
                ┌────────────┴────────────┐
                │                         │
             Vector                  VectorStore
                │                         │
                └────────────┬────────────┘
                             │
                       Vector Backend
                             │
                ┌────────────┼────────────┐
                │            │            │
              Scalar        AVX2      AVX2 + FMA
                │            │            │
                └────────────┴────────────┘
                             │
                   Distance / Similarity
                             │
                ┌────────────┴────────────┐
                │                         │
         Batch Processing            Top-K Search
```

## SIMD Backend Selection

Cortex detects CPU capabilities at runtime and automatically selects the fastest supported backend.

**Plaintext**

```
CPU supports AVX2 + FMA
        ↓
   AVX2 + FMA

CPU supports AVX2
        ↓
      AVX2

Otherwise
        ↓
     Scalar
```

This allows the same application to use hardware-specific optimizations without requiring the caller to manually select an implementation.

## Memory Layout

Cortex uses aligned memory for individual vectors.

**Vector**

**Plaintext**

```
┌─────────────────────────────────────────────┐
│ f0 │ f1 │ f2 │ f3 │ ... │ f1534 │ f1535 │
└─────────────────────────────────────────────┘
                      ↓
               32-byte aligned
```

**VectorStore** provides contiguous storage for multiple vectors:

**Plaintext**

```
Vector 0: [f0 f1 f2 ... f1535]
Vector 1: [f0 f1 f2 ... f1535]
Vector 2: [f0 f1 f2 ... f1535]
...
Vector N: [f0 f1 f2 ... f1535]
```

The contiguous representation allows the distance kernels to operate directly on raw floating-point memory and avoids unnecessary temporary vector copies.

## Supported Operations

### L2 Distance

Cortex provides three implementations:

**C++**

```
l2_distance_scalar(a, b);
l2_distance_avx2(a, b);
l2_distance_avx2_fma(a, b);
```

The public API can automatically select the best backend:

**C++**

```
l2_distance(a, b);
```

### Dot Product

Three implementations are available:

**C++**

```
dot_product_scalar(a, b);
dot_product_avx2(a, b);
dot_product_avx2_fma(a, b);
```

The automatic interface is:

**C++**

```
dot_product(a, b);
```

### Cosine Similarity

Cortex provides cosine similarity using the optimized vector operations:

**C++**

```
cosine_similarity(a, b);
```

### Batch Distance

Multiple vectors can be compared against a single query:

**C++**

```
batch_l2_distance(
    query,
    vectors,
    distances
);
```

The backend selected by Cortex is used for the distance computation.

### Top-K Search

Cortex provides Top-K similarity/distance search over a collection of vectors.

**Plaintext**

```
    Query
      │
      ▼
Distance computation
      │
      ▼
Candidate distances
      │
      ▼
Top-K selection
      │
      ▼
Nearest vectors
```

This forms the basis for the future ANN indexing layer.

## Performance

Benchmarks are implemented using Google Benchmark.

The current development machine supports:

**Plaintext**

```
AVX2 : supported
FMA  : supported

Selected Backend:
AVX2 + FMA
```

Example batch benchmark:

**Plaintext**

```
BM_BatchL2Distance

~100–130 µs
~8–10M vectors/sec
```

For the benchmark configuration:

**Plaintext**

```
Vectors          : 1000
Dimensions/vector: 1536
```

This corresponds to roughly:

> **~12–15 billion vector dimensions processed per second**

Actual benchmark results depend on CPU, compiler, build configuration, memory hierarchy, and system load.

## Example Output

**Plaintext**

```
====================
Cortex Vector Engine
====================

CPU Features
------------
AVX2 : supported
FMA  : supported

Selected Backend
----------------
AVX2 + FMA

Vector Operations
-----------------
Dimension         : 1536
L2 Distance       : 34.7387
Dot Product       : 2413.56
Cosine Similarity : 1

Batch Processing
----------------
Vectors           : 1000
Dimensions/vector : 1536
Distances computed: 1000
First distance    : 23.3151
=================================
```

## Project Structure

**Plaintext**

```
Cortex/
│
├── include/
│   ├── core/
│   │   ├── buffer.hpp
│   │   └── cpu_features.hpp
│   │
│   └── vector/
│       ├── vector.hpp
│       ├── vector_store.hpp
│       ├── distance.hpp
│       ├── dot_product.hpp
│       ├── similarity.hpp
│       ├── backend.hpp
│       ├── batch_distance.hpp
│       └── search.hpp
│
├── src/
│   ├── core/
│   │   ├── buffer.cpp
│   │   └── cpu_features.cpp
│   │
│   ├── vector/
│   │   ├── vector.cpp
│   │   ├── vector_store.cpp
│   │   ├── distance.cpp
│   │   ├── distance_scalar.cpp
│   │   ├── dot_product.cpp
│   │   ├── similarity.cpp
│   │   ├── backend.cpp
│   │   ├── batch_distance.cpp
│   │   └── search.cpp
│   │
│   └── main.cpp
│
├── tests/
│   ├── buffer_test.cpp
│   ├── vector_test.cpp
│   ├── distance_test.cpp
│   └── batch_distance_test.cpp
│
├── benchmarks/
│   └── distance_benchmark.cpp
│
├── CMakeLists.txt
└── README.md
```

## Building

### Requirements

* C++20 compiler
* CMake 3.20+
* Ninja or Visual Studio build tools
* x64 processor
* GoogleTest
* Google Benchmark

Dependencies are automatically fetched through CMake FetchContent.

### Build

**Bash**

```
cmake -S . -B build
cmake --build build --config Release
```

For Visual Studio/Ninja multi-configuration builds:

**Bash**

```
cmake --build out/build/x64-Release --config Release
```

## Running

### Application

**Bash**

```
./cortex
```

On Windows:

**DOS**

```
cortex.exe
```

### Tests

Cortex uses GoogleTest.

Run the test executable:

**DOS**

```
cortex_tests.exe
```

Or use CTest:

**Bash**

```
ctest --test-dir build
```

All Phase 1 correctness tests currently pass.

### Benchmarks

Run:

**DOS**

```
cortex_benchmarks.exe
```

Example benchmark categories include:

* Scalar L2
* AVX2 L2
* AVX2 + FMA L2
* Automatic L2
* Scalar Dot Product
* AVX2 Dot Product
* AVX2 + FMA Dot Product
* Automatic Dot Product
* Cosine Similarity
* Batch L2 Distance
* Top-K Search

## Design Goals

Cortex is designed around several systems-level principles:

1. **Hardware-aware computation:** Use runtime CPU feature detection to select optimized implementations.
2. **SIMD-friendly memory:** Use aligned and contiguous floating-point data to improve vectorized processing.
3. **Separation of computation and dispatch:** The backend abstraction separates:
   **Plaintext**

   ```
   What operation?
         ↓
   Which implementation?
         ↓
   Which CPU instructions?
   ```
4. **Benchmark-driven optimization:** Performance changes are evaluated using repeatable microbenchmarks rather than assuming that an optimization is faster.
5. **Foundation for vector search:** The vector engine is intentionally separated from the future indexing layer.

## Roadmap

### Phase 1 — Vector Engine

**Status:** Complete

* [X] Vector representation
* [X] Aligned memory
* [X] Scalar distance
* [X] SIMD distance
* [X] SIMD dot product
* [X] Cosine similarity
* [X] Runtime backend selection
* [X] Batch processing
* [X] VectorStore
* [X] Top-K search
* [X] Correctness tests
* [X] Performance benchmarks

### Phase 2 — ANN Indexing

**Status:** Planned

* [ ] HNSW graph
* [ ] Graph construction
* [ ] Nearest-neighbor search
* [ ] Configurable `M`
* [ ] `efConstruction`
* [ ] `efSearch`
* [ ] Layered graph structure
* [ ] Search performance benchmarks
* [ ] Recall@K evaluation

### Phase 3 — Production Optimization

**Status:** Planned

* [ ] Improved memory allocator
* [ ] Memory pooling
* [ ] Parallel batch search
* [ ] Multi-threaded indexing
* [ ] NUMA-aware optimizations
* [ ] Advanced cache optimization
* [ ] AVX-512 backend
* [ ] More extensive profiling

## Testing Philosophy

The project uses correctness tests alongside performance benchmarks.

The tests cover:

* Vector construction
* Dimensions
* Buffer behavior
* Distance correctness
* SIMD/scalar equivalence
* Dot product correctness
* Cosine similarity
* Batch processing
* Invalid dimensions
* Top-K behavior

The benchmark suite is kept separate from correctness tests so performance experiments do not affect functional verification.

## Technical Highlights

| **Parameter**    | **Specification** |
| ---------------------- | ----------------------- |
| **Language**     | C++20                   |
| **Architecture** | x64                     |
| **SIMD**         | AVX2                    |
| **FMA**          | AVX2 FMA                |
| **Memory**       | 32-byte aligned         |
| **Build System** | CMake + Ninja           |
| **Testing**      | GoogleTest              |
| **Benchmarking** | Google Benchmark        |

## Current Status

**Plaintext**

```
Cortex Vector Engine
====================

Phase 1: COMPLETE

Vector operations   ████████████████████ 100%
SIMD acceleration   ████████████████████ 100%
Batch processing    ████████████████████ 100%
Top-K search        ████████████████████ 100%
Testing             ████████████████████ 100%
Benchmarking        ████████████████████ 100%

Next:
HNSW indexing layer
```

## License

This project is currently under development.
