# Cortex — production-oriented core hardening

This is a source-level patch to the supplied Cortex snapshot, not a claim that the entire application is production-ready. The original architecture, CLI, index format v2 and existing test/benchmark sources are retained.

## Changes

- **BM25 lexical retrieval:** actual application search now uses a lazily rebuilt in-memory inverted index (term frequency, document frequency, length normalization, k1=1.2, b=0.75). Repeated searches between mutations score only matching postings; top-k is selected using partial sort. Cache is invalidated before insert/remove and reconstructed on reload. Hybrid RRF consumes these ranks without changing its API. Removed unsolicited per-query ranking output from Retriever. ASCII tokenization only, no multilingual segmentation. Query term repetitions are treated as one term. Index access must be externally serialized, as before; this is not a concurrent service.
- **Snapshot reliability:** writes are staged in a uniquely created directory on the same filesystem, checked for stream flush/close failures, then published by rename (POSIX) or MoveFileExW (Windows). Failure cleans staging and does not intentionally truncate the old snapshot. POSIX staging directory uses mkdtemp. This is atomic publication, not guaranteed power-loss durability: no fsync/parent-directory sync or checksum is implemented. Single writer, trusted local directory assumed; permissions and concurrent writer policy still need review. Crash-abandoned staging directories may remain.
- **Defensive loading:** check byte counts against the opened file length before allocations, cap dimension (65,536), M (65,536), ef values (10 million), max graph level (64), reject non-finite vector values, invalid IDs, mismatched node/record counts and trailing data. These are deliberate limits for the local application and may reject exotic previously accepted indexes. Native-endian v2 file format retained; no cross-endian support. Limits do not constitute a complete untrusted-file sandbox.
- **CPU portability:** isolate AVX2 kernels in dedicated translation units, keep public dispatch and scalar code baseline, use per-function FMA target attributes on GCC/Clang and check Windows CPUID/OSXSAVE/XGETBV before AVX. MSVC conservatively uses scalar on AVX2-without-FMA hosts. Public explicit SIMD entry points fall back when unsupported. Benchmark reports should record the detected backend to avoid labeling a fallback as SIMD. Scope is x86/x86-64; ARM is not supported by this patch. Aligned buffers now support POSIX and guard rounding overflow.
- **Build/test:** dependency-free `CORTEX_CORE_ONLY` option; CI matrix for Linux/Windows. Full build accepts ONNXRUNTIME_ROOT and selects the platform library. Linux full build is not verified; the user must supply matching ONNX SDK, CURL, nlohmann_json and model assets. The old full GoogleTest suite is retained but not run here.

## Run core regressions (Windows PowerShell or Linux shell)

From project root:

```sh
cmake -S . -B build-core -DCORTEX_CORE_ONLY=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build-core --config Release --parallel 2
ctest --test-dir build-core -C Release --output-on-failure
```

No models, ONNX Runtime binaries, curl, GoogleTest or network access are required for this mode. It does not build the application executable. Regression executable checks BM25 term frequency and length, punctuation and case, no matches, mutations/reload, atomic replacement, failed replacement cleanup, corrupted count rejection and scalar/SIMD parity. On Windows, use Visual Studio Developer PowerShell with the C++ workload installed.

## Full application validation

Restore the original dependencies/models (never commit them or real emails). Configure a new build directory using your existing package-manager/toolchain settings and `-DONNXRUNTIME_ROOT=...`. Run the entire existing GoogleTest suite, then test ingest -> save -> restart -> ask with a synthetic email collection. Rerun all benchmarks; old benchmark numbers are not evidence for the modified code. Benchmark cold vs warm BM25 separately, record corpus size, hardware, build flags, latency percentiles, memory, Recall@10 and MRR@10. Evaluate on representative labeled queries. Compare only equivalent scoring methods/datasets when claiming speedups.

## Remaining release gates

1. Test on Windows/MSVC, non-AVX CPU, and full ONNX/LLM pipeline; CI workflow is included but has not run remotely.
2. Add checksummed/versioned portable snapshots, durable fsync/FlushFileBuffers protocol and real crash/fault injection tests. Concurrent writer locks and secure snapshot permissions are required for multi-user deployments.
3. Incremental lexical indexing for mutation-heavy workloads; Unicode tokenizer and memory/resource budgets. Existing cache uses full rebuild after mutations.
4. Review HNSW deletion/graph invariants and graph restore semantics, benchmark at larger scales, add sanitizer/fuzz coverage. Existing result pointers are invalidated by mutations.
5. Verify end-to-end privacy (local endpoint only, consent for email ingestion, deletion and retention, no sensitive logs, backups, authentication if exposed). No claims of compliance or real customer deployment are made.
