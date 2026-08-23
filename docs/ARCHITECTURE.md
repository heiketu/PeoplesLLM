# PeoplesLLM architecture

This document maps the public implementation to its runtime data flows. User-facing setup belongs in [QUICKSTART.md](QUICKSTART.md), configuration semantics belong in [PARAMETERS.md](PARAMETERS.md), and measured results belong in [BENCHMARKS.md](BENCHMARKS.md).

## Design principles

PeoplesLLM treats local inference as a hierarchy of execution domains rather than one uniform CPU and one uniform GPU:

```text
CPU cores and caches
  -> NUMA nodes and sockets
    -> local GPU backends
      -> remote NUMA workers
        -> remote machines
```

The implementation follows four rules:

1. Keep weight reads local whenever the workload is memory-bandwidth bound.
2. Use fine-grained row or tensor partitioning only inside low-latency domains.
3. Cross high-latency boundaries with coarse expert work and compact activations.
4. Preserve a deterministic merge order when a path claims bit-identical behavior.

No single runtime switch enables every mechanism. NUMA placement, CPU kernels, GPU prefill, speculative decoding, and remote EP are separate modules with explicit eligibility gates and fallbacks.

## Component map

| Area | Main files | Responsibility |
|---|---|---|
| Model loading and placement | `src/llama-model.cpp`, `src/llama-model-loader.cpp`, `src/llama-numa.cpp` | Select buffer types, load tensors, place expert pages, and initialize optional resident experts |
| Graph construction | `src/llama-graph.cpp`, `src/models/deepseek4.cpp` | Build attention, dense, router, MoE, sparse-attention, remote-EP, and hot-expert graph paths |
| Context execution | `src/llama-context.cpp` | Build ubatches, choose layer-major eligibility, control op offload, and execute graphs |
| CPU backend | `ggml/src/ggml-cpu/ggml-cpu.c`, `ggml/src/ggml-cpu/repack.cpp` | Thread pools, NUMA barriers, row claims, repack dispatch, `MUL_MAT`, and `MUL_MAT_ID` |
| x86 kernels | `ggml/src/ggml-cpu/arch/x86/repack.cpp` | AVX2/AVX-512/VNNI/VBMI GEMV and GEMM kernels, arec paths, and format-specific panels |
| Quantized formats | `ggml/src/ggml-common.h`, `ggml/src/ggml-quants.c`, `ggml/src/ggml-cpu/traits.cpp` | Block definitions, reference quantize/dequantize code, validation, and backend traits |
| AMX backend | `ggml/src/ggml-cpu/amx/` | AMX packing and matrix kernels with runtime hardware permission checks |
| CUDA backend | `ggml/src/ggml-cuda/` | Attention, top-k, MoE MMQ, P2P all-reduce, and GPU execution |
| GPU streaming prefill | `src/llama-layer-major.cpp`, `ggml/src/ggml-backend.cpp` | Layer-major execution, host-resident expert streaming, prefetch slots, and async readback |
| Hot-expert execution | `src/llama-hot-expert.cpp`, `src/llama-hot-expert.h` | Keep a profiled expert subset on a GPU and fork hot/cold MoE work inside the graph |
| Router profiling | `ggml/src/ggml-cpu/xllama-hot-trace.cpp`, `ggml/src/ggml-cpu/xllama-hot-trace.h` | Capture bounded temporal expert-selection traces for exact cache-policy replay |
| Remote EP master | `src/llama-remote-ep.cpp`, `src/llama-remote-ep.h` | Endpoint negotiation, expert dispatch, reconnect, and deterministic response merge |
| Remote EP worker | `tools/epd/llama-epd.cpp`, `tools/epd/llama-epd-runtime.h` | Load owned experts and execute remote MoE requests |
| EP protocol and transport | `tools/epd/llama-ep-protocol.*`, `tools/epd/llama-ep-transport.*`, `tools/epd/llama-ep-rdma.cpp` | Frame formats, TCP, RoCEv2, capability negotiation, and transport fallback |
| EP scheduling policy | `tools/epd/llama-ep-dealer.h`, `tools/epd/llama-ep-topology.h`, `tools/epd/llama-ep-capability.h` | Holder maps, coverage checks, repeat-aware assignment, and reusable scheduling workspace |
| Speculative decoding | `common/speculative.cpp` | Draft model execution, DSpark integration, acceptance, and optional profiling |
| Shared sharding model | `ggml/include/ggml-shard-plan.h` | Describe owner, mirrored, split, and partial placement across logical domains |

## Decode data flow

The common hybrid decode path keeps attention, dense layers, the router, and KV state on GPU backends while routed experts use the CPU repack backend.

```text
GPU layer input
  -> attention / dense / router
  -> selected expert ids + activations
  -> CPU_REPACK MUL_MAT_ID expert FFN
  -> GPU residual and next layer
```

With `GGML_NUMA_EP=1`, each expert tensor stays logically single-copy. Rows inside every expert plane are placed in NUMA-local windows. CPU workers claim output rows from their local windows, and each destination row has one writer. This avoids a cross-socket output reduction.

The hot-expert path is an optional fork inside the MoE graph:

```text
router output
  -> GPU resident hot experts -----+
  -> CPU cold experts -------------+-> ordered merge
```

The hot path is restricted by model type, tensor format, token count, and device capacity. Unsupported shapes use the ordinary CPU path.

Speculative decoding wraps the target decode loop. DSpark drafts candidate tokens on its configured device, then the target model verifies them as a small batch. Draft acceptance changes the target `MUL_MAT_ID` shape, so decode kernel dispatch must remain workload-aware.

## Prefill data flow

Ordinary chunked prefill processes one ubatch through every layer before advancing to the next ubatch. Layer-major prefill reverses that nesting when the request and cache state are eligible:

```text
token-major: ubatch 0 through all layers -> ubatch 1 through all layers
layer-major: all ubatches through layer 0 -> all ubatches through layer 1
```

Layer-major execution allows a host-resident expert tensor to be uploaded once per layer and reused across token tiles. `ggml/src/ggml-backend.cpp` owns streaming slots and events; `src/llama-layer-major.cpp` owns request slicing, layer boundaries, and fallback conditions. KV and SWA capacity checks remain authoritative; an executor cannot enlarge physical cache capacity by overriding a tile size.

GPU expert-axis EP is separate from remote CPU EP. It splits a layer's expert dimension across local GPUs and merges compact partial activations. It is useful only for large prefill shapes and stays behind explicit eligibility gates.

## Remote expert parallelism

Remote EP sends activations and routing metadata, not expert weights. A request follows this sequence:

```text
master graph
  -> validate holder coverage and endpoint capabilities
  -> build a deterministic assignment plan
  -> send compact per-endpoint requests
  -> workers execute locally owned experts
  -> receive partial outputs
  -> merge by global slot order
```

Each NUMA node can run one `llama-epd` process. Multiple workers cooperate on the same model request and slot; they are not independent model replicas. `SCHED_KLOCAL=0` is strict pure EP: the master cannot silently fall back to local routed-expert weights when coverage or transport fails.

Transport selection is runtime-configured. RDMA requires matching support on master and worker; connection failure can fall back to TCP where the protocol permits it. Capability negotiation includes expert ownership and kernel compatibility. A reconnect is accepted only when the replacement worker reports the same execution contract.

## Placement and ownership

Three placement levels must not be conflated:

- Physical placement decides where tensor pages or device buffers live.
- Compute ownership decides which domain executes an output range or expert slot.
- Merge ownership decides where partial results become the next layer input.

`ggml-shard-plan.h` provides shared vocabulary for these decisions. Current paths use `OWNER`, `MIRRORED`, `SPLIT`, and `PARTIAL` placements. A new backend should map its topology to logical domain IDs instead of introducing another independent split formula.

## Quantized format lifecycle

A CPU format crosses four boundaries:

```text
GGUF bytes
  -> loader validation
  -> optional backend layout preparation
  -> GEMV/GEMM or MUL_MAT_ID dispatch
  -> scalar/generic fallback when the ISA is unavailable
```

New formats must provide a reference representation before an optimized kernel:

1. Define block layout and row size in ggml common headers.
2. Add reference quantize/dequantize and validation cases.
3. Register type traits and conversion policy.
4. Implement generic compute or a correct fallback.
5. Add architecture kernels and runtime ISA gates.
6. Test `MUL_MAT` and `MUL_MAT_ID`, including small decode rows and large prefill batches.
7. Validate an actual GGUF with the native loader, not only an external conversion script.

Repack is a backend layout, not a quantization format. A byte-only panelization can preserve every logical weight value while still changing storage order at load time. Documentation must distinguish bit-exact weight values, bit-exact accumulation, and quality-equivalent end-to-end output.

Temporal expert traces are separate from aggregate frequency counters. Aggregate counters size a static resident set; a temporal trace evaluates replacement policy. The trace writer is default-off, buffers records in memory, writes a footer with drop/overlap metadata, and only calls a replay exact when every capture ordinal and integrity field is complete.

## Configuration levels

Configuration is divided by stability rather than by file location:

- CLI and profiles are the supported deployment interface. See `tools/peoplesllm-run.sh` and `docs/QUICKSTART.md`.
- Documented environment variables are advanced controls with defined defaults. See `docs/PARAMETERS.md`.
- Debug, trace, forced-kernel, and rejected-path switches are experimental. They must remain default-off and must not appear in recommended commands.

Every new variable needs one source read point, a documented default, a fallback, and a test or reproducible diagnostic. A variable that parses but does not change behavior is dead configuration and should be removed.

## Correctness contracts

The project uses three different correctness levels:

- Bit-identical paths preserve the operation and merge order and compare output bytes or hashes.
- Numerically equivalent paths change floating-point association and compare bounded error plus deterministic generation.
- Quality-equivalent formats change representation and require perplexity or task-level evaluation in addition to kernel tests.

A result must state which contract applies. Run-to-run determinism does not prove equivalence to another backend, and a matching checksum over an empty output is not a valid test.

## Validation map

| Change area | Minimum validation |
|---|---|
| CPU repack or format | `test-repack-kernels`, `test-repack-iq`, format-specific tests, ASan when buffer ownership changes |
| NUMA placement or claim | CPU build, repack tests, shard-plan tests, fixed-output comparison, matched performance A/B |
| Remote EP | dealer, topology, credit, session, protocol, expert-map tests and `llama-epd --selftest` |
| Layer-major or prefill scheduler | `test-layer-major`, `test-server-prefill-scheduler`, KV eligibility and fallback tests |
| CUDA op or collective | CUDA backend-op test, deterministic output check, single-device fallback, and matched A/B |
| Hot-expert path | resident-set initialization, hot/cold hash comparison, capacity check, fallback with the feature disabled |
| Speculative decoding | target output comparison, draft/accepted counts, and NMAX sweep under one prompt |

The current build and test commands are maintained in [DEVELOPMENT.md](DEVELOPMENT.md).

## Upstream merge boundaries

The largest conflict surfaces are upstream files that contain fork hot paths:

- `ggml/include/ggml.h` and `ggml/src/ggml-common.h` for new public types.
- `ggml/src/ggml-cpu/ggml-cpu.c` for barriers and execution.
- `ggml/src/ggml-cpu/repack.cpp` and x86 `repack.cpp` for dispatch and kernels.
- `src/llama-model.cpp` for placement and loading.
- `src/llama-graph.cpp` for graph eligibility and hybrid execution.
- `ggml/src/ggml-backend.cpp` for scheduler and streaming state.

Prefer new, independently testable modules for policy, topology, protocol, and profiling. Keep architecture-specific inner loops in their existing translation units when extraction would prevent inlining or duplicate kernel state. After every upstream merge, rebuild both CPU and CUDA configurations and rerun the relevant fallback paths; a successful compile is not sufficient evidence that buffer-type selection or graph placement stayed correct.
