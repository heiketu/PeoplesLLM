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

The implementation follows five rules:

1. Keep weight reads local whenever the workload is memory-bandwidth bound.
2. Use fine-grained row or tensor partitioning only inside low-latency domains.
3. Cross high-latency boundaries with coarse expert work and compact activations.
4. Preserve a deterministic merge order when a path claims bit-identical behavior.
5. Treat activation quantization, rounding, accumulation, nonlinearities, and speculative acceptance as an explicit cross-device precision contract.

No single runtime switch enables every mechanism. NUMA placement, CPU kernels, GPU prefill, speculative decoding, and remote EP are separate modules with explicit eligibility gates and fallbacks.

## Three-engine planning model

The runtime decisions are organized into three logical engines. They are interfaces implemented across the planner, loader, backends, and runtime, not three required daemon processes.

| Engine | Owns | Produces |
|---|---|---|
| Topology-Aware Engine (TAE) | CPU cache/die/NUMA topology, PCIe/UPI/RDMA paths, capacity and load, layer/expert/tensor granularity, mirror/split/owner placement, device-native layouts | `PlacementPlan`, `PhysicalLayoutPlan`, and `TransportPlan` |
| Unified Precision Engine (UPE) | Canonical tensor and data epoch, replica synchronization, weight interpretation, activation quantization, rounding/accumulation, nonlinearities, fold order, and validation profile | `PrecisionContract`, `SyncContract`, and the set of semantically valid replicas/kernels |
| Dynamic Matrix Engine (DME) | The current TG, PP, or speculative-verification shape after TAE/UPE eligibility checks | `MatrixExecutionPlan`: GEMV/GEMM, ISA/kernel, NR/MR, tile, batch/ubatch, fusion, and pipeline |

In compact form, TAE generates the physically feasible set, UPE removes stale or numerically invalid candidates, and DME minimizes the phase-specific critical path over their intersection. Hardware support for AMX or CUDA belongs to TAE; whether its numerical path is acceptable belongs to UPE; whether AMX, AVX512-VNNI, or CUDA wins for the current shape belongs to DME.

The same boundary applies to data movement. UPE states which logical tensor/data epoch and precision must be synchronized and when publication is complete. TAE chooses the physical NVLink, PCIe, UPI, or RDMA route. DME places copies, events, and barriers on the execution timeline. TAE should reject or warn about fine-grained plans across a GPU behind the PCH, a downgraded PCIe link, a remote socket detour, or a path without usable P2P; if no candidate crosses the measured break-even threshold, it should report that no profitable parallel plan exists.

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
| Unified-precision validation | `src/llama-hot-expert.cpp`, `ggml/src/ggml-cuda/mmvq.cu`, `common/speculative.cpp` | Strict slot folds, CPU/GPU shadow statistics, precision experiments, PPL/hash, and accepted/drafted gates |
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
  -> GPU resident hot experts -> six router-slot partials --+
  -> CPU cold experts --------------------------------------+-> slot 0..5 left-fold merge
```

The hot path is restricted by model type, tensor format, token count, and device capacity. Unsupported shapes use the ordinary CPU path. For each of the supported one-to-four target tokens, the strict path does not merge one aggregate GPU partial with one aggregate CPU partial: it returns one vector per router slot, then restores the baseline slot order on the CPU. AVX512 may vectorize across hidden rows, but never across the six-slot dependency chain. This distinction is required for the path's numerical contract.

The current staging buffers and completion events are process-global and have been validated only for one active slot. A multi-slot implementation must make this state context/slot-owned before enabling concurrent hot forks; it must not reuse the single-slot path speculatively.

Strict pure remote EP can opt into the same fork for one-to-four-token decode or speculative verification. The hot GPU computes slots whose expert is in the resident K set; the dealer receives a `[token,slot]` active mask and sends only cold slots to the NUMA workers. Sync REQ4 returns one unweighted vector per cold slot. The master applies each router weight once and folds GPU and CPU vectors in the original router-slot order for each token. This bridge is restricted to one active slot, KLOCAL=0, weight-on-master REQ4, and non-pipe execution. A per-layer in-flight guard rejects concurrent reuse of the global staging buffer.

Speculative decoding wraps the target decode loop. DSpark drafts candidate tokens on its configured device, then the target model verifies them as a small batch. Draft acceptance changes the target `MUL_MAT_ID` shape, so decode kernel dispatch must remain workload-aware.

## Unified Precision Engine

Hybrid EP has a numerical control plane in addition to placement and ownership. Two endpoints can decode the same GGUF weights yet disagree because they use different activation block formats, scale precision, rounding rules, dot-product grouping, FMA policy, nonlinear kernels, or reduction trees. Run-to-run determinism on each endpoint does not prove cross-device equivalence.

PeoplesLLM assigns this contract to the Unified Precision Engine (UPE). UPE also tracks the logical tensor and data epoch represented by each device copy, so synchronization completion and numerical validity share one publication boundary. A complete endpoint capability should eventually include a precision-contract ID covering:

- logical weight interpretation and scale/codebook validation;
- activation block size, `amax` reduction, scale storage, and rounding mode;
- integer accumulation width and floating-point scale recovery;
- clamp/GLU semantics and fusion order;
- router-weight rounding and global slot-fold order;
- the required validation profile: bit-exact, verify-strict, or quality-bounded.

The existing implementation covers a concrete but still incomplete part of this contract. Hot experts return one vector per router slot, and the master restores the slot 0-to-5 left fold. Shadow mode computes the same hot slot on CPU and GPU and records per-expert and whole-layer error. Release gates additionally track paired PPL, response hashes, and DSpark accepted/drafted counts. Scheduled EPD CAP now negotiates a versioned precision extension containing activation, dot, FFN-schema and per-slot-merge IDs plus model-schema, data-epoch and total-contract hashes. A new master requests the extension explicitly, so an old flags=0 master still receives the old CAP byte layout. `GGML_REMOTE_EP_UPE_STRICT=1` rejects a missing contract, an unknown/mismatched `GGML_EP_DATA_EPOCH`, cross-worker contract differences, and contract changes after reconnect. Automatic CPU/GPU strict/fallback placement and a GPU-side contract are still incomplete.

The extension has been exercised on the full DSV4 strict-modulo topology: two e64 workers on the master host and two on the remote host all report `contract=abf4d5f8cc4b7f1d`, `schema=9ffbda4782e1eea6`, and `epoch=fed537d7ada4d79e`; strict REQ4 completes a 32-token smoke. Substituting an incorrect master epoch aborts negotiation at the first CAP before expert work. This proves the CPU-endpoint gate, not CPU/GPU bit-exactness; the smoke throughput is not a formal performance sample.

The GPU-side produced-value boundary is observable in `test-cuda-q8-contract`. It invokes the same CUDA Q8\_1 quantizer used by MXFP4 MMVQ and copies its temporary blocks back. Smooth input gives zero code and FP16-scale mismatches against CPU Q8\_0; a deliberately half-step input gives 1,876/4,096 code mismatches but zero scale mismatches. Full replay of 489,856 real blocks (15,675,392 values) gives only 34 code mismatches, zero scale mismatches, and a maximum mismatching half-step distance of 1.5259e-5. The default-off phase-aware boundary fallback can reroute only sensitive GPU-hot slots to CPU workers. It restores the CPU body hash in a raw smoke while retaining part of the hot-path speed, but the DSpark nominal-only point reaches 34.8 tok/s and 67/120 accepted/drafted versus ordinary GPU-hot 37.9 and 66/122. One extra accepted token does not justify the tax or prove general acceptance recovery, so the policy remains an experimental causal gate. Dot recovery, nonlinear evaluation, and accumulation trees remain in the GPU-side contract scope.

A later hot-scoped CPU-Q8 hook provides a sharper separation. It changes only `hot_expert.*` MXFP4 activation-code generation to the CPU Q8\_0 RNE rule, reducing a real sample from 33/10,144 code mismatches to zero with zero scale mismatches. Ordinary-weight parity still has 2.29e-7–4.80e-7 relative error after code agreement, localizing the remainder to CUDA warp accumulation. In NMAX=2 warm-server A-B-B-A, it raises 35.0614 to 37.9573 tok/s (+8.2595%) and changes acceptance from 266/490 to 273/474, but five-chunk PPL worsens to 2.7855 versus strict-hot 2.7412 and no-hot 2.7647. The path is therefore production-rejected and default-off. This counterexample makes code/scale equality, speculative acceptance, and statistical quality separate UPE gates.

`GGML_HOT_EXPERT_UPE_VERIFY_CPU=1` provides a stricter phase-control experiment: n_tokens>1 bypasses GPU submission instead of computing both paths. It restores the CPU-remote body hash and 66/122 acceptance exactly, but reaches 30.7 tok/s versus CPU 31.2 because all 2,895 target-layer calls in this NMAX=2 sample are multi-token. This proves that the remaining trajectory difference is inside multi-token GPU expert execution rather than the draft model, while also proving that bypass is not an optimization. The required implementation target is a CPU-equivalent multi-token CUDA execution contract.

The weight-value contract now also covers malformed MXFP4 E8M0 bytes. Layer-21 experts 202/205 contain seven OCP-reserved `0xff` blocks and finite exponents up to 250. CUDA's native E8M0 conversion produced NaN while CPU_REPACK historically consumed `0xff` with a finite half-scale of $2^{127}$. CUDA MXFP4 MMVQ now spells out the CPU-compatible half-scale; injected 0xff, e250, and combined edge matrices are bit-identical. A spec-oriented global `0xff -> 0` policy was rejected because it worsened pure-CPU five-chunk PPL by 0.56%. Since the real experts remain sensitive to accumulation order across several extreme blocks, TAE can keep only `21:202,21:205` in the CPU domain through `GGML_HOT_EXPERT_EXCLUDE`, while replacing them with the next hot candidates on GPU. Paired 10-chunk PPL passes (2.5843 versus CPU 2.5853), but a three-prompt warm-server run lowers mean TG from 34.124 to 33.019 tok/s and worsens one prompt's acceptance from 61/130 to 56/140. Repeated hashes are deterministic inside each placement. The placement therefore remains default-off: local shadow/PPL repair is not sufficient evidence of a held-out system Pareto gain.

This distinction matters most for speculative verification. Acceptance is a discrete target-token decision, so a small expert-vector perturbation near a logit tie can reject a draft token and alter the following trajectory even when cosine error is small and both backends are individually deterministic. Consequently, bandwidth or kernel speed is not accepted as useful EP capacity until the selected precision profile also passes target-token and quality gates.

The evidence levels must not be collapsed. An accepted/drafted change with identical final target tokens establishes only a speculative-efficiency effect. A reproducible target-logit, top-1, or generated-token difference between pure CPU and GPU expert offload proves that the numerical difference reaches model decisions and can affect quality, but it does not say which path is better. A quality direction or degradation claim requires paired non-speculative PPL, task-accuracy, or behavioral evidence on held-out data.

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

In the verified raw/no-DSpark bridge, CUDA1 hot slots reduce CPU assignments from 1,806 to 916 across 301 decode-layer calls. Four strict-cover workers receive 239/220/213/244 assignments, while TG512 improves from 25.25 to 28.85 tok/s and paired five-chunk PPL improves from 2.7647 to 2.7412. This result is not MAX_EFFORT replica mode and does not cover multiple slots or async PIPE.

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

The project uses four different correctness levels:

- Bit-identical paths preserve the operation and merge order and compare output bytes or hashes.
- Numerically equivalent paths change floating-point association and compare bounded error plus deterministic generation.
- Quality-equivalent formats change representation and require perplexity or task-level evaluation in addition to kernel tests.
- Speculative-equivalent paths also preserve or explicitly gate target tokens, first divergence, and accepted/drafted counts.

A result must state which contract applies. Run-to-run determinism does not prove equivalence to another backend, and a matching checksum over an empty output is not a valid test.

## Validation map

| Change area | Minimum validation |
|---|---|
| CPU repack or format | `test-repack-kernels`, `test-repack-iq`, format-specific tests, ASan when buffer ownership changes |
| NUMA placement or claim | CPU build, repack tests, shard-plan tests, fixed-output comparison, matched performance A/B |
| Remote EP | dealer, topology, credit, session, protocol, expert-map tests and `llama-epd --selftest` |
| Layer-major or prefill scheduler | `test-layer-major`, `test-server-prefill-scheduler`, KV eligibility and fallback tests |
| CUDA op or collective | CUDA backend-op test, deterministic output check, single-device fallback, and matched A/B |
| Hot-expert path | resident-set initialization, 64-mask scalar/AVX slot-order comparison, CPU/GPU shadow error, fixed-seed repeatability, paired PPL, capacity check, disabled-feature fallback, and single-slot ownership gate |
| Speculative decoding | target output/hash comparison, first divergence, draft/accepted counts, and NMAX sweep under one prompt |

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
