# PeoplesLLM

> **[中文 README](README.md)**

**Run 200B-3T MoE models on local servers while making multiple CPU sockets, consumer GPUs, and remote memory contribute to one inference job.**

PeoplesLLM is a llama.cpp fork for heterogeneous local inference. It evolves along three connected tracks: a multi-NUMA/multi-device execution architecture, AVX512 and GPU kernels, and compute-oriented GGUF weight formats. The standalone `tools/epd/` runtime provides expert parallelism across machines.

## Scope

The main model's attention, dense layers, router, and KV cache can stay on the GPU while routed MoE experts run on a dual-socket CPU or remote NUMA workers. The goal is not to run one model replica or one slot per node: multiple NUMA nodes cooperate on **the expert work of one layer for one slot**.

The author chooses the architecture and hardware direction; implementation, testing, and tuning are AI-assisted. Every public performance claim below comes from a measured, matched comparison. This is still a research and engineering project, so production use requires output and capacity validation on the target system.

Primary test system: dual Intel Ice Lake-SP sockets (152 logical threads, 251 GiB DDR4-3200), 2 x RTX 3090 24 GiB with NVLink P2P, and a dual-socket Ice Lake-SP worker connected through ConnectX-5 100 GbE/RoCEv2. The main tuning targets are DeepSeek-V4 / DSV4-Flash and GLM-5.2.

## Verified results

Each row below is an independent benchmark scope. Percentages from different rows must not be multiplied together. Decode results are explicitly labeled `raw/no-DSpark` or `DSpark speculative`; PP is prefill and contains no draft-acceptance gain.

| Track | Matched baseline | Current result | Change |
|---|---:|---:|---:|
| DSV4 pp2048, quality-equivalent MXFP4 -> E4A | 312.48 tok/s | **362.92 tok/s** | **+16.1%** |
| DSV4 145.26 GiB full-model CPU_REPACK load + smoke | 163.48 s | **98.66 s** | **-39.65% wall** |
| DSV4 DSpark speculative decode, n2/p0 | 23.9 tok/s | **30.1 tok/s** | **+26%** |
| DSV4 E4A raw/no-DSpark decode, strict hot experts, 12-run A/B | 25.02 tok/s | **30.23 tok/s** | **+20.85%** |
| DSV4 16K GPU MoE prefill | 213 tok/s | **334.5 tok/s** | **+57%** |
| DSV4 single-slot raw/no-DSpark pure EP, 2 -> 4 NUMA workers | 22.1 tok/s | **25.2 tok/s** | **+14.03%** |
| DSV4 single-slot raw/no-DSpark, four-worker strict remote-only -> GPU-hot + CPU-remote | 25.25 tok/s | **28.85 tok/s** | **+14.26%** |
| GLM-5.2 pp512, true EP on 2 -> 4 NUMA workers | 24.13 tok/s | **40.59 tok/s** | **+68.2%** |

![Milestones across three performance tracks](docs/img/evolution-staircase.png)

## Three-engine control plane

PeoplesLLM decomposes one heterogeneous execution plan into three logical engines with explicit boundaries:

- **TAE (Topology-Aware Engine)** decides where computation and physical data live. It models cores/caches, CCDs or dies, NUMA/UPI, GPUs/PCIe, and remote workers; selects layer/expert/tensor granularity, mirror/split/owner placement, and device-native weight layouts; and warns about costly paths such as a GPU behind the PCH.
- **UPE (Unified Precision Engine)** decides whether replicas still implement the same model semantics. It manages canonical tensors and data epochs, synchronization, activation quantization, rounding/accumulation, nonlinearities, and slot-fold contracts, with joint shadow, PPL, hash, and DSpark accepted/drafted gates.
- **DME (Dynamic Matrix Engine)** decides how to execute the current phase. For TG, PP, and speculative-verification shapes, it selects GEMV/GEMM, tiles, batch/ubatch, fusion/pipelines, and AVX512-VNNI, future AMX, or CUDA kernels.

The decision rule is: **TAE constructs the physically feasible set, UPE removes numerically or semantically invalid paths, and DME minimizes the current phase's critical path over their intersection**. E4A therefore spans all three: TAE chooses the device layout, UPE validates the logical values and version, and DME selects the NR16/VNNI, AMX, or GPU consumer kernel.

## Architecture evolution

### 1. NUMA collapse: start with measurement

Naive interleaving on a dual-socket machine sends a large fraction of weight reads over UPI while every thread waits for the slowest participant at a barrier. Measured streaming bandwidth is **122 GB/s with interleaving versus 313.5 GB/s with local pages and local thread binding**, a 2.57 x gap. MoE decode consists of short, low-arithmetic-intensity bursts, so adding threads cannot solve this placement problem.

### 2. Mirror: eliminate remote reads first

`--numa mirror` keeps one complete weight replica per socket and binds pages and threads to the same node. It removes cross-UPI expert traffic and reaches 30.35 tok/s on DSV4 raw/no-DSpark tg512, about 8%-13% above the same-machine upstream distribute result of 26.9-28.0 tok/s. The cost is 2 x expert-weight memory.

### 3. NUMA EP: half the memory, but static ownership is too coarse

The next design stores each expert on only one node. Expert memory is halved and pp512 roughly matches mirror (343-348 versus 338-347 tok/s). At batch=1, however, one expert is one indivisible GEMV. Statically placing eight active experts into two nodes leaves a long-tail bubble and reaches only 23.9-24.2 tok/s on raw/no-DSpark tg512. This stage established the central constraint: **placement must preserve fine-grained parallelism and scheduling freedom.**

### 4. NUMA row-window TP + DME

The final layout alternates 128-row windows of every expert plane across the two nodes; execution dynamically claims work in 64-row quanta. Both sockets stream every expert, each output row has one writer, and no cross-node reduction is needed. The environment variable remains `GGML_NUMA_EP=1`, but the execution semantics are expert-internal row-window tensor parallelism.

![NUMA row-window TP on/off](docs/img/numa-tp-onoff.png)

| Workload | Off | On | Change |
|---|---:|---:|---:|
| hybrid raw/no-DSpark tg512 | 16.59 | **25.01** | **+51%** |
| hybrid pp2048 | 267.05 | **298.98** | +12% |
| pure-CPU raw/no-DSpark tg128 | 7.23 | **11.65** | **+61%** |
| pure-CPU pp512 | 101.41 | **107.91** | +6.4% |

On top of row-window TP, DME (Dynamic Matrix Engine) handles irregular MoE shapes inside each node:

- the claim quantum grows from 16 to 64 rows, raising microbenchmark bandwidth from 145 to 165-179 GB/s;
- `nrows` and batch shape select GEMV versus GEMM, while 2-8 UDNL tail rows for the same expert are batched;
- correcting a mismatch between core binding and row ownership cuts `MUL_MAT_ID` from 155.7 to 88.4 us/call;
- a two-level NUMA barrier and repeat-aware scheduling reduce tail waits.

### 5. Hybrid GPU execution

In the production-style hybrid path, the GPU runs attention, dense layers, the router, and KV; the CPU runs routed experts. Three complementary mechanisms target different phases:

- **Hot-expert residency**: the top 24 experts per layer cover about 57.6% of selections; compact MXFP4 weights for all 43 layers use 12.85 GiB. The GPU hot branch and CPU cold branch fork and join inside the graph. The GPU returns per-router-slot values, the CPU restores the baseline slot 0..5 left fold, and AVX512 vectorizes only across hidden rows. Twelve raw/no-DSpark tg512 runs improve from 25.0167 to 30.2333 tok/s (+20.85%); PP2048 is 361.47 tok/s and five-chunk PPL improves from 2.7758 to 2.7548.
- **NVLink P2P TP**: capture-safe P2P allreduce and asynchronous D2H readback improve the raw/no-DSpark TP path itself from 13.6 to 20.3 tok/s (+49%) and reduce `cudaStreamSynchronize` from about 690 to 150 calls per token. This is an internal TP-path A/B, not a comparison against the best layer-placement path.
- **Streaming MoE prefill**: expert weights remain in pinned host memory. Long prompts upload them once per layer, reuse them across tiles, and overlap H2D with dual-GPU expert-axis EP and three-slot prefetch.

![GPU streaming MoE prefill](docs/img/gpu-prefill-streaming.png)

| 16K prompt | tok/s |
|---|---:|
| chunked token-major | 213 |
| layer-major + pipe + device-HC | 248.5 |
| + dual-GPU EP + `PREFETCH=2` | **334.5** |

### 6. Multi-machine EP: several NUMA workers, one slot

`tools/epd/` transfers KB-scale activations and results, not expert weights during a request. Every NUMA node on the master and worker machine can be an independent worker. In strict pure EP mode (`SCHED_KLOCAL=0`), the master does not need a complete routed-expert replica. The runtime builds a holder map with hot replicas from real router frequencies and dispatches according to in-flight work.

![Multi-machine expert parallelism](docs/img/remote-ep.png)

| Test | Before | After | Change |
|---|---:|---:|---:|
| GLM-5.2 MoE pp512, 2 -> 4 workers | 24.13 | **40.59** | **1.682 x** |
| DSV4 MXFP4 raw/no-DSpark TG512, 2 -> 4 workers | 22.1 | **25.2** | **1.140 x** |
| DSV4 raw/no-DSpark TG512, four-worker strict remote-only -> GPU-hot + CPU-remote | 25.25 | **28.85** | **+14.26%** |
| DSV4 16K PP, UB64 -> UB256 | 235.37 | **269.36** | +14.4% |
| 64 B RPC RTT, TCP -> RoCEv2 | 42-74 us | **10-13 us** | about 4-6 x lower |

GLM dense layers and attention remain on the GPU; the first row scales only CPU MoE. Aggregate raw/no-DSpark decode time for 75 MoE layers falls from 86.71 to 69.58 ms/token (1.246 x), with byte-identical 128-token output. A separate DSV4 four-worker pure-EP run explicitly using DSpark speculative decode (NMAX=3) reaches 37.921 tok/s on average and 38.423 tok/s at peak, with the same output SHA256 as its matched single-machine reference; it is not mixed with the raw 25.2 tok/s row above.

The 2 -> 4 DSV4 row is a matched single run with one slot, one master, one model, and one command. Two local workers own 128 experts each; four cross-machine workers own 64 each, while `KLOCAL=0` makes the master skip all routed-expert weights. Generated-response SHA256 is identical across the two topologies. The hosts have different core counts and network latency, so this system result is not an ideal 2 x scaling claim.

GPU-hot + CPU-remote-EP has now passed one narrowly scoped end-to-end configuration: one slot, raw/no-DSpark, four-worker modulo strict cover, synchronous REQ4, and no MAX_EFFORT. Eight TG512 runs in ABBA+BAAB order are A (remote-only) `[24.9, 25.2, 25.5, 25.4]` and B (K24 hot + remote cold) `[29.6, 28.3, 28.6, 28.9]` tok/s, for 25.25 -> **28.85 tok/s (+14.26%)**. Same-command b1/ub1 five-chunk PPL is 2.7647 -> **2.7412 (-0.85%)**. All four A hashes agree and all four B hashes agree, but A differs from B, so this is not a bit-exact claim across CPU and GPU kernels. Every B run has 43/43 hot-fork and 43/43 remote-bridge markers.

An eight-token verbose diagnostic checks structure only and is not a performance sample. Across 301 one-token MoE calls, CPU assignments fall from 1,806 to 916 (-49.28%), endpoint totals are 239/220/213/244, all-four-worker fanout falls from 117 to 18 calls, mean remote wait falls from 0.444 to 0.310 ms, and mixed merge averages 0.0296 ms. These raw/no-DSpark percentages must not be multiplied by independent path gains.

The DSpark bridge now supports one to four target tokens with `[token,slot]` REQ4 masks and strict merges. K24 is the only tested hot count that passes the quality gate: five-chunk PPL changes from 2.7647 to 2.7412. Warm-server performance is only 35.251 -> 35.950 tok/s (+1.98%) at NMAX=3. At NMAX=2, six B runs across two independent server processes average 36.353 tok/s versus A=34.645, a **4.928%** gain with 2.466% CV. This misses the frozen 5% gate by 0.072 percentage points, so the path remains default-off, does not replace the production default, and is not claimed to reach 40 tok/s.

A stricter UPE ablation applies CPU-Q8_0-style RNE code generation only to `hot_expert.*`. It reduces real activation code mismatches from 33/10,144 to zero while scale mismatches remain zero; with identical codes, ordinary MXFP4 operators still show about 2.29e-7–4.80e-7 relative error from CUDA warp reduction. In an NMAX=2 warm-server A-B-B-A, the ordinary path is 35.0614±0.2720 and the CPU-Q8 candidate is 37.9573±0.4868 tok/s (+8.2595%); acceptance changes from 266/490 to 273/474, with stable hashes inside each path but different trajectories. Candidate five-chunk PPL is nevertheless 2.7855, 1.62% worse than strict-hot 2.7412 and 0.75% worse than no-hot 2.7647, beyond the 0.3% gate, while absolute speed remains below 40. `GGML_CUDA_HOT_MXFP4_CPU_Q8` therefore stays default-off and is rejected for production. Code agreement and improved acceptance do not replace a non-speculative quality gate.

### 7. Unified Precision Engine: the numerical and data control plane of hybrid EP

CPU AVX512, CUDA MMQ, and remote workers can produce different expert vectors from the same logical weights because activation Q8 quantization, rounding, scale recovery, accumulation trees, clamp/GLU, and FMA policies differ. Per-path determinism does not imply cross-device equivalence. In DSpark, a small perturbation near a logit tie can flip the target token, reduce draft acceptance, and change the subsequent trajectory. PeoplesLLM therefore treats the Unified Precision Engine (UPE) as part of hybrid EP rather than a post-hoc correctness test. UPE also manages which logical tensor and data epoch each replica represents and when a synchronized version is semantically published; it is not merely a floating-point tolerance test.

The evidence has three distinct levels. If only accepted/drafted changes while the final target tokens stay fixed, the result establishes a speculative-efficiency effect. If pure CPU and GPU expert offload are each deterministic under the same greedy configuration but generate different target tokens or response text, they instantiate different effective models and can affect generation quality. Only paired non-speculative PPL, task accuracy, and behavioral evaluation can determine the direction and magnitude of a quality change. DSpark is a sensitive error amplifier, not a standalone quality metric.

The implemented pieces include per-slot return values, a strict slot 0-to-5 left fold, CPU/GPU shadow execution, and joint PPL, response-hash, and accepted/drafted gates. EPD CAP can now negotiate a precision contract containing activation/dot/FFN-schema/per-slot-merge IDs, a model-schema hash, data epoch, and total contract hash. `GGML_REMOTE_EP_UPE_STRICT=1` rejects missing, cross-worker-different, or reconnect-changed contracts and requires the master and workers to share a non-empty `GGML_EP_DATA_EPOCH`. Besides CUDA/non-CUDA builds and unit tests, this has passed a real DSV4 four-worker run: two e64 workers on each host report identical contract/schema/epoch values, and strict REQ4 completes a 32-token smoke at PP 52.4/TG 24.7 tok/s. A deliberately wrong epoch is rejected at the first CAP before expert execution. The short speed sample is a protocol/functional gate, not a TG512 headline. K16 provides a large speed gain but worsens PPL from 2.7647 to 2.8066; K24 is the first tested count that passes quality. Shadow results also show that the largest local-error experts are not necessarily the quality cause. Speculative verification still needs a stricter produced-code/scale and acceptance profile than ordinary raw decode.

A new regression test reads the actual CUDA Q8\_1 temporary blocks rather than inferring them from final expert vectors. For a smooth input, all 4,096 codes and FP16 scales agree with CPU Q8\_0. On a half-step input, 1,876 CPU/GPU codes differ while every scale still agrees; the actual GPU codes also differ from a host emulation of the CUDA formula in 470 positions. Verify-strict must therefore compare produced device code/scale values, not just formula names.

On a full replay of 15,675,392 real DSV4 hidden values, only 34 CPU/GPU codes differ (2.17 ppm), every scale agrees, and the largest mismatching distance to a half step is 1.5259e-5. A default-off boundary fallback was implemented from this evidence. In a 64-token raw smoke, the low threshold reroutes 37 layer-tokens/112 hot slots, restores the CPU-remote body hash, and reaches 27.2 tok/s versus CPU 25.1 and ordinary GPU-hot 28.5. With DSpark NMAX=2, the nominal-only policy reaches 34.8 tok/s and 67/120 accepted/drafted, between CPU 31.2 and 66/122 and ordinary GPU-hot 37.9 and 66/122. One extra accepted token and a non-CPU body hash are not enough evidence to justify the performance tax, so the policy remains default-off and is not a production or general speed claim.

A second default-off phase selector, `GGML_HOT_EXPERT_UPE_VERIFY_CPU=1`, skips GPU-hot submission entirely for n_tokens>1 to remove duplicate fallback work. In the same DSpark test, all 2,895 multi-token layer calls use CPU EP and reach 30.7 tok/s with 66/122 and a body hash identical to CPU remote-only. It restores the contract but provides no net speedup. This localizes the remaining trajectory difference to multi-token GPU expert execution; the next target is a CPU-equivalent CUDA Q8/dot/accumulation/clamp-SwiGLU contract, not bypassing the GPU.

A weight-value audit then isolates seven OCP-reserved `E8M0=0xff` blocks and finite exponents 249/250 in layer-21 experts 202/205. Native CUDA conversion produced NaNs while CPU_REPACK historically used a half-scale of $2^{127}$; an edge test initially produced 1,536/1,536 non-finite CUDA outputs. The current path preserves CPU semantics and explicitly uses the same half-scale in CUDA MXFP4 MMVQ. The 0xff-only, e250-only, and combined edge cases are now bit-identical across CPU/GPU. A global `0xff -> 0` interpretation was rejected and reverted because pure-CPU PPL worsened from 2.7647 to 2.7801 (+0.56%).

Real extreme blocks remain accumulation-order sensitive, so `GGML_HOT_EXPERT_EXCLUDE=21:202,21:205` keeps only those experts in the CPU/cold domain and replaces them with the next layer-21 hot candidates. A single CLI run reaches 37.2 tok/s and 66/120=55.0% acceptance. Paired 10-chunk PPL is 2.5843 versus CPU 2.5853, a 0.039% improvement. However, in an ABCABC three-prompt warm-server run, mean TG falls from ordinary-hot 34.124 to 33.019 tok/s (-3.24%); two prompt pairs retain the same content/acceptance, while one changes content and drops from 61/130 to 56/140. Repeated hashes are stable within each placement, so this is a deterministic placement effect rather than noise. The candidate does not improve the cross-prompt Pareto frontier and remains default-off as UPE causal/negative evidence.

The hot-scoped CPU-Q8 result is a stronger counterexample: it eliminates produced activation-code differences and improves both warm-server speed and accepted tokens, yet PPL regresses beyond the quality gate. UPE therefore cannot stop at a code/scale contract; partial accumulation, epilogue, complete expert output, and non-speculative quality remain independent publication conditions.

## Kernel evolution

### AVX512 repack: route every MoE shape into a batch kernel

The early audit found that several `mul_mat_id` branches still called `vec_dot` one row at a time, so large batches never reached GEMM. Buffer-type ordering in CUDA builds could also place CPU weights in a pinned buffer without repack support. The fork fills in 8 x 8 repack traits, load-time conversion, GEMV/GEMM dispatch, and the EPD worker path for Q2_K-Q6_K, Q8_0, MXFP4, and IQ formats.

### arec panel-stationary: stream weights from DRAM once

An arec (activation record) precomputes the Q8 activation and scale once. Weight panels then remain in L1/L2 and are reused across eight-row tiles. End-to-end pp2048 improves as follows:

- UDNL_W4: 263.33 -> **366.58 tok/s (+39%)**;
- corrected UDNL_MX, regenerated after the imatrix-contract fix, reaches **418.95±3.50 tok/s**.

### Fusion and asynchronous boundaries

- the five-kernel DSV4 router chain becomes one single-warp kernel, and small-row indexer top-k becomes one radix kernel; GPU busy time falls from 18.76 to 18.23 ms/token and raw/no-DSpark tg rises from 25.15 to 25.91;
- pinned staging plus event draining cuts CPU input readback from 13 to 4.6 ms/token;
- MXFP4 repack GEMV software prefetch adds 2.6% raw/no-DSpark TG;
- 64-row claims and UDNL tail batching improve DSpark speculative n2/p0 from 26.50 to 30.10 tok/s without changing acceptance or output.

### Generic loading-time stream baking

The on-disk model remains a compatible GGUF of portable row blocks. With `GGML_STREAM_BAKE=1`, non-mmap CPU_REPACK loading uses two bounded `pread` staging buffers and writes the final CPU panels while reading. The current traits cover Q3_K/Q4_K, IQ2_XXS/IQ2_XS/IQ3_XXS, MXFP4, E4A, and Q2_K when its repack trait is enabled; no requantization occurs. For a real 1.14085 GB MXFP4 tensor, full and streamed loading both produce FNV-1a64 `84d0d08b79287973`, while wall time falls from 3.23 to 2.12 s and peak RSS from 2,420,908 to 1,307,104 KiB. An 822.08 MB IQ3_XXS tensor is also hash-identical and falls from 4.37 to 3.11 s.

A 145.26 GiB full-model load plus `pp1` smoke falls from 163.48 to 98.66 s (-39.65%). With NUMA EP and three PP256/TG128 repetitions, total wall falls from 256.83 to 177.55 s (-30.87%); PP means are 229.87 and 230.14, while TG is 27.14±0.13 versus 26.75±0.41. The TG difference is below 2σ and the final repacked bytes are identical, so the supported conclusion is **faster loading with no detectable inference-speed change**. A 1/4/16/64 MiB sweep gives 4 MiB only about a 1% local advantage, so the lower-staging 1 MiB default remains; the feature is still opt-in.

The GPU path does not reuse the CPU x8 layout. A zero-overhead MXFP4 `E8M0-plane + code-plane` candidate improved an isolated two-token projection by about 4%–5%, but the complete K24/six-slot hot FFN remained approximately 1.00× for one, two, and four tokens. The production branch was therefore removed. Future GPU-native formats need cross-token/slot tile reuse, operator fusion, or a native tensor-core datatype rather than a local AoS-to-SoA permutation.

A subsequent multi-token gate/up + DSV4-clamp + GLU fusion improves the complete hot-FFN kernel by about 6.3%/4.3%/2.6% for one/two/four tokens, with bit-identical fusion-on/off output. However, a K24, NMAX=2, four-worker remote-EP 512-token ABBA-BAAB changes 39.675 to 39.225 tok/s (-1.13%, fusion-on CV 3.21%). The GPU saving is hidden behind the remote-CPU critical path, so `GGML_CUDA_MOE_CLAMPED_FUSION` remains default-off and is not presented as an end-to-end speedup.

## Format evolution

### The “why is Q2_K slower than expected?” starting point

An initial 90.9 GiB model named `Q2_K` reached only 223.60 pp2048 and 22.87 raw/no-DSpark tg512. Its TG was below the larger Q3_R at 25.30 tok/s, although its PP was above the not-yet-optimized Q3_R GEMM path at 174.60 tok/s. GGUF metadata then revealed that it was actually IQ2_XXS at 2.0625 bpw, not Q2_K. The apparent “Q2_K < Q3_K” result therefore separated into a labeling problem and a kernel-efficiency problem: codebook lookup, scale expansion, dequantization-chain length, and access to batch GEMM can dominate nominal bpw.

The format work therefore moves from “fewest bits” to storage layouts that follow the kernel access order:

- **UDNL_W4** uses fixed-size blocks whose codebook output feeds AVX512 VNNI directly;
- **UDNL_MX** uses an importance matrix to allocate W2/W3/W4 blocks and reuses the same arec panel kernel;
- **E4A** preserves MXFP4 values bit-exactly and aligns row-block nibble pairing with the NR16 kernel. Loading performs arithmetic-free byte-only panelization, with no decoding or requantization.

![Full MoE format matrix by model size and PP/TG speed](docs/img/quant-formats.png)

The figure and table use one 2026-08-24 `performance` recipe for 17 format points. Sixteen original matrix points share one binary; corrected UDNL_MX was rerun after the fix with the same hardware, arguments, and recipe, replacing its old row. The fixed arguments are `-ngl 99 -ncmoe 99 -fa 1 -dev CUDA0 -sm layer -t 72 --numa distribute -b 4096 -ub 1024 --load-mode none -p 2048 -n 512 -r 3`; all 152/152 CPU governors and EPP settings were `performance`, turbo was enabled, and `numa_balancing=1` was recorded. These are raw single-slot results with neither DSpark nor hot experts. Standard UD formats are ordinary circles, MXFP4 is the anchor, the in-house UDNL_W4/UDNL_MX/E4A formats are large stars, and red hollow diamonds flag points that are smaller yet at least 3% slower than a larger format. Trend lines are fitted only to standard UD points plus the MXFP4 anchor; the three in-house formats are excluded from the fit.

UDNL_MX was regenerated from the verified source and rerun with the same `performance` recipe, replacing the old tainted row in the chart; its PPL uses the historical 20×512 WikiText-2 scope. Other PPL values are reused from prior measurements of the same weights because CPU frequency does not affect PPL. The PP/TG value for `UD-IQ4_XS`, 215.93/21.10, is weighted by repeat count from the original three runs at 219.09/21.62 and five supplemental runs at 214.03/20.78. The current loader identifies the routed MoE tensors in `UD-Q4_K_XL` as MXFP4, so that row is not a pure-Q4 expert endpoint. See [BENCHMARKS](docs/BENCHMARKS.md) for all 17 points and the historical scope.

| Format | Size | pp2048 | tg512 raw/no-DSpark | WikiText-2 PPL | Role |
|---|---:|---:|---:|---:|---|
| MXFP4 | 145.26 GiB | 312.48 | 26.87 | 3.5830 | quality and speed anchor |
| **UDNL_W4 + arec** | 146.36 GiB | 370.87 | 25.10 | 3.7997 | fixed 4-bit compute-oriented layout |
| **UDNL_MX + arec (corrected)** | **116.13 GiB** | **418.95±3.50** | **26.90±0.25** | 4.6047 | capacity/PP research point; 14.6% worse quality than Q3_K_XL |
| **E4A** | 145.26 GiB | **362.92** | 24.83 | 3.5830 | bit-exact MXFP4 values |
| UD-Q3_K_M | 119.28 GiB | 207.20 | 25.49 | 4.0242 | standard UD reference |
| UD-Q3_K_XL | 119.40 GiB | 207.69 | 25.34 | 4.0189 | standard UD reference |

Corrected UDNL_MX is about 2.7% smaller than Q3_K_XL while delivering about 2.0× PP and +6.2% raw/no-DSpark TG, but its PPL is 14.6% worse. It is therefore **not recommended as a Q3_K_XL replacement**. E4A strict-hot remains the quality-gated path, while UDNL_W4 is the fixed 4-bit kernel research line.

Q2/Q3/Q4 v1 was rejected when anomalous MXFP4 source data in the `blk.21` gate/up pair made a Q3 fp16 scale become `inf`. V2 preserves that complete atomic pair as MXFP4 and was materialized at **108.816 GiB** with `Q2_K/Q3_K/Q4_K/MXFP4 = 61/56/10/2`; all 129 tensor types, the file size, and SHA passed. Its matched 20-chunk PPL is nevertheless **4.7379**, **17.89%** worse than the matched UD-Q3_K_XL value of 4.0189. The quality gate rejected it, so PP/TG was intentionally not run and proxy improvements are not presented as a usable model. The format track therefore pivots to dynamic stream baking of established high-quality GGUFs.

A K24 strict-hot raw/no-DSpark pilot on corrected UDNL_MX raises TG from 26.9 to 31.3 tok/s (+16.36%) and improves same-command five-chunk PPL from 3.5614 to 3.5218. However, the cold GGUF plus 12.8496 GiB of hot weights total about **128.978 GiB**, 8.02% larger than Q3_K_XL, and no same-command 20-chunk Q3 comparison exists. This is a speed/negative-quality ablation, not a same-quality headline or recommended configuration.

![Size, speed, and quality tradeoff of corrected UDNL_MX](docs/img/udnl-mx-tradeoff.png)

## Build and run

```bash
# CUDA hybrid build
cmake -B build-cuda -DGGML_CUDA=ON
cmake --build build-cuda -j

# CPU-only build
cmake -B build-cpu -DGGML_CUDA=OFF
cmake --build build-cpu -j

# DSV4 single-machine hybrid: GPU dense/attention/KV, dual-socket CPU routed experts
GGML_NUMA_EP=1 build-cuda/bin/llama-server \
  -m /path/to/model.gguf -ngl 99 -ncmoe 99 \
  -t 72 --threads-batch 72 --numa distribute -fa on -b 4096 -ub 1024
```

`tools/peoplesllm-run.sh` provides `dsv4-prod`, `dsv4-dual`, `glm-dual`, and `cpu-pure` profiles. See the [quick start](docs/QUICKSTART.md), [parameter reference](docs/PARAMETERS.md), and [EPD guide](tools/epd/README.md) for remote workers, RDMA/TCP fallback, expert maps, and all options. Thread and ubatch settings must be recalibrated for the target CPU, memory channels, VRAM, and prompt length.

## Measurement and correctness

- Performance A/B uses the same model, prompt, thread count, and offload setup. Important results use ABBA/reverse-order repeats to control the measured run-order effect of about 25%.
- Paths that preserve reduction order use fixed-seed greedy decode with byte-for-byte or SHA256 comparisons.
- Cross-device reductions can flip one token at near-tied logits because floating-point addition order changes. Those paths use run-to-run determinism and acceptance checks instead and are not described as bit-exact.
- `pp` and `tg` are llama.cpp benchmark metrics; they are not an SLA for arbitrary concurrency, context length, or hardware.

See [docs/BENCHMARKS.md](docs/BENCHMARKS.md) for full measurements and scopes, and [docs/CHANGES.md](docs/CHANGES.md) for the technical change list.

## Documentation

- [Quick start](docs/QUICKSTART.md): builds, single/dual-machine recipes, and common issues
- [Developer architecture](docs/ARCHITECTURE.md): module boundaries, TG/PP/remote-EP data flows, and validation gates
- [Parameter reference](docs/PARAMETERS.md): environment variables, CLI options, and profiles
- [Benchmark archive](docs/BENCHMARKS.md): detailed A/B data and charts
- [Change list](docs/CHANGES.md): NUMA, kernel, GPU, and EP changes
- [Dual-machine EP](tools/epd/README.md): workers, transport, and deployment

## License

This project is a fork of [llama.cpp](https://github.com/ggml-org/llama.cpp). The original project is copyrighted by ggml-org and llama.cpp contributors and released under the MIT license. Changes in this fork use the same license and preserve the original notices and [LICENSE](LICENSE).
