# PeoplesLLM

> **[中文 README](README.md)**

**Run 200B-3T MoE models on local servers while making multiple CPU sockets, consumer GPUs, and remote memory contribute to one inference job.**

PeoplesLLM is a llama.cpp fork for heterogeneous local inference. It evolves along three connected tracks: a multi-NUMA/multi-device execution architecture, AVX512 and GPU kernels, and compute-oriented GGUF weight formats. The standalone `tools/epd/` runtime provides expert parallelism across machines.

## Scope

The main model's attention, dense layers, router, and KV cache can stay on the GPU while routed MoE experts run on a dual-socket CPU or remote NUMA workers. The goal is not to run one model replica or one slot per node: multiple NUMA nodes cooperate on **the expert work of one layer for one slot**.

The author chooses the architecture and hardware direction; implementation, testing, and tuning are AI-assisted. Every public performance claim below comes from a measured, matched comparison. This is still a research and engineering project, so production use requires output and capacity validation on the target system.

Primary test system: dual Intel Ice Lake-SP sockets (152 logical threads, 251 GiB DDR4-3200), 2 x RTX 3090 24 GiB with NVLink P2P, and a dual-socket Ice Lake-SP worker connected through ConnectX-5 100 GbE/RoCEv2. The main tuning targets are DeepSeek-V4 / DSV4-Flash and GLM-5.2.

## Verified results

Each row below is an independent benchmark scope. Percentages from different rows must not be multiplied together.

| Track | Matched baseline | Current result | Change |
|---|---:|---:|---:|
| DSV4 pp2048, MXFP4 -> UDNL_MX | 307.58 tok/s | **384.87 tok/s** | **+25.1%** |
| DSV4 pp2048, quality-equivalent MXFP4 -> E4A | 315.7 tok/s | **370.0 tok/s** | **+17.2%** |
| DSV4 DSpark decode, n2/p0 | 23.9 tok/s | **30.1 tok/s** | **+26%** |
| DSV4 raw decode, matched hot-expert A/B | 26.0 tok/s | **28.5 tok/s** | **+9.7%** |
| DSV4 16K GPU MoE prefill | 213 tok/s | **334.5 tok/s** | **+57%** |
| GLM-5.2 pp512, true EP on 2 -> 4 NUMA workers | 24.13 tok/s | **40.59 tok/s** | **+68.2%** |
| DSV4 GGUF size, MXFP4 -> UDNL_MX | 145.3 GiB | **116.1 GiB** | **-20.1%** |

![Milestones across three performance tracks](docs/img/evolution-staircase.png)

## Architecture evolution

### 1. NUMA collapse: start with measurement

Naive interleaving on a dual-socket machine sends a large fraction of weight reads over UPI while every thread waits for the slowest participant at a barrier. Measured streaming bandwidth is **122 GB/s with interleaving versus 313.5 GB/s with local pages and local thread binding**, a 2.57 x gap. MoE decode consists of short, low-arithmetic-intensity bursts, so adding threads cannot solve this placement problem.

### 2. Mirror: eliminate remote reads first

`--numa mirror` keeps one complete weight replica per socket and binds pages and threads to the same node. It removes cross-UPI expert traffic and reaches 30.35 tok/s on DSV4 tg512, about 8%-13% above the same-machine upstream distribute result of 26.9-28.0 tok/s. The cost is 2 x expert-weight memory.

### 3. NUMA EP: half the memory, but static ownership is too coarse

The next design stores each expert on only one node. Expert memory is halved and pp512 roughly matches mirror (343-348 versus 338-347 tok/s). At batch=1, however, one expert is one indivisible GEMV. Statically placing eight active experts into two nodes leaves a long-tail bubble and reaches only 23.9-24.2 tok/s on tg512. This stage established the central constraint: **placement must preserve fine-grained parallelism and scheduling freedom.**

### 4. NUMA row-window TP + DME

The final layout alternates 128-row windows of every expert plane across the two nodes; execution dynamically claims work in 64-row quanta. Both sockets stream every expert, each output row has one writer, and no cross-node reduction is needed. The environment variable remains `GGML_NUMA_EP=1`, but the execution semantics are expert-internal row-window tensor parallelism.

![NUMA row-window TP on/off](docs/img/numa-tp-onoff.png)

| Workload | Off | On | Change |
|---|---:|---:|---:|
| hybrid tg512 | 16.59 | **25.01** | **+51%** |
| hybrid pp2048 | 267.05 | **298.98** | +12% |
| pure-CPU tg128 | 7.23 | **11.65** | **+61%** |
| pure-CPU pp512 | 101.41 | **107.91** | +6.4% |

On top of row-window TP, DME (Dynamic Matrix Execution) handles irregular MoE shapes inside each node:

- the claim quantum grows from 16 to 64 rows, raising microbenchmark bandwidth from 145 to 165-179 GB/s;
- `nrows` and batch shape select GEMV versus GEMM, while 2-8 UDNL tail rows for the same expert are batched;
- correcting a mismatch between core binding and row ownership cuts `MUL_MAT_ID` from 155.7 to 88.4 us/call;
- a two-level NUMA barrier and repeat-aware scheduling reduce tail waits.

### 5. Hybrid GPU execution

In the production-style hybrid path, the GPU runs attention, dense layers, the router, and KV; the CPU runs routed experts. Three complementary mechanisms target different phases:

- **Hot-expert residency**: the corrected trace shows that the top 16 experts per layer cover 49.1% of selections; compact MXFP4 weights for all 43 layers use about 8.8 GiB. An in-graph GPU hot branch forks against the CPU cold branch and joins before the residual. Matched tg512 runs improve from 26.1/25.9 to 28.4/28.6 tok/s, or 9.7% on the means.
- **NVLink P2P TP**: capture-safe P2P allreduce and asynchronous D2H readback improve the TP path itself from 13.6 to 20.3 tok/s (+49%) and reduce `cudaStreamSynchronize` from about 690 to 150 calls per token. This is an internal TP-path A/B, not a comparison against the best layer-placement path.
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
| DSV4 16K PP, UB64 -> UB256 | 235.37 | **269.36** | +14.4% |
| 64 B RPC RTT, TCP -> RoCEv2 | 42-74 us | **10-13 us** | about 4-6 x lower |

GLM dense layers and attention remain on the GPU; the first row scales only CPU MoE. Aggregate decode time for 75 MoE layers falls from 86.71 to 69.58 ms/token (1.246 x), with byte-identical 128-token output. Fixed-acceptance DSV4 pure EP over four workers and RDMA reaches 37.921 tok/s on average and 38.423 tok/s at peak, with the same output SHA256 as the single-machine reference.

## Kernel evolution

### AVX512 repack: route every MoE shape into a batch kernel

The early audit found that several `mul_mat_id` branches still called `vec_dot` one row at a time, so large batches never reached GEMM. Buffer-type ordering in CUDA builds could also place CPU weights in a pinned buffer without repack support. The fork fills in 8 x 8 repack traits, load-time conversion, GEMV/GEMM dispatch, and the EPD worker path for Q2_K-Q6_K, Q8_0, MXFP4, and IQ formats.

### arec panel-stationary: stream weights from DRAM once

An arec (activation record) precomputes the Q8 activation and scale once. Weight panels then remain in L1/L2 and are reused across eight-row tiles. End-to-end pp2048 improves as follows:

- UDNL_W4: 263.33 -> **366.58 tok/s (+39%)**;
- UDNL_MX: 223.02 -> **384.87 tok/s (+73%)**.

### Fusion and asynchronous boundaries

- the five-kernel DSV4 router chain becomes one single-warp kernel, and small-row indexer top-k becomes one radix kernel; GPU busy time falls from 18.76 to 18.23 ms/token and tg rises from 25.15 to 25.91;
- pinned staging plus event draining cuts CPU input readback from 13 to 4.6 ms/token;
- MXFP4 repack GEMV software prefetch adds 2.6% TG;
- 64-row claims and UDNL tail batching improve DSpark n2/p0 from 26.50 to 30.10 tok/s without changing acceptance or output.

## Format evolution

### The “why is Q2_K slower than expected?” starting point

An initial 90.9 GiB model named `Q2_K` reached only 223.60 pp2048 and 22.87 tg512. Its TG was below the larger Q3_R at 25.30 tok/s, although its PP was above the not-yet-optimized Q3_R GEMM path at 174.60 tok/s. GGUF metadata then revealed that it was actually IQ2_XXS at 2.0625 bpw, not Q2_K. The apparent “Q2_K < Q3_K” result therefore separated into a labeling problem and a kernel-efficiency problem: codebook lookup, scale expansion, dequantization-chain length, and access to batch GEMM can dominate nominal bpw.

The format work therefore moves from “fewest bits” to storage layouts that follow the kernel access order:

- **UDNL_W4** uses fixed-size blocks whose codebook output feeds AVX512 VNNI directly;
- **UDNL_MX** uses an importance matrix to allocate W2/W3/W4 blocks and reuses the same arec panel kernel;
- **E4A** preserves MXFP4 values bit-exactly and aligns row-block nibble pairing with the NR16 kernel. Loading performs arithmetic-free byte-only panelization, with no decoding or requantization.

![Full MoE format matrix by model size and PP/TG speed](docs/img/quant-formats.png)

The chart connects 19 formats from the matched 2026-08-21 matrix in ascending size order. A red diamond marks a smaller format that is at least 3% slower than some larger format. Repeated reversals in both PP and TG show that this workload is not controlled by DRAM bytes alone: codebook expansion, scale handling, storage-to-kernel layout, and access to batch kernels also matter. The table below lists representative Pareto points after later kernel work and must not be mixed point-by-point with the earlier matrix.

| Format | Size | pp2048 | tg512 | WikiText-2 PPL | Role |
|---|---:|---:|---:|---:|---|
| MXFP4 | 145.3 GiB | 307.58 | 26.26 | 3.5830 | quality baseline |
| **UDNL_W4 + arec** | 146.4 GiB | 366.58 | 24.99 | 3.7997 | fixed 4-bit compute-oriented layout |
| **UDNL_MX + arec** | **116.1 GiB** | **384.87** | 26.38 | 4.6274 | smallest size and highest PP |
| **E4A** | about 147.2 GiB | **370.0** | 26.37 | approximately MXFP4 | bit-exact MXFP4 values; load about 22 -> 11 s |
| Q3_R (MoE) | 114.1 GiB | 174.60 | 25.30 | 4.7636 | small-format reference |
| IQ2_XXS (file named Q2_K) | 90.9 GiB | 223.60 | 22.87 | 6.6338 | dequantization-cost reference |

Choose UDNL_MX when capacity and PP matter most; choose E4A when MXFP4 values and quality must be preserved. The fixed format harness covers the main rows, while E4A quality was rechecked with a five-chunk PPL run; see [BENCHMARKS](docs/BENCHMARKS.md) for the exact scopes.

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
