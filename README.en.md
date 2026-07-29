# PeoplesLLM

> **[中文版 README →](README.md)**

**Run giant MoE models locally on cheap hardware.** A specialized llama.cpp fork focused on 200B – 3T-parameter-class MoE inference on dual-socket or more CPUs + consumer GPUs.

## Background

The author comes from a hardware background — university courses covered only basic C, Python, and embedded assembly — and has long been unhappy with two things about running LLMs locally: **slow speed and a high barrier to entry**. This project was therefore tuned using Kimi K3 together with Kimi Code CLI.

The author only specified the technical concepts — EP (expert parallelism), GEMM kernels, instruction-set optimization (AVX512/VNNI/VBMI), NUMA optimization, and the like. Everything else — design, coding, testing, and tuning — was **done entirely by AI**. As a result, code readability may be poor and latent bugs may exist. The author takes responsibility only for running on their **own platform** (dual Xeon 8360Y + 2× RTX 3090).

Current tuning targets: **GLM-5.2** and **DeepSeek-V4**. Issues are welcome.

## Benchmarks

DeepSeek-V4 284B (Q3_K quant), dual Xeon 8360Y (Ice Lake) + 2× RTX 3090:

| Implementation | TG (t/s) | PP (t/s) |
|---|---|---|
| Upstream llama.cpp (b10173) | 5.85 | 23.6 |
| **PeoplesLLM (NUMA mirror)** | **~33** | — |
| **PeoplesLLM (NUMA-EP)** | **28.5** | **310** |

> Note: all DeepSeek-V4 numbers above are measured **without MTP speculative decoding**; TG goes higher with MTP enabled (MTP figures to be published separately).

GLM-5.2 745B (UD-Q2_K quant): TG 12.0 t/s (EP + GPU expert offload).

### CPU kernel full benchmark (AVX512/VNNI/VBMI 8×8 repack vs upstream legacy vec_dot)

Measured on Ice Lake 8360Y, shape nc=16384 k=8192 (exceeds L3 — the real DRAM-bandwidth view); values are speedups (reproduce: `tests/test-repack-kernels --perf [threads]`; * = x86 kernels new/completed in this fork):

**Single thread:**

| Format | gemv (TG) | gemm nr=4 | nr=8 | nr=16 | nr=32 |
|---|---|---|---|---|---|
| Q2_K | 0.98× | 0.91× | 0.89× | 1.13× | 1.03× |
| Q3_K* | 1.03× | 2.14× | 2.68× | 2.63× | 3.02× |
| Q4_0 | 1.21× | 2.82× | 2.75× | 3.46× | 3.67× |
| Q4_K | 1.26× | 0.07× | 0.08× | 4.85× | 4.87× |
| Q5_K* | 1.03× | 3.24× | 3.22× | 4.70× | 4.23× |
| Q6_K* | 0.88× | 0.01× | 0.01× | 4.70× | 4.85× |
| MXFP4 | 1.20× | 2.42× | 2.43× | 3.42× | 3.53× |
| Q8_0* | 1.16× | 3.69× | 3.69× | 3.65× | 3.76× |
| IQ1_S* | 1.20× | 3.00× | 3.05× | 3.64× | 3.67× |
| IQ1_M* | 0.91× | 2.22× | 2.27× | 3.59× | 3.81× |

**72 threads (both sockets, production config):**

| Format | gemv (TG) | gemm nr=4 | nr=8 | nr=16 | nr=32 |
|---|---|---|---|---|---|
| Q2_K | 1.32× | 1.31× | 0.59× | 1.09× | 0.95× |
| Q3_K* | 1.25× | 1.28× | 1.36× | 1.38× | 1.54× |
| Q4_0 | 1.12× | 1.93× | 1.99× | 2.31× | 3.04× |
| Q4_K | 1.45× | 0.09× | 0.14× | 1.72× | 2.01× |
| Q5_K* | 0.96× | 1.94× | 2.32× | 2.41× | 2.53× |
| Q6_K* | 1.03× | 0.04× | 0.03× | 2.12× | 2.96× |
| MXFP4 | 1.45× | 1.74× | 1.70× | 2.06× | 3.01× |
| Q8_0* | 1.26× | 1.62× | 1.52× | 2.09× | 2.79× |
| IQ1_S* | 1.23× | 2.11× | 2.05× | 2.49× | 3.15× |
| IQ1_M* | 1.20× | 1.90× | 1.73× | 2.71× | 3.11× |

- gemv (TG, single token) ≈1× is the physical memory-bandwidth ceiling; the gemm (PP) gains come from the 8×8 repack amortizing weight-read bandwidth
- Q4_K/Q6_K at nr≤8 use scalar tail blocks (0.01–0.14×) — the production router deliberately avoids them with `gemm_min_nrows=16`; all other formats have fully vectorized 4-row tails
- Known items: Q2_K shows no gain anywhere (weights already at ~2.6 bit — nothing left to amortize); Q6_K gemv 0.62× on cache-resident shapes is a pre-existing upstream regression (back to ~1× on large shapes/multithread)
- The cache-resident small shape (2048×4096) follows the same pattern: best nr=16 is 3.96× (IQ1_M), best gemv 1.39× (Q4_0); all 300 cells reproducible with the command above

### repack end-to-end comparison (DeepSeek-V4 284B, llama-bench A/B)

Identical config (NUMA-EP + mirror, 72 threads, fa=1, batch 4096/ubatch 1024); `--no-repack` is a switch this fork adds to llama-bench:

| Config | pp512 (t/s) | tg128 (t/s) |
|---|---|---|
| CPU-only · repack on (default) | **114.33** | 17.34 |
| CPU-only · repack off | 97.17 | **17.66** |
| GPU expert offload · repack on (production) | **151.12** | 29.04 |
| GPU expert offload · repack off | 151.09 | **29.10** |

- CPU-only: PP **+17.7%**, TG **-1.8%** (TG is bandwidth-bound; repack gemv is slightly slower than legacy, matching the microbenchmark)
- GPU offload: difference **<1%** — the bottleneck moves to GPU-side attention and expert compute, and the repack gain washes out as the CPU matmul share shrinks
- Takeaway: keep repack on (default) for CPU-only / long-prompt workloads; either way is fine with GPU offload; use `--no-repack` only if chasing peak TG

## What's inside

- **NUMA mirror**: duplicate non-expert weights + KV per socket, pin threads per node — zero UPI traffic
- **NUMA-EP**: single-copy expert placement across sockets (mbind policy-level), local-first `mul_mat_id` — halves expert memory, makes 745B-class models fit
- **AVX512/VNNI 8×8 repack kernels**: full coverage of Q2_K–Q6_K, Q8_0, MXFP4, IQ1_S/IQ1_M — up to 4.9× on PP batches (gemm nr≥16); the Q3_K/Q5_K/Q6_K/Q8_0 x86 kernels and the entire IQ1_S/IQ1_M stack (block layout + repack + kernels) are new in this fork
- **Fused ops**: DeepSeek-V4 hyper-connection CUDA kernel, fused MoE router, RMS_NORM absorption, GLM-DSA Lightning Indexer
- **MTP speculative decoding** (DeepSeek-V4)
- **Cross-machine EP (WIP)**: activation dispatch (KB-scale traffic) instead of weight transfer, InfiniBand EDR interconnect, scalable CPU MoE nodes — targeting 2.8T-class models

## Status

Early development. `main` branch = production-ready; cross-machine EP transport layer (`tools/epd`) is ready, master-side integration in progress.

**Full changelog and feature documentation: [docs/CHANGES.md](docs/CHANGES.md)** (Chinese) — NUMA architecture, CPU kernel format matrix, fused ops, environment variable reference.

Upstream tracking: based on llama.cpp `e8f19cc0a` (2026-07-16); `vendor` branch holds the base, merged regularly.

## License & Copyright

This project is a fork of [llama.cpp](https://github.com/ggml-org/llama.cpp). **The original project is copyrighted by ggml-org and the llama.cpp contributors** and released under the MIT License. All modifications in this fork are likewise released under the MIT License, with the original copyright and license notices retained (see [LICENSE](LICENSE)).
