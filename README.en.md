# PeoplesLLM

> **[中文版 README →](README.md)**

**Run giant MoE models locally on cheap hardware.** A specialized llama.cpp fork focused on 200B – 3T-parameter-class MoE inference on dual-socket or more CPUs + consumer GPUs.

## Benchmarks

DeepSeek-V4 284B (Q3_K quant), dual Xeon 8360Y (Ice Lake) + 2× RTX 3090:

| Implementation | TG (t/s) | PP (t/s) |
|---|---|---|
| Upstream llama.cpp (b10173) | 5.85 | 23.6 |
| **PeoplesLLM (NUMA mirror)** | **~33** | — |
| **PeoplesLLM (NUMA-EP)** | **28.5** | **310** |

GLM-5.2 745B (UD-Q2_K quant): TG 12.0 t/s (EP + GPU expert offload).

### CPU kernel microbenchmark (AVX512/VNNI/VBMI 8×8 repack vs upstream legacy vec_dot)

Single thread, nc=2048 k=4096, measured on Ice Lake 8360Y (reproduce: `tests/test-repack-kernels --perf`):

| Format | gemv (TG, 1 token) | gemm nr=4 (small batch) | gemm nr=16 (PP batch) |
|---|---|---|---|
| Q2_K | 0.98× | 0.90× | 1.10× |
| Q3_K | 0.92× | 1.85× | 2.44× |
| Q4_0 | 1.68× | — | 3.03× |
| Q4_K | 1.10× | — | 3.21× |
| Q5_K | 1.02× | 3.01× | 3.99× |
| Q6_K | 0.64× | — | 2.63× |
| MXFP4 | 1.40× | — | 2.78× |
| IQ1_S | 1.17× | 3.28× | 3.00× |
| IQ1_M | 0.92× | 2.18× | 3.87× |

- gemv (TG) ≈1× is the physical memory-bandwidth ceiling for single-token decode, not a kernel issue; the gemm (PP) gains come from the 8×8 repack amortizing weight-read bandwidth
- "—": these formats use `gemm_min_nrows=16`, so small batches route to gemv and nr=4 gemm is never hit in production (the Q4_K/Q6_K 4-row tail blocks are scalar — the router deliberately avoids them)
- Known items: Q2_K gemm nr=4 is a slight loss (0.90×); Q6_K gemv 0.64× is a pre-existing upstream regression — both on the fix list

## What's inside

- **NUMA mirror**: duplicate non-expert weights + KV per socket, pin threads per node — zero UPI traffic
- **NUMA-EP**: single-copy expert placement across sockets (mbind policy-level), local-first `mul_mat_id` — halves expert memory, makes 745B-class models fit
- **AVX512/VNNI 8×8 repack kernels**: full coverage of Q2_K–Q6_K, Q8_0, MXFP4, IQ1_S/IQ1_M — 2.4–4× on PP batches (gemm nr≥16)
- **Fused ops**: DeepSeek-V4 hyper-connection CUDA kernel, fused MoE router, RMS_NORM absorption, GLM-DSA Lightning Indexer
- **MTP speculative decoding** (DeepSeek-V4)
- **Cross-machine EP (WIP)**: activation dispatch (KB-scale traffic) instead of weight transfer, InfiniBand EDR interconnect, scalable CPU MoE nodes — targeting 2.8T-class models

## Status

Early development. `main` branch = production-ready; cross-machine EP transport layer (`tools/epd`) is ready, master-side integration in progress.

**Full changelog and feature documentation: [docs/CHANGES.md](docs/CHANGES.md)** (Chinese) — NUMA architecture, CPU kernel format matrix, fused ops, environment variable reference.

Upstream tracking: based on llama.cpp `e8f19cc0a` (2026-07-16); `vendor` branch holds the base, merged regularly.

## License

MIT (same as upstream llama.cpp)
