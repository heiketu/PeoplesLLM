# PeoplesLLM

> **[中文版 README →](README.md)**

**Run giant MoE models locally on cheap hardware.** A specialized llama.cpp fork focused on 200B – 3T-class MoE inference on dual-socket or more CPUs + consumer GPUs.

## Benchmarks

DeepSeek-V4 284B (Q3_K, 94.6GB GGUF), dual Xeon 8360Y (Ice Lake) + 2× RTX 3090:

| Implementation | TG (t/s) | PP (t/s) |
|---|---|---|
| Upstream llama.cpp (b10173) | 5.85 | 23.6 |
| **PeoplesLLM (NUMA mirror)** | **~33** | — |
| **PeoplesLLM (NUMA-EP)** | **28.5** | **310** |

GLM-5.2 (UD-Q2_K, 236GB GGUF): TG 12.0 t/s (EP + GPU expert offload).

## What's inside

- **NUMA mirror**: duplicate non-expert weights + KV per socket, pin threads per node — zero UPI traffic
- **NUMA-EP**: single-copy expert placement across sockets (mbind policy-level), local-first `mul_mat_id` — halves expert memory, makes 236GB-class models fit
- **AVX512/VNNI 8×8 repack kernels**: Q2_K–Q6_K, Q8_0, MXFP4, IQ1_S/IQ1_M — 2–4× prompt-processing speedup
- **Fused ops**: DeepSeek-V4 hyper-connection CUDA kernel, fused MoE router, RMS_NORM absorption, GLM-DSA Lightning Indexer
- **MTP speculative decoding** (DeepSeek-V4)
- **Cross-machine EP (WIP)**: activation dispatch (KB-scale traffic) instead of weight transfer, InfiniBand EDR interconnect, scalable CPU MoE nodes — targeting 2.8T-class models

## Status

Early development. `main` branch = production-ready; cross-machine EP transport layer (`tools/epd`) is ready, master-side integration in progress.

**Full changelog and feature documentation: [docs/CHANGES.md](docs/CHANGES.md)** (Chinese) — NUMA architecture, CPU kernel format matrix, fused ops, environment variable reference.

Upstream tracking: based on llama.cpp `e8f19cc0a` (2026-07-16); `vendor` branch holds the base, merged regularly.

## License

MIT (same as upstream llama.cpp)
