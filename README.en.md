# PeoplesLLM

> **[中文 README →](README.md)**

**Run 200B–3T-parameter MoE models locally on cheap hardware — fast.**
A specialized llama.cpp fork for dual/multi-socket CPU servers plus consumer GPUs, with four self-developed optimization tracks: CPU compute kernels, the NUMA subsystem, GPU long-context prefill, and cross-machine expert parallelism.

![Headline speedups](docs/benchmarks/headline_speedups.png)

## About

The author comes from a hardware background; all design, coding, testing and tuning in this project were done by AI (Kimi K3 + Kimi Code CLI) — the author only picks the technical directions (EP, GEMM kernels, AVX512/VNNI/VBMI, NUMA, etc.). The code may not be pretty and may contain bugs; it is only guaranteed on the author's own rig (dual Xeon 8360Y + 2× RTX 3090). Main tuning targets: **DeepSeek-V4 / DSV4-Flash** and **GLM-5.2**. Issues welcome.

Test platform: dual Xeon 8360Y (Ice Lake, 152 threads, 251G RAM) + 2× RTX 3090 24G (NVLink) + a ConnectX-5 100G direct-linked slave (dual Ice Lake ES, 188G). Baseline = upstream llama.cpp, same machine, same methodology A/B.

## Features

### CPU inference

| Feature | When to use | Impact |
|---|---|---|
| **AVX512/VNNI/VBMI 8×8 repack kernels** (Q2_K–Q6_K, Q8_0, MXFP4, IQ1_S/IQ1_M/IQ2_XXS) | All CPU compute; biggest win on long-prompt batches | prefill gemm up to **4.9×** (micro-bench); GLM-5.2 end-to-end **PP 4.1×** |
| **IQ2_XXS AVX512 repack gemv/gemm kernel** (x8 layout) | CPU inference of IQ2_XXS-weight models | gemv micro-bench **17×**; full-model A/B still flat (bottleneck is the uncovered expert mul_mat_id) — model-level gains pending extension |
| **NUMA mirror** (`--numa mirror`) | Dual-socket servers with RAM to spare | Best TG (**+9%** vs upstream), zero cross-socket UPI traffic |
| **NUMA row-window EP** (`GGML_NUMA_EP=1`) | RAM-constrained, loading oversized models; also works inside EPD workers spanning NUMA nodes | **Half the expert memory**, TG matches mirror, single-writer dst rows with zero merge; ABAB-measured hybrid tg512 **+49%**, pure-CPU tg128 **+61%** (incl. load-degradation resistance), pp +6~12% |
| **Hierarchical barrier** (`GGML_NUMA_HIER_BARRIER=1`) | Multi-NUMA-node inference | +0.9% (noise-level but free) — recommended always-on |
| **repack gemv software prefetch** (`GGML_REPACK_GEMV_PREFETCH=1`) | repack-kernel decode | TG **+2.6%** — recommended always-on |
| **Fused ops** (hyper-connection HC, MoE router, RMS_NORM absorption, DSA Lightning Indexer) | DSV4 / GLM everywhere | Eliminates decomposed paths and intermediate activation traffic |
| **MTP speculative decoding** | DSV4 decode | Further TG gains (all numbers here are without MTP) |

### GPU inference (long context)

| Feature | When to use | Impact |
|---|---|---|
| **Layer-major long prefill** (`llama_decode_layer_major()`, layer weights resident in CUDA slots) | 4K–1M long prompts | 16K PP 161→**604 tok/s** (**752** with sparse compact opt-in), 4.7× |
| **Streaming MoE prefill + 3-slot dual-GPU prefetch** | Hybrid inference with expert weights in host RAM | Each layer's weights uploaded once, reused across tiles |
| **True dual-GPU same-layer expert-axis EP** (`GGML_CUDA_MOE_PP_EP`, opt-in) | Dual-GPU prefill | 2K PP **+63%** (277→452 tok/s), bit-identical output |
| **Batched top-k** (`GGML_CUDA_BATCHED_TOPK`) | DSA sparse-attention models, long context | top-k kernel **21.6×**, e2e PP +8.2% |
| **q1 32-head FA + raw-SWA decode ring** (ring on by default; `LLAMA_DSV4_COMPACT_DECODE_SWA=0` to disable) | Long-context decode | q1 decode graph width decoupled from prompt length: fixed TG64 8.89→**12.09** (+33~36%), TG512 +4~8%; automatic fallback for multi-slot |
| **q8 compact sparse FA** (on by default, q8_0 KV) | Long-context decode with q8 KV | q1 decode materializes only top-k selected rows: TG64 9.75→**10.69** (+10%) — fastest q1 decode path (beats f16 dense); f16 fused sparse stays opt-in (16K measured -12%) |
| **Multi-stream (small-q) sparse FA** (`LLAMA_DSV4_FUSED_INDEXED_FA=3/4`, `LLAMA_DSV4_Q8_SPARSE_FA=2`, opt-in) | Multi-slot concurrent decode | Correctness verified three-way byte-identical; parity at ≤16K (weight-bandwidth bound), payoff targeted at 256K+ context |
| **GPU streaming prefill** (plan A, server integration) | Long prompts in llama-server | Full-tile path PP 63→**127.6 tok/s** (2×); falls back to chunked via eligibility gate under compact ring SWA cache, `--swa-full` to enable |
| **GPU expert offload** (`-ot blk.N.ffn_*_exps=CUDA0/1`) | Spare VRAM | ~+1% TG per offloaded layer |

### Cross-machine distributed EP (`tools/epd/`)

| Feature | When to use | Impact |
|---|---|---|
| **Cross-machine expert parallelism** (activation dispatch, KB-scale traffic — not weight transfer) | Model doesn't fit in one machine's RAM | DSV4 two-machine matches single-machine speed with **26% less** master RAM |
| **RoCEv2 RDMA transport** (`GGML_REMOTE_EP_RDMA=1`, automatic TCP fallback) | IB/RoCE NICs; plain gigabit falls back seamlessly | RTT 42-74µs→**10-13µs**; after the large-frame fix, PP1020 33.4→**76.1 tok/s (2.3×)** |
| **MAX-EFFORT layer mirroring** (`GGML_REMOTE_EP_MIRROR=1`) | Spare master RAM, decode-heavy workloads | Remote segment leaves the critical path, TG **+9~11%** |
| **True four-NUMA, single-slot EP** (`SCHED_KLOCAL=0` + hot-expert replicas) | Two dual-socket machines compute one request together | GLM-5.2 PP512: 2 NUMA **24.13**→4 NUMA **40.59 tok/s (1.682×)**; byte-identical output across 75 layers |
| **EP planner** (`tools/epd/ep-plan.py`) | Pre-deployment layer-split decisions | Recommends splits from measured bandwidth/latency, calibration error ≤1.5% |

### Model support

DeepSeek-V4 / DSV4-Flash (incl. native MXFP4), GLM-5.2 (DSA), MiniMax-M3; plus a byte-level GGUF repair toolchain (quant-block corruption scan + patch — fixed 138 corrupted blocks in the official GLM-5.2 file that caused garbled output).

## Quick start

```bash
# CUDA build (single-machine hybrid inference)
cmake -B build-cuda -DGGML_CUDA=ON && cmake --build build-cuda -j

# Production recipe: DSV4-Flash mxfp4, dual-socket NUMA EP + GPU offload
GGML_NUMA_EP=1 GGML_NUMA_HIER_BARRIER=1 GGML_REPACK_GEMV_PREFETCH=1 \
  build-cuda/bin/llama-server -m model.gguf -ngl 99 -ncmoe 99 -t 72 \
  --numa distribute -fa 1 -b 4096 -ub 1024
```

Two-machine EP, pure-CPU and AVX2-only builds, and full deployment details: **[docs/QUICKSTART.md](docs/QUICKSTART.md)** (Chinese).

## Docs

- **[docs/QUICKSTART.md](docs/QUICKSTART.md)** — build, recommended single-/dual-machine configs, common pitfalls
- **[docs/PARAMETERS.md](docs/PARAMETERS.md)** — full env/CLI parameter manual and production recipes
- **[docs/BENCHMARKS.md](docs/BENCHMARKS.md)** — complete measured data and iteration history (same-machine A/B vs upstream)
- **[docs/CHANGES.md](docs/CHANGES.md)** — technical changelog (NUMA, CPU kernels, fused ops, distributed EP)
- **[tools/epd/README.md](tools/epd/README.md)** — two-machine EP quick start

## Status

Early development. Validate output hashes, slot contexts and VRAM headroom yourself before production use. Upstream sync target: llama.cpp `4df29be4f` (2026-08-15, 96 commits merged; AVX2/AVX512 dual-build regression passed (tg flat, pp -5.4% attributed to upstream changes, see BENCHMARKS)). Ongoing: GPU prefill throughput, CPU/GPU cooperative pipelining, unified weight representation.

## License & Copyright

This project is a fork of [llama.cpp](https://github.com/ggml-org/llama.cpp). **Copyright of the original project belongs to ggml-org and all llama.cpp contributors**, released under the MIT license. All changes in this fork are likewise released under the MIT license, preserving the original copyright notices and license text (see [LICENSE](LICENSE)).
