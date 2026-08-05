# PeoplesLLM

> **[中文版 README →](README.md)**

**Run giant MoE models locally on cheap hardware.** A specialized llama.cpp fork focused on 200B – 3T-parameter-class MoE inference on dual-socket or more CPUs + consumer GPUs.

## Background

The author comes from a hardware background — university courses covered only basic C, Python, and embedded assembly — and has long been unhappy with two things about running LLMs locally: **slow speed and a high barrier to entry**. This project was therefore tuned using Kimi K3 together with Kimi Code CLI.

The author only specified the technical concepts — EP (expert parallelism), GEMM kernels, instruction-set optimization (AVX512/VNNI/VBMI), NUMA optimization, and the like. Everything else — design, coding, testing, and tuning — was **done entirely by AI**. As a result, code readability may be poor and latent bugs may exist. The author takes responsibility only for running on their **own platform** (dual Xeon 8360Y + 2× RTX 3090).

Current tuning targets: **GLM-5.2** and **DeepSeek-V4**. Issues are welcome.

## Benchmarks

DeepSeek-V4 284B (Q3_K quant), dual Xeon 8360Y (Ice Lake) + 2× RTX 3090 (14 expert layers on GPU, 72 threads, `--no-mmap`; re-measured same-config 2026-08-01):

| Implementation | tg512 (t/s) | pp512 (t/s) | MoE op wall |
|---|---|---|---|
| Upstream llama.cpp (NUMA distribute) | 26.9-28.0 | 150.5-161.3 | ~102.5-125µs |
| PeoplesLLM (isolate, single-socket ref) | 25.7-25.9 | 363.6-370.2 | 133.4µs |
| **PeoplesLLM (NUMA row-window EP, half memory)** | **30.34-30.48** | 318-333 | **88.4µs** |
| **PeoplesLLM (NUMA mirror)** | **30.35-30.44** | **344-349** | 80.3µs |

> Note: all DeepSeek-V4 numbers above are measured **without MTP speculative decoding**; TG goes higher with MTP enabled (MTP figures to be published separately).
> For PP-heavy or single-socket scenarios isolate is actually fastest (pp512 370); with ample RAM mirror wins TG; when memory-bound, row-window EP trades ~10% MoE overhead for half the expert memory.

GLM-5.2 745B (UD-Q2_K_MXFP4, single machine + 2× 3090, 9 expert layers on GPU; same-config A/B 2026-08-01):

| Implementation | tg512 (t/s) | pp512 (t/s) | pp1020 (t/s) |
|---|---|---|---|
| PeoplesLLM w/o IQ traits | 11.23-11.27 | 56.8 | 98.2-99.1 |
| **PeoplesLLM + IQ2_XS/IQ3_XXS traits** | **11.95 (+6%)** | **298-307** | **399-406 (4.1×)** |

### Row-window EP + thread-affinity root-cause fix (2026-08-01)

- **Row-window EP**: expert-parallel placement moved from whole-expert granularity to **row windows within each expert** (node n owns the n-th window of every expert, single mbind copy) — single-writer dst rows, zero merge, structurally kills the batch=1 tail imbalance that made whole-expert EP perform like a single socket.
- **Thread-affinity root-cause fix**: DISTRIBUTE pinning (`ith % n_nodes`, interleaved) disagreed with the EP compute side's window ownership (block mapping) — **exactly half the threads were pinned to the wrong node and read their "local" windows 100% over UPI** (155.7µs vs mirror's 80.3µs per MoE op). Fixed: 88.4µs (89% of the gap recovered); EP tg512 20.7-26.3 → **30.34-30.48, matching mirror and beating upstream distribute**; correctness: mirror==EP token-identical.
- The residual ~10% MoE overhead is attributed to the two-phase claim compute path (not placement); `GGML_NUMA_EP_STATIC/EP_CLAIM/EP_CHUNK` diagnostic switches documented in the params manual.

### IQ2_XS/IQ3_XXS repack kernels + mul_mat_id gemm dispatch (2026-08-01)

- GLM-5.2's dominant expert quants **IQ2_XS/IQ3_XXS got full 8×8 interleaved repack traits** (layouts + conversion + generic & AVX512 (VNNI+VBMI gated) gemv/gemm): micro-bench gemm **2.3-2.5×** vs vec_dot, native==generic bit-exact, test-backend-ops 1996/1996 PASS.
- **mul_mat_id gemm dispatch**: repacked MoE batch paths previously ran row-by-row gemv (Q2_K had the same gap); now 4-row tiles are gathered + quantized into interleaved tiles for gemm — GLM single-machine **PP1020 99→406 (4.1×), tg512 +6%**, EP and mirror token-identical.

### Long-context layer-major prefill + dual-GPU MoE EP (2026-08-04, DSV4-Flash 0731)

New experimental API `llama_decode_layer_major()`: layer-outer / token-tile-inner execution; the current layer's expert weights stay resident in private CUDA slots and are not re-uploaded for later tiles of the same layer — cutting "re-upload all model weights per ubatch" down to "one upload per layer". 16K prefill went from 161 tok/s (first version) to **604 tok/s** (exact dense baseline, bit-identical final logits fingerprint), and **752 tok/s** with sparse raw-KV compaction (opt-in):

![16K PP progression](docs/benchmarks/longctx_pp_progression.png)

- **True dual-GPU same-layer expert-axis EP** (`GGML_CUDA_MOE_PP_EP`, opt-in): experts of the same layer split across CUDA0/CUDA1 in parallel — 2K 277 → **452 tok/s (+63%)** vs the serial correct baseline, 581 tok/s @16K, bit-identical output.
- **Fully ordered K/V tile reuse**: FA keeps the original 128+128 KQ order while widening the shared-memory row stride to reuse the full tile — 16K PP +3.9%, identical logits fingerprint (a 256-merge variant that changed reduction grouping was rejected and reverted).
- **Batched top-k** (`GGML_CUDA_BATCHED_TOPK`): the DSA indexer's 16384×4096 k=512 top-k went from 65.5ms to **3.03ms (21.6×)**; kernel launches 1.39M → 19.5k; e2e PP +8.2%.
- **Sparse raw-KV compaction** (opt-in, query batch ≥256): compacts the valid raw-KV range to ≤512 rows plus the compressed top-k union — 16K PP 596→**752 tok/s**; floating-point reduction grouping differs, so it stays off by default and q1 decode falls back to dense FA.

The native MXFP4 DSV4-Flash build (155GB; 137GiB experts as a single CPU_REPACK copy on dual-socket NUMA EP; dense/attention/KV on two 3090s) after a CPU audit (three fixes: NUMA claim false sharing / row-window misalignment / large-chunk scratch overrun, plus MXFP4 scale SIMD and Q8 streaming interleave):

![MXFP4 Hybrid CPU audit and dual-GPU EP](docs/benchmarks/mxfp4_hybrid_cpu_audit.png)

Long-context decode degradation was root-caused to the physical dense scan of GPU attention/KV (41 layers of full-width concat), being fixed item by item:

![16K TG improvements](docs/benchmarks/longctx_tg_improvements.png)

- **q1 FA 32-head MMA instances**: DSV4's 64 Q heads share one 512-dim K/V head; 32-head is auto-selected at K≥1024 — 16K micro-bench 185.6→73.9µs, fixed TG64 **+17.0%**.
- **raw-SWA decode ring** (opt-in, in validation): after prefill only the newest 128 raw SWA cells are kept, bounding decode physical width to 256 — fixed TG512 11.30→**12.18 (+7.8%)**, CUDA graph warmup resets 130→48.

16K GPU-side time breakdown (Nsight, 604 tok/s path) — FA and weight H2D are the next battlegrounds:

![16K hotspot breakdown](docs/benchmarks/longctx_hotspots.png)

> Reproduction & methodology: `tests/test-layer-major.cpp` (`BENCH_GPU_STREAM` unified PP+TG in one load, `LLAMA_BENCH_FIXED_TG=1` fixed-route A/B); full tensor split and cross-tile dual-scheduler were both measured and rejected (PP -44% / CPU threadpool semantics), see `docs/CHANGES.md`.

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

### Two-machine expert-parallel (GLM-5.2 745B UD-Q2_K_MXFP4, full sweep 2026-07-31)

Topology: master (dual Xeon 8360Y-class / 251G / 2×3090) ←100G RoCEv2 direct link→ slave (dual 36-core Ice Lake ES / **188G, symmetric ~157 GB/s per node, ~315 GB/s total** — 1.8× the old 174 GB/s after the 2026-07-30 memory rework). The slave runs the `llama-epd` worker owning MoE layers 3-17 (15 layers ≈43.5G, `--no-mmap` resident); master keeps 52 local layers + 9 GPU-offloaded layers. Correctness: every configuration below produces gen48 output (temp 0 / seed 42) **byte-identical** to the single-machine baseline.

| Config | TG96 | TG512 | PP5 | PP63 | PP254 | PP1020 |
|---|---|---|---|---|---|---|
| Single machine (NUMA-EP + mirror) | — | **13.20-13.25** | ~20 | 5.94 | 13.9 | 34.9-35.1 |
| Two-machine 15L (TCP) | 11.64 | 11.58-11.85 | 18.2 | 7.17 | 18.7 | 34.1-34.5 |
| Two-machine 15L (RDMA) | 12.21 | **12.01-12.16** | 19.9 | ~~collapsed~~ fixed | ~~collapsed~~ fixed | ~~collapsed~~ fixed |
| 15L + `MIRROR=1` (TCP, ABBA) | 12.76 | **12.53-12.93 (+9.4%)** | 19.9 | 4.31-5.33 (-31%) | 15.1 (-19%) | 31.1-33.1 (-6%) |
| Two-machine 32L (3-34, TCP; planner's new optimum, measured) | 9.24 | 9.46-10.26 | 16.4 | 8.30 | 19.2 | 31.7 |

> Note: this table predates the IQ traits merge (2026-07-31 data); with traits merged, single-machine pp1020 is now 399-406 (4.1×) — two-machine re-measurement in progress.

Highlights:

- **Decode sped up across the board with 1.8× slave bandwidth**: worker compute is now 0.85-0.94 ms/layer (2.6 ms estimated on the old memory); two-machine TCP TG512 10.71→11.8 (+10%), RDMA →12.1; single-machine TG512 ~10→13.2. RDMA adds another +3% for decode (KB-scale frames).
- **RDMA large-frame collapse FIXED (2026-07-31, root cause: min_rnr_timer)**: rdma_cm's default RNR retry timer (~80 ms scale) meant that once a bulk frame drained the 8-slot receive ring, every RNR NAK cost a full timer tick and cascaded (measured: 6.2 MB frames at ~3 MB/s, 16 MB frames with p99 7.5 s stalls). Setting min_rnr_timer=0.01 ms via `ibv_modify_qp` after connect gives **5.5 GB/s on 16 MB frames with zero stalls (2× TCP) — GLM two-machine PP now beats TCP across the board: PP63 14.4 / PP254 38.1 / PP1020 up to 76.1 t/s (same-session TCP: 7.4 / ~18.7 / 33.4)**, with decode (small frames) unchanged. Also fixed: rdma_cm connect had no timeout (now 5 s + existing TCP fallback; stale workers no longer hang the master).
- **MAX-EFFORT layer mirroring (`GGML_REMOTE_EP_MIRROR=1`)**: decode gain holds (TG512 +9.4%, vs +10.6% on the old memory), but the PP gain has **inverted** (PP1020: +27% before → -6% now; PP63: -18.6% → -31%) — master-local MoE prefill bandwidth measured ~2× lower this session (reproduced on both old and new binaries; post-reboot environment suspected, numa_balancing ruled out), so moving prefill compute back to the master now loses. **Mirror on for decode, off for PP.**
- **Layer split stays at 15 slave layers**: the EP planner (`tools/epd/ep-plan.py`, new `--model glm` preset) predicted Ls* shifting 15→32 with the new bandwidth, but measurement refutes it (32L TG -13%) — the remote segment is nearly serial in decode, so a remote layer still costs more than a local one; DSV4 prediction 8→11 layers (not measured).
- Worker thread scan: -t 70 (12.16) beats -t 36 (11.66) — stay at physical cores; worker `GGML_EPD_NUMA=weighted` (NUMA weighted interleave) verified in production (symmetric slave nodes, 1:1 weights ≡ interleave).
- Known issues: master-local MoE prefill effective bandwidth is still the small-prompt PP bottleneck; RDMA large frames and missing rdma_cm reconnect timeout (master warmup can hang forever against a stale worker) are on the fix list.

Setup guide and full env reference: [tools/epd/README.md](tools/epd/README.md); bench scripts `tools/epd/bench-glm-{master,worker}.sh` + `bench-glm-client.py` (TG96/TG512 + PP amortization curve 5/63/254/1020 + gen48 comparison sampling).

## What's inside

- **NUMA mirror**: duplicate non-expert weights + KV per socket, pin threads per node — zero UPI traffic
- **NUMA row-window EP**: single-copy expert placement by **row windows within each expert** across sockets (mbind policy-level), local-first `mul_mat_id` + per-(expert,node) claim dispatch — halves expert memory, single-writer dst rows with zero merge, TG matches mirror; thread-affinity block-mapping fix (half the threads were reading over UPI)
- **AVX512/VNNI 8×8 repack kernels**: full coverage of Q2_K–Q6_K, Q8_0, MXFP4, IQ1_S/IQ1_M/IQ2_XS/IQ3_XXS — up to 4.9× on PP batches (gemm nr≥16); the Q3_K/Q5_K/Q6_K/Q8_0 x86 kernels and the entire IQ1_S/IQ1_M/IQ2_XS/IQ3_XXS stacks (block layout + repack + kernels) are new in this fork; batched MoE `mul_mat_id` dispatched to 4-row tile gemm (GLM PP 4.1×)
- **Fused ops**: DeepSeek-V4 hyper-connection CUDA kernel, fused MoE router, RMS_NORM absorption, GLM-DSA Lightning Indexer
- **MTP speculative decoding** (DeepSeek-V4)
- **Cross-machine EP (live)**: activation dispatch (KB-scale traffic) instead of weight transfer; the slave `llama-epd` worker owns MoE layers over a 100G RoCEv2 direct link (TCP/RDMA dual transports, RDMA opt-in); MAX-EFFORT layer mirroring (`GGML_REMOTE_EP_MIRROR`) overlaps the remote segment off the decode critical path (TG +9~11%); worker per-request fixed overhead eliminated (DSV4 PP +75%); NUMA weighted interleave (`GGML_EPD_NUMA=weighted`); the EP planner `tools/epd/ep-plan.py` recommends layer splits from measured bandwidth

## Status

Early development. `main` branch = production-ready; two-machine expert-parallel is live (DSV4 matches single-machine speed with 26% lower master RAM; GLM-5.2 numbers in the table above), with the EPD worker / RDMA transport / layer mirroring / planner in `tools/epd`. 2026-08-01: row-window EP + affinity fix and IQ2_XS/IQ3_XXS traits + mul_mat_id gemm dispatch merged (GLM PP 4.1×). On the GPU side we evaluated upstream's meta-backend tensor parallelism (`-sm tensor`): on 2× RTX 3090 + NVLink it measured **-20% (structural regression** — MIRRORED duplicated compute + allreduce boundaries + launch overhead; numerical correctness verified, KL=0), so it stays on an experimental branch and is not merged. 2026-08-04: long-context layer-major prefill (16K **604 tok/s** exact baseline / **752 tok/s** sparse opt-in, 4.7× over its first version), true dual-GPU same-layer MoE EP (+63%), batched top-k (21.6×), CPU repack/NUMA audit (MXFP4 Hybrid 4K +144%), and a CPU_REPACK offload correctness fix — see the long-context section above and `docs/CHANGES.md`.

**Full changelog and feature documentation: [docs/CHANGES.md](docs/CHANGES.md)** (Chinese) — NUMA architecture, CPU kernel format matrix, fused ops, environment variable reference.

Upstream tracking: based on llama.cpp `e8f19cc0a` (2026-07-16); `vendor` branch holds the base, merged regularly.

## License & Copyright

This project is a fork of [llama.cpp](https://github.com/ggml-org/llama.cpp). **The original project is copyrighted by ggml-org and the llama.cpp contributors** and released under the MIT License. All modifications in this fork are likewise released under the MIT License, with the original copyright and license notices retained (see [LICENSE](LICENSE)).
