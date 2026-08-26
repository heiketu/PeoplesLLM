# PeoplesLLM

> **[中文 README](README.md)**

**Run 200B-3T MoE models on a local server, stacking multiple CPUs, consumer GPUs, and multi-machine memory.**

PeoplesLLM is a llama.cpp fork for heterogeneous local inference: multi-NUMA/multi-device execution architecture, AVX-512 and GPU kernels, compute-oriented GGUF weight formats, plus a standalone expert-parallel runtime `tools/epd/` for multi-machine. Dense/attention/router/KV stay on GPU; routed MoE experts run on dual-socket CPU or remote NUMA workers - multiple nodes compute the **same slot, same layer's expert tasks** for one request.

Platform: dual-socket Intel Ice Lake-SP (152 logical threads, 251 GiB) + 2 x RTX 3090 (NVLink P2P) + ConnectX-5 100G direct-connected slave. Main models: DeepSeek-V4 / DSV4-Flash, GLM-5.2. All figures are target-only and non-speculative unless labeled `DSpark speculative`.

## Verified results

| Track | Paired baseline | Current | Change |
|---|---:|---:|---:|
| DSV4 pp2048, same-quality MXFP4 -> BAKED-MXFP4 | 312.48 tok/s | **362.92 tok/s** | **+16.1%** |
| DSV4 raw/no-DSpark decode, strict hot experts, 12-run A/B | 25.02 tok/s | **30.23 tok/s** | **+20.85%** |
| DSV4 single-slot raw/no-DSpark, four-worker strict remote-only -> GPU-hot + CPU-remote | 25.25 tok/s | **28.85 tok/s** | **+14.26%** |
| DSV4 single-slot raw/no-DSpark pure EP, 2 -> 4 NUMA workers | 22.1 tok/s | **25.2 tok/s** | **+14.03%** |
| DSV4 16K GPU MoE prefill | 213 tok/s | **334.5 tok/s** | **+57%** |
| DSV4 full-model CPU_REPACK load + smoke | 163.48 s | **98.66 s** | **-39.65% wall** |
| DSV4 DSpark speculative decode, n2/p0 | 23.9 tok/s | **30.1 tok/s** | **+26%** |
| GLM-5.2 pp512, 2 -> 4 NUMA worker EP | 24.13 tok/s | **40.59 tok/s** | **+68.2%** |

![Performance evolution](docs/img/evolution-staircase.png)

## Three-engine control plane

An execution plan is generated jointly by three logical engines:

- **TAE** (Topology-Aware): devices, interconnects, capacity, load; layer/expert/tensor granularity and mirror/split/owner placement.
- **UPE** (Unified Precision): weight-value and execution contracts, data epochs, sync semantics; shadow, PPL, hash, accepted/drafted joint gates.
- **DME** (Dynamic Matrix): GEMV/GEMM, tiles, fusion, ISA kernels per TG/PP/verify shape.

DME minimizes the phase-specific critical path over the TAE/UPE intersection. E4A is a co-design case: TAE selects the device layout, UPE validates logical values and version, DME picks the NR16/VNNI kernel.

## Architecture data

### NUMA row-window TP (DME)

![NUMA row-window TP on/off](docs/img/numa-tp-onoff.png)

| Workload | off | on | Change |
|---|---:|---:|---:|
| Hybrid TG512 | 16.59 | **25.01** | **+51%** |
| Hybrid PP2048 | 267.05 | **298.98** | +12% |
| Pure CPU TG128 | 7.23 | **11.65** | **+61%** |
| Pure CPU PP512 | 101.41 | **107.91** | +6.4% |

### Hybrid GPU execution

![GPU streamed MoE prefill](docs/img/gpu-prefill-streaming.png)

| Path | Baseline | Optimized | Change |
|---|---:|---:|---:|
| Dual-GPU TP decode | 13.6 tok/s | 20.3 tok/s | +49% |
| 16K GPU MoE streamed prefill | 213 tok/s | 334.5 tok/s | +57% |
| strict+AVX E4A/MXFP4 top-24 raw TG | 25.0167 tok/s | 30.2333 tok/s | +20.85% |
| GPU-hot + four-CPU-EP raw TG | 25.25 tok/s | 28.85 tok/s | +14.26% |

The strict slot-order + AVX512 combination passes TG/PP/PPL gates: TG +20.85%, PP2048 361.47, five-chunk PPL 2.7758 -> 2.7548 (-0.7565%).

### Multi-machine EP

![Remote expert parallelism](docs/img/remote-ep.png)

| Test | Before | After | Change |
|---|---:|---:|---:|
| GLM-5.2 MoE pp512, 2 -> 4 workers | 24.13 | **40.59** | **1.682 x** |
| DSV4 raw/no-DSpark TG512, 2 -> 4 workers | 22.1 | **25.2** | **1.140 x** |
| DSV4 raw/no-DSpark TG512, strict remote-only -> GPU-hot+CPU-remote | 25.25 | **28.85** | **+14.26%** |
| 64 B RPC RTT, TCP -> RoCEv2 | 42-74 us | **10-13 us** | about 4-6 x |

| Experiment | Data | Verdict |
|---|---|---|
| DSpark NMAX=2 multi-token bridge | K24 is the only K passing the PPL gate (2.7412) | warm-server +4.928% < 5% gate, default-off |
| MAX_EFFORT replica map | strict/max same text and PPL 2.7520 | paired TG +3.18%, numerically transparent |
| E8M0 value-contract fix | PPL +0.394% vs old baseline | REJECT, old baseline not reused |

## Kernel data

### Format size vs throughput (decode is not a strict size-bound workload)

| Observation | Data |
|---|---|
| 17-point hybrid matrix fit | TG/size R²=0.50, PP/size R²=0.25 |
| Counter-example: bigger is faster | Q4_K_XL 144 GiB beats IQ4_XS 127 GiB by 21.6% TG, 45.5% PP |
| Stream baking | load wall -30~40%, RSS -46%, throughput unchanged within 2 sigma |

![Size vs throughput dual-series trend](docs/img/size-vs-throughput.png)

### Stream baking

Covers K-quants, IQ, MXFP4, E4A:

| Object | Old | Baked | Result |
|---|---:|---:|---|
| 1.14085 GB MXFP4 tensor | 3.23 s | 2.12 s | RSS -46%, hash identical |
| 822 MB IQ3_XXS tensor | 4.37 s | 3.11 s | RSS -44.6%, hash identical |
| 145.26 GiB full-model load+smoke | 163.48 s | 98.66 s | -39.65% |
| NUMA-EP load+PP/TG three-round wall | 256.83 s | 177.55 s | TG diff <2 sigma |

### arec / reduction

- arec panel-stationary: UDNL_W4 PP 263.33 -> **366.58** (+39%); corrected UDNL_MX 418.95±3.50.
- Strict router-slot left fold + AVX512 merge: 15.41x (29,681.8 -> 1,926.6 ns/layer), PPL 2.7846 -> 2.7548.
- More kernel data: [BENCHMARKS](docs/BENCHMARKS.md).

## Format data

![Format size and PP/TG matrix](docs/img/quant-formats.png)

| Format | Size | pp2048 | tg512 | PPL | Role |
|---|---:|---:|---:|---:|---|
| MXFP4 | 145.26 GiB | 312.48 | 26.87 | 3.5830 | anchor |
| **UDNL_W4 + arec** | 146.36 GiB | 370.87 | 25.10 | 3.7997 | 4-bit compute-affine |
| **UDNL_MX + arec (corrected)** | 116.13 GiB | 418.95±3.50 | 26.90±0.25 | 4.6047 | capacity/PP; PPL 14.6% worse than Q3_K_XL |
| **BAKED-MXFP4 (E4A)** | 145.26 GiB | 362.92 | 24.83 | 3.5830 | MXFP4 value bit-exact |

![UDNL_MX tradeoff](docs/img/udnl-mx-tradeoff.png)

## UPE numerical audit

CPU AVX-512, CUDA MMVQ, and remote workers can produce different expert vectors from the same weights due to Q8 activation quantization, rounding, accumulation trees, FMA, and nonlinearity; path determinism does not imply cross-device equality, and near-tie logits can flip target tokens under DSpark.

**Evidence and interventions (all default-off):**

| Experiment | Data | Decision |
|---|---|---|
| Q8 produced-code replay | 34 CPU/GPU code differences in 15.68M real values (2.17 ppm), zero scale mismatches; 1876/4096 on half-step synthetic | verify against produced code/scale, not formula names |
| hot-scoped CPU-Q8 | code mismatch 33/10,144 -> 0; warm-server TG +8.26%, acceptance 266/490 -> 273/474 | five-chunk PPL 2.7855 fails 0.3% gate, REJECT |
| E8M0 0xff weight semantics | CUDA native NaN vs CPU 2^127 half-scale; unified edge tests bit-identical; global zeroing PPL +0.56% | keep CPU semantics; value-contract changes must re-pass quality gates |
| exclude 21:202,21:205 | layer21 RMSE 0.019 -> 1.69e-5; 10-chunk PPL 2.5843 passes | three-prompt warm-server 34.124 -> 33.019 (-3.24%), default-off |
| six-cell ISA/Vulkan audit | scalar/AVX2/AVX512-VNNI-on-off/CUDA/Vulkan each deterministic; same-top 96.1%-100%; first-divergence waterfall Q5 dot -> RoPE -> Q8 V -> QK reduction | exploratory: ISA/backend is part of the effective model |

## Build and run

```bash
# CUDA hybrid
cmake -B build-cuda -DGGML_CUDA=ON
cmake --build build-cuda -j

# Pure CPU
cmake -B build-cpu -DGGML_CUDA=OFF
cmake --build build-cpu -j

# DSV4 single-machine hybrid: GPU dense/attention/KV, dual-socket CPU routed experts
GGML_NUMA_EP=1 build-cuda/bin/llama-server \
  -m /path/to/model.gguf -ngl 99 -ncmoe 99 \
  -t 72 --threads-batch 72 --numa distribute -fa on -b 4096 -ub 1024
```

`tools/peoplesllm-run.sh` provides `dsv4-prod`, `dsv4-dual`, `glm-dual`, `cpu-pure` profiles. Workers, RDMA/TCP fallback, and full parameters: [QUICKSTART](docs/QUICKSTART.md), [PARAMETERS](docs/PARAMETERS.md), [EPD docs](tools/epd/README.md).

## Measurement and correctness

- A/B uses the same model, prompt, threads and offload; key results repeated ABBA/reverse; fixed-seed greedy outputs byte- or SHA256-checked.
- `pp`/`tg` are llama.cpp bench metrics, not guarantees for arbitrary concurrency, context, or hardware.
- Full data: docs/[BENCHMARKS.md](docs/BENCHMARKS.md), [CHANGES.md](docs/CHANGES.md), [ARCHITECTURE.md](docs/ARCHITECTURE.md).

## License

This project is a fork of [llama.cpp](https://github.com/ggml-org/llama.cpp), copyright ggml-org and contributors, MIT licensed. Fork changes are MIT as well; original copyright notice and [LICENSE](LICENSE) are retained.
