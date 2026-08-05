# PeoplesLLM

> **[English README →](README.en.md)**

**让 200B~3T 级超大 MoE 模型在本地廉价硬件上跑起来、跑得快。**
llama.cpp 特化分支，面向双路/多路 CPU 服务器 + 消费级 GPU：CPU 计算内核、NUMA 体系、GPU 超长上下文 prefill、跨机专家并行四条主线全部自研优化。

![头条加速比](docs/benchmarks/headline_speedups.png)

## 简介

作者硬件方向出身，本项目全部设计、编码、测试、调参由 AI（Kimi K3 + Kimi Code CLI）完成，作者只指定技术方向（EP、GEMM 内核、AVX512/VNNI/VBMI、NUMA 等）。代码可读性可能不佳、可能有潜在 BUG，目前只对个人平台（双路 Xeon 8360Y + 2× RTX 3090）上的运行负责。主要调优对象：**DeepSeek-V4 / DSV4-Flash** 与 **GLM-5.2**。欢迎 issue。

测试平台：双路 Xeon 8360Y（Ice Lake，152 线程，251G RAM）+ 2× RTX 3090 24G（NVLink）+ ConnectX-5 100G 直连从机（双路 Ice Lake ES，188G）。基线 = 上游 llama.cpp 同机同口径 A/B。

## 特性一览

### CPU 推理

| 特性 | 使用场景 | 性能影响 |
|---|---|---|
| **AVX512/VNNI/VBMI 8×8 repack 内核**（Q2_K~Q6_K、Q8_0、MXFP4、IQ1_S/IQ1_M/IQ2_XS/IQ3_XXS 全格式） | 所有 CPU 计算；长 prompt 批量收益最大 | prefill gemm 微基准最高 **4.9×**；GLM-5.2 端到端 **PP 4.1×** |
| **NUMA 镜像 mirror**（`--numa mirror`） | 双路服务器、内存充裕 | TG 最优（vs 主线 **+9%**），跨插槽 UPI 流量归零 |
| **NUMA 行窗 EP**（`GGML_NUMA_EP=1`） | 内存受限、超大模型装载 | 专家权重**内存减半**，TG 追平 mirror，dst 行单写者零归并 |
| **融合算子**（超连接 HC、MoE router、RMS_NORM 吸收、DSA 闪电索引器） | DSV4 / GLM 全场景 | 消除分解路径与中间激活读写，PP 固定开销显著降低 |
| **MTP 投机解码** | DSV4 decode | TG 进一步提升（本文数据均未开 MTP） |

### GPU 推理（超长上下文）

| 特性 | 使用场景 | 性能影响 |
|---|---|---|
| **layer-major 超长 prefill**（`llama_decode_layer_major()`，层权重驻留 CUDA slot） | 4K~1M 超长 prompt | 16K PP 161→**604 tok/s**（稀疏 compact opt-in **752**），4.7× |
| **MoE 流式 prefill + 3-slot 双卡预取** | 专家权重在 host 内存的混合推理 | 每层权重只上传一次，跨 tile 复用 |
| **真双卡同层 expert-axis EP**（`GGML_CUDA_MOE_PP_EP`，opt-in） | 双 GPU prefill | 2K PP **+63%**（277→452 tok/s），输出逐位一致 |
| **batched top-k**（`GGML_CUDA_BATCHED_TOPK`） | DSA 稀疏注意力模型长上下文 | top-k 内核 **21.6×**，e2e PP +8.2% |
| **q1 32-head FA / raw-SWA decode ring** | 长上下文 decode | fixed TG64 **+17%** / TG512 **+7.8%**（ring 为 opt-in 验收中） |
| **GPU 专家卸载**（`-ot blk.N.ffn_*_exps=CUDA0/1`） | 显存富余 | 每卸载一层 TG 约 +1% |

### 跨机分布式 EP（`tools/epd/`）

| 特性 | 使用场景 | 性能影响 |
|---|---|---|
| **跨机专家并行**（激活派发，KB 级流量，非权重传输） | 单机内存装不下模型 | DSV4 双机追平单机速度 + master 内存**省 26%** |
| **RoCEv2 RDMA 传输**（`GGML_REMOTE_EP_RDMA=1`，TCP 自动兜底） | 有 IB/RoCE 网卡；纯千兆环境自动回退 | RTT 42-74µs→**10-13µs**；大帧修复后 PP1020 33.4→**76.1 tok/s（2.3×）** |
| **MAX-EFFORT 层镜像**（`GGML_REMOTE_EP_MIRROR=1`） | master 内存有空闲、decode 为主 | 远程段从关键路径消失，TG **+9~11%** |
| **EP 规划器**（`tools/epd/ep-plan.py`） | 部署前分层决策 | 按实测带宽/延迟给分层点，校准误差 ≤1.5% |

### 模型支持

DeepSeek-V4 / DSV4-Flash（含原生 MXFP4）、GLM-5.2（DSA）、MiniMax-M3；GGUF 字节级修复工具链（量化块腐化扫描+补丁，修复过 GLM-5.2 官方文件 138 个腐化块导致的乱码）。

## 性能实测

### 对标主线 llama.cpp（同机同口径 A/B）

DeepSeek-V4 284B（Q3_K），GPU 卸载 14 层专家，72 线程：

![DSV4 vs upstream](docs/benchmarks/dsv4_vs_upstream.png)

行窗 EP 与 mirror 双双反超主线：**PP 2.1-2.2×、TG +9%**；行窗 EP 额外省一半专家内存。PP 优先或单路场景 isolate 配置 pp512 可达 370 tok/s。

### GLM-5.2 745B：IQ traits + gemm 分流

![GLM traits](docs/benchmarks/glm_traits.png)

### CPU 内核：8×8 repack vs 主线 legacy vec_dot

![CPU kernel speedup](docs/benchmarks/cpu_kernel_speedup.png)

gemv（decode）≈1× 是内存带宽物理上限；prefill gemm 提升来自 8×8 重排摊销权重读带宽。完整 300 格数据可用 `tests/test-repack-kernels --perf` 复现。

### 超长上下文：layer-major prefill（DSV4-Flash，16K）

![16K PP 演进](docs/benchmarks/longctx_pp_progression.png)

![MXFP4 Hybrid CPU 审计与双卡 EP](docs/benchmarks/mxfp4_hybrid_cpu_audit.png)

原生 MXFP4 版（155GB，137GiB 专家单份 CPU_REPACK + 双路 NUMA EP）经 CPU 审计三连修后 4K PP +144%。

### 长上下文 decode（16K，固定负载 A/B）

![16K TG 改进](docs/benchmarks/longctx_tg_improvements.png)

长上下文 TG 衰减根因已定位为 GPU attention/KV 的物理 dense 扫描，逐项修复中。16K GPU 侧时间分解（Nsight）：

![16K 热点分解](docs/benchmarks/longctx_hotspots.png)

### 双机 expert-parallel（GLM-5.2，100G RoCEv2 直连）

![双机 EP](docs/benchmarks/dual_machine_ep.png)

### 生产 server：DSV4-Flash 8 槽 1M 上下文

![Flash PP/TG 曲线](docs/benchmarks/flash_pp_tg_curve.png)

8 槽共享 1M 上下文（每槽 128K），PP 峰值 511 tok/s（ubatch 1024-4096），TG 全程 20-25 tok/s 平稳。

> 所有数字均为实测，口径与复现方式见 `docs/CHANGES.md` 与 `docs/benchmarks/`（绘图脚本同目录）。已实测否决的路线（full tensor split -44%、meta-backend TP -20%、跨 tile 双 scheduler）也记录在案。

## 文档

- **[docs/CHANGES.md](docs/CHANGES.md)** — 完整改动清单（NUMA 体系、CPU 内核格式矩阵、融合算子、分布式 EP）
- **[docs/PEOPLESLLM-PARAMS.md](docs/PEOPLESLLM-PARAMS.md)** — 全部 env/CLI 参数手册与生产配方
- **[docs/DEVELOPMENT.md](docs/DEVELOPMENT.md)** — 构建、测试与分支约定
- **[docs/LONG-CONTEXT-1M.md](docs/LONG-CONTEXT-1M.md)** — 1M 上下文内存/显存预算与验收门槛
- **[tools/epd/README.md](tools/epd/README.md)** — 双机 EP 快速上手

## 当前状态

早期开发阶段，`main` 分支 = 生产可用。持续推进：GPU prefill 向 vLLM/KTransformers 级吞吐演进（16K 目标 1000+ tok/s）、CPU/GPU 联合流水、统一 GPU-prefill/CPU-decode 权重表示。主线跟踪：基于 llama.cpp `e8f19cc0a`（2026-07-16），`vendor` 分支保留基线，定期合并。

## 许可证与版权

本项目是 [llama.cpp](https://github.com/ggml-org/llama.cpp) 的分支。**原项目版权归 ggml-org 及 llama.cpp 全体贡献者所有**，采用 MIT 许可证发布。本分支的全部改动同样以 MIT 许可证发布，并保留原项目的版权声明与许可证全文（见 [LICENSE](LICENSE)）。
