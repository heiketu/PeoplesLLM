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
| **AVX512/VNNI/VBMI 8×8 repack 内核**（Q2_K~Q6_K、Q8_0、MXFP4、IQ1_S/IQ1_M/IQ2_XXS 全格式） | 所有 CPU 计算；长 prompt 批量收益最大 | prefill gemm 微基准最高 **4.9×**；GLM-5.2 端到端 **PP 4.1×** |
| **IQ2_XXS AVX512 repack gemv/gemm 内核**（x8 布局） | IQ2_XXS 权重模型 CPU 推理 | gemv 微基准 **17×**；全模型 A/B 暂持平（瓶颈在专家 mul_mat_id 未覆盖），模型级收益待扩展 |
| **NUMA 镜像 mirror**（`--numa mirror`） | 双路服务器、内存充裕 | TG 最优（vs 主线 **+9%**），跨插槽 UPI 流量归零 |
| **NUMA 行窗 EP**（`GGML_NUMA_EP=1`） | 内存受限、超大模型装载；EPD worker 单机跨 node 同样适用 | 专家权重**内存减半**，TG 追平 mirror，dst 行单写者零归并；ABAB 实测混合 tg512 **+49%**、纯 CPU tg128 **+61%**（含抗负载退化），pp +6~12% |
| **层级 barrier**（`GGML_NUMA_HIER_BARRIER=1`） | 多 NUMA 节点推理 | +0.9%（噪声级但零成本），建议常开 |
| **repack gemv 软件预取**（`GGML_REPACK_GEMV_PREFETCH=1`） | repack 内核 decode | TG **+2.6%**，建议常开 |
| **融合算子**（超连接 HC、MoE router、RMS_NORM 吸收、DSA 闪电索引器） | DSV4 / GLM 全场景 | 消除分解路径与中间激活读写，PP 固定开销显著降低 |
| **MTP 投机解码** | DSV4 decode | TG 进一步提升（本文数据均未开 MTP） |

### GPU 推理（超长上下文）

| 特性 | 使用场景 | 性能影响 |
|---|---|---|
| **layer-major 超长 prefill**（`llama_decode_layer_major()`，层权重驻留 CUDA slot） | GPU-local MoE 的 4K~1M 超长 prompt | 16K PP 161→**604 tok/s**（稀疏 compact opt-in **752**），4.7×；远程 CPU EP + DSpark 不适用，见最新进展 |
| **MoE 流式 prefill + 3-slot 双卡预取** | 专家权重在 host 内存的混合推理 | 每层权重只上传一次，跨 tile 复用 |
| **真双卡同层 expert-axis EP**（`GGML_CUDA_MOE_PP_EP`，opt-in） | 双 GPU prefill | 2K PP **+63%**（277→452 tok/s），输出逐位一致 |
| **batched top-k**（`GGML_CUDA_BATCHED_TOPK`） | DSA 稀疏注意力模型长上下文 | top-k 内核 **21.6×**，e2e PP +8.2% |
| **q1 32-head FA + raw-SWA decode ring**（ring 默认开，`LLAMA_DSV4_COMPACT_DECODE_SWA=0` 可关） | 长上下文 decode | q1 decode 图宽与 prompt 长度解耦，fixed TG64 8.89→**12.09**（+33~36%）、TG512 +4~8%；多 slot 自动回退 |
| **q8 compact 稀疏 FA**（默认开，q8_0 KV） | q8 KV 长上下文 decode | q1 decode 只物化 top-k 选中行，TG64 9.75→**10.69**（+10%），最快 q1 decode 路径（超 f16 dense）；f16 fused sparse 保持 opt-in（16K 实测 -12%） |
| **多流（小 q）稀疏 FA**（`LLAMA_DSV4_FUSED_INDEXED_FA=3/4`、`LLAMA_DSV4_Q8_SPARSE_FA=2`，opt-in） | 多 slot 并发 decode | 正确性三方 byte-identical；≤16K 持平（权重带宽主导），收益主场 256K+ 长上下文 |
| **GPU 流式 prefill**（方案A，server 集成） | server 长 prompt | 整 tile 路径 PP 63→**127.6 tok/s**（2×）；紧凑环形 SWA 缓存下按资格门回落 chunked，`--swa-full` 启用 |
| **GPU 专家卸载**（`-ot blk.N.ffn_*_exps=CUDA0/1`） | 显存富余 | 每卸载一层 TG 约 +1% |

### 跨机分布式 EP（`tools/epd/`）

| 特性 | 使用场景 | 性能影响 |
|---|---|---|
| **跨机专家并行**（激活派发，KB 级流量，非权重传输） | 单机内存装不下模型 | DSV4 双机追平单机速度 + master 内存**省 26%** |
| **RoCEv2 RDMA 传输**（`GGML_REMOTE_EP_RDMA=1`，TCP 自动兜底） | 有 IB/RoCE 网卡；纯千兆环境自动回退 | RTT 42-74µs→**10-13µs**；大帧修复后 PP1020 33.4→**76.1 tok/s（2.3×）** |
| **MAX-EFFORT 层镜像**（`GGML_REMOTE_EP_MIRROR=1`） | master 内存有空闲、decode 为主 | 远程段从关键路径消失，TG **+9~11%** |
| **单 slot 四 NUMA 真 EP**（`SCHED_KLOCAL=0` + 热点专家副本） | 2 台双路机器共同计算同一请求 | GLM-5.2 PP512：2 NUMA **24.13**→4 NUMA **40.59 tok/s（1.682×）**；75 层输出逐字节一致 |
| **EP 规划器**（`tools/epd/ep-plan.py`） | 部署前分层决策 | 按实测带宽/延迟给分层点，校准误差 ≤1.5% |

### 模型支持

DeepSeek-V4 / DSV4-Flash（含原生 MXFP4）、GLM-5.2（DSA）、MiniMax-M3；GGUF 字节级修复工具链（量化块腐化扫描+补丁，修复过 GLM-5.2 官方文件 138 个腐化块导致的乱码）。
## 快速开始

```bash
# CUDA 构建（单机混合推理）
cmake -B build-cuda -DGGML_CUDA=ON && cmake --build build-cuda -j

# 生产推荐：DSV4-Flash mxfp4，双路 NUMA EP + GPU 卸载
GGML_NUMA_EP=1 GGML_NUMA_HIER_BARRIER=1 GGML_REPACK_GEMV_PREFETCH=1 \
  build-cuda/bin/llama-server -m model.gguf -ngl 99 -ncmoe 99 -t 72 \
  --numa distribute -fa 1 -b 4096 -ub 1024
```

双机 EP、纯 CPU、AVX2-only 平台构建与全部部署细节见 **[docs/QUICKSTART.md](docs/QUICKSTART.md)**。

## 文档

- **[docs/QUICKSTART.md](docs/QUICKSTART.md)** — 上手指南：构建、单机/双机推荐配置、常见坑
- **[docs/PARAMETERS.md](docs/PARAMETERS.md)** — 全部 env/CLI 参数手册与生产配方
- **[docs/BENCHMARKS.md](docs/BENCHMARKS.md)** — 完整实测数据与迭代进展（对标主线 A/B）
- **[docs/CHANGES.md](docs/CHANGES.md)** — 技术改动清单（NUMA 体系、CPU 内核、融合算子、分布式 EP）
- **[tools/epd/README.md](tools/epd/README.md)** — 双机 EP 快速上手

## 当前状态

早期开发阶段，生产使用前请按输出哈希、slot context 与显存余量门槛自行验收。主线同步目标：llama.cpp `4df29be4f`（2026-08-15，96 提交合并完成，AVX2/AVX512 双构建回归通过（tg 持平，pp -5.4% 归因上游改动，见 BENCHMARKS））。持续推进：GPU prefill 吞吐、CPU/GPU 联合流水、统一权重表示。

## 许可证与版权

本项目是 [llama.cpp](https://github.com/ggml-org/llama.cpp) 的分支。**原项目版权归 ggml-org 及 llama.cpp 全体贡献者所有**，采用 MIT 许可证发布。本分支的全部改动同样以 MIT 许可证发布，并保留原项目的版权声明与许可证全文（见 [LICENSE](LICENSE)）。
