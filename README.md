# PeoplesLLM

> **[English README →](README.en.md)**

**让超大 MoE 模型在本地廉价硬件上跑起来。** llama.cpp 特化分支，专注：双路乃至多路 CPU + 消费级 GPU 上的 200B ~ 3T 参数级 MoE 推理。

## 实测数据

DeepSeek-V4 284B（Q3_K 量化），双路 Xeon 8360Y（Ice Lake）+ 2× RTX 3090：

| 实现 | 生成速度 (t/s) | 提示词处理 (t/s) |
|---|---|---|
| 主线 llama.cpp (b10173) | 5.85 | 23.6 |
| **PeoplesLLM（NUMA 镜像）** | **~33** | — |
| **PeoplesLLM（NUMA-EP）** | **28.5** | **310** |

GLM-5.2 745B（UD-Q2_K 量化）：生成 12.0 t/s（EP + GPU 专家卸载）。

### CPU 内核微基准（AVX512/VNNI/VBMI 8×8 重排 vs 主线 legacy vec_dot）

单线程，nc=2048 k=4096，Ice Lake 8360Y 实测（复现：`tests/test-repack-kernels --perf`）：

| 格式 | gemv（TG 单 token） | gemm nr=4（小批） | gemm nr=16（PP 批量） |
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

- gemv（TG）≈1× 是单 token 内存带宽的物理上限，不是内核问题；gemm（PP）的提升来自 8×8 重排摊销权重读带宽
- "—"：这些格式 `gemm_min_nrows=16`，小批量走 gemv，nr=4 的 gemm 生产路径不触发（Q4_K/Q6_K 的 4 行尾块是标量实现，路由刻意避开）
- 已知观察项：Q2_K gemm nr=4 略亏（0.90×）；Q6_K gemv 0.64× 为主线内核的既有回退，均在待修清单上

### repack 端到端对比（DeepSeek-V4 284B，纯 CPU）

同一 llama-bench 配置（NUMA-EP + mirror、72 线程、fa=1、batch 4096/ubatch 1024、无 GPU 卸载），`--no-repack` 为本分支给 llama-bench 补的开关：

| 配置 | pp512 (t/s) | tg128 (t/s) |
|---|---|---|
| repack 开（默认） | **114.33** | 17.34 |
| repack 关 | 97.17 | **17.66** |

- PP **+17.7%**：端到端增幅小于内核微基准（2.4~4×），因为注意力等非矩阵乘开销占比大
- TG **-1.8%**：单 token 受内存带宽封顶，repack gemv 略慢于 legacy vec_dot（与微基准一致）；不跑长 prompt、只追求极限 TG 时可 `--no-repack`

## 核心技术

- **NUMA 镜像**：非专家权重 + KV 双节点复制，线程绑核，UPI 流量归零
- **NUMA-EP**：专家单副本按插槽放置（mbind 策略级），mul_mat_id 本地优先计算 —— 内存减半，超大模型可装载
- **AVX512/VNNI 8×8 重排内核**：Q2_K~Q6_K、Q8_0、MXFP4、IQ1_S/IQ1_M 全格式覆盖，PP 批量（gemm nr≥16）加速 2.4~4×；其中 Q3_K/Q5_K/Q6_K/Q8_0 x86 内核与 IQ1_S/IQ1_M 全套（块布局+repack+内核）为本分支新增
- **融合算子**：dsv4 超连接 CUDA 内核、融合 MoE 路由器、RMS_NORM 吸收、GLM-DSA 闪电索引器
- **MTP 投机解码**（dsv4）
- **跨机 EP（开发中）**：激活派发（KB 级流量）而非权重传输，InfiniBand EDR 互联可扩展 CPU MoE 节点，目标 2.8T 级模型

## 当前状态

早期开发阶段。`main` 分支 = 生产可用；跨机 EP 传输层（`tools/epd`）已就绪，主机侧集成进行中。

**完整改动清单见 [docs/CHANGES.md](docs/CHANGES.md)**（NUMA 体系、CPU 内核格式支持、融合算子、环境变量速查）。

主线跟踪：基于 llama.cpp `e8f19cc0a`（2026-07-16），`vendor` 分支保留基线，定期合并。

## 许可证

MIT（与上游 llama.cpp 一致）
