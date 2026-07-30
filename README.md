# PeoplesLLM

> **[English README →](README.en.md)**

**让超大 MoE 模型在本地廉价硬件上跑起来。** llama.cpp 特化分支，专注：双路乃至多路 CPU + 消费级 GPU 上的 200B ~ 3T 参数级 MoE 推理。

## 前情提要

作者是硬件方向出身，大学里只教过简单的 C、Python 和嵌入式汇编，对本地跑大模型一直有两点不满：**速度慢、门槛高**。于是选择使用 Kimi K3 配合 Kimi Code CLI 进行本项目调优。

本人只对技术概念进行指定——EP（专家并行）、GEMM 内核、指令集优化（AVX512/VNNI/VBMI）、NUMA 优化等——其余设计、编码、测试、调参**全部由 AI 完成**。因此代码可读性可能不佳，且可能存在潜在 BUG。本人目前只对**个人平台**（双路 Xeon 8360Y + 2× RTX 3090）上的运行负责。

目前主要调优对象：**GLM-5.2** 与 **DeepSeek-V4**。有问题欢迎提 issue。

## 实测数据

DeepSeek-V4 284B（Q3_K 量化），双路 Xeon 8360Y（Ice Lake）+ 2× RTX 3090：

| 实现 | 生成速度 (t/s) | 提示词处理 (t/s) |
|---|---|---|
| 主线 llama.cpp (b10173) | 5.85 | 23.6 |
| **PeoplesLLM（NUMA 镜像）** | **~33** | — |
| **PeoplesLLM（NUMA-EP）** | **28.5** | **310** |

> 注：以上 DeepSeek-V4 速度均为**未启用 MTP 投机解码**的实测值；开启 MTP 后 TG 还会更高（MTP 数据另行补充）。

GLM-5.2 745B（UD-Q2_K 量化）：生成 12.0 t/s（EP + GPU 专家卸载）。

### CPU 内核完整基准（AVX512/VNNI/VBMI 8×8 重排 vs 主线 legacy vec_dot）

Ice Lake 8360Y 实测，shape nc=16384 k=8192（超 L3 容量，真实 DRAM 带宽视角），数值为加速比（复现：`tests/test-repack-kernels --perf [线程数]`；* = 本分支新增/补全的 x86 内核）：

**单线程：**

| 格式 | gemv（TG） | gemm nr=4 | nr=8 | nr=16 | nr=32 |
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

**72 线程（双路满核，生产配置）：**

| 格式 | gemv（TG） | gemm nr=4 | nr=8 | nr=16 | nr=32 |
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

- gemv（TG 单 token）≈1× 是内存带宽物理上限；gemm（PP）提升来自 8×8 重排摊销权重读带宽
- Q4_K/Q6_K 的 nr≤8 为标量尾块（0.01~0.14×），生产路由 `gemm_min_nrows=16` 刻意避开；其余格式 4 行尾块全向量化
- 已知观察项：Q2_K 各档 ≈1× 无增益（权重已压到 ~2.6 bit，摊销空间殆尽）；Q6_K gemv 缓存内小形状 0.62× 为主线既有回退（大形状/多线程回到 ~1×）
- 小形状（2048×4096，L3 内）规律相同：nr=16 最高 3.96×（IQ1_M）、gemv 最高 1.39×（Q4_0）；全部 300 格数据均可用上述命令复现

### repack 端到端对比（DeepSeek-V4 284B，llama-bench 同配置 A/B）

NUMA-EP + mirror、72 线程、fa=1、batch 4096/ubatch 1024；`--no-repack` 为本分支给 llama-bench 补的开关：

| 配置 | pp512 (t/s) | tg128 (t/s) |
|---|---|---|
| 纯 CPU · repack 开（默认） | **114.33** | 17.34 |
| 纯 CPU · repack 关 | 97.17 | **17.66** |
| GPU 专家卸载 · repack 开（生产配置） | **151.12** | 29.04 |
| GPU 专家卸载 · repack 关 | 151.09 | **29.10** |

- 纯 CPU：PP **+17.7%**、TG **-1.8%**（TG 带宽封顶，repack gemv 略慢于 legacy，与微基准一致）
- GPU 卸载：差异 **<1%**——瓶颈移至 GPU 侧注意力与专家计算，CPU 矩阵乘占比缩小后 repack 收益被摊薄
- 结论：纯 CPU / 长 prompt 场景开 repack（默认）；GPU 卸载下开不开均可；只追极限 TG 可 `--no-repack`

### 双机 expert-parallel（GLM-5.2 745B UD-Q2_K_MXFP4，2026-07-31 全量测速）

拓扑：master（双路 8360Y 级 / 251G / 2×3090）←RoCEv2 100G 直连→ slave（双路 36 核 Ice Lake ES / **188G，双节点对称各 ~157 GB/s，合计 ~315 GB/s**（2026-07-30 内存整改后，为旧 174 GB/s 的 1.8×））。slave 以 `llama-epd` worker 认领 MoE 层 3-17（15 层 ≈43.5G，`--no-mmap` 常驻），master 本地 52 层 + GPU 专家卸载 9 层。正确性：以下全部配置 gen48（temp 0 / seed 42）与单机基线**逐字 IDENTICAL**。

| 配置 | TG96 | TG512 | PP5 | PP63 | PP254 | PP1020 |
|---|---|---|---|---|---|---|
| 单机（NUMA-EP + mirror） | — | **13.20-13.25** | ~20 | 5.94 | 13.9 | 34.9-35.1 |
| 双机 15 层（TCP） | 11.64 | 11.58-11.85 | 18.2 | 7.17 | 18.7 | 34.1-34.5 |
| 双机 15 层（RDMA） | 12.21 | **12.01-12.16** | 19.9 | ⚠塌陷 | ⚠塌陷 | ⚠塌陷 |
| 双机 15 层 + `MIRROR=1`（TCP，ABBA） | 12.76 | **12.53-12.93（+9.4%）** | 19.9 | 4.31-5.33（-31%） | 15.1（-19%） | 31.1-33.1（-6%） |
| 双机 32 层（3-34，TCP，规划器新最优点实测） | 9.24 | 9.46-10.26 | 16.4 | 8.30 | 19.2 | 31.7 |

要点：

- **slave 带宽 ×1.8 后 decode 全面提速**：worker 每层 compute 0.85-0.94ms（旧带宽估计 2.6ms），双机 TCP TG512 10.71→11.8（+10%）、RDMA →12.1；单机 TG512 ~10→13.2。RDMA 对 decode（KB 级帧）再 +3%。
- **⚠ RDMA 大帧塌陷（新发现，待修）**：PP 的 MB 级 REQ 帧在 RDMA 环上仅 ~3MB/s（每 256KB chunk ~77ms），PP 全面不可用；TCP 同帧正常。RDMA 目前只建议 decode 场景；PP 用 TCP。
- **MAX-EFFORT 层镜像（`GGML_REMOTE_EP_MIRROR=1`）**：decode 收益稳定（TG512 +9.4%，与旧带宽 +10.6% 相当）；但 PP 收益**反转**（PP1020 旧 +27% → 现 -6%，PP63 旧 -18.6% → 现 -31%）——master 本地 MoE prefill 有效带宽本次实测约减半（新旧二进制同现，与重启后环境相关，numa_balancing 已排除，待查），镜像把 prefill 计算搬回 master 由赚变亏。**decode 开镜像，PP 场景暂关**。
- **分层点维持 slave 15 层**：EP 规划器（`tools/epd/ep-plan.py`，新增 `--model glm` 预设）按新带宽预测 Ls* 15→32，实测 32 层 TG -13% 否定——decode 远程段近串行，远程每层成本仍高于本地；DSV4 预测 8→11 层（未实测）。
- worker 线程扫描：-t 70（12.16）> -t 36（11.66），维持物理核档位；worker `GGML_EPD_NUMA=weighted`（NUMA 加权交织）实测正常（slave 双节点对称，权重 1:1 等价 interleave）。
- 已知遗留：master 本地 MoE prefill 有效带宽偏低仍是小档 PP 主瓶颈；RDMA 大帧与 rdma_cm 重连无超时（worker 陈旧状态下 master warmup 会无限等待）列入待修。

双机快速上手与全部 env 见 [tools/epd/README.md](tools/epd/README.md)；测速脚本 `tools/epd/bench-glm-{master,worker}.sh` + `bench-glm-client.py`（TG96/TG512 + PP 摊销曲线 5/63/254/1020 + gen48 对拍采样）。

## 核心技术

- **NUMA 镜像**：非专家权重 + KV 双节点复制，线程绑核，UPI 流量归零
- **NUMA-EP**：专家单副本按插槽放置（mbind 策略级），mul_mat_id 本地优先计算 —— 内存减半，超大模型可装载
- **AVX512/VNNI 8×8 重排内核**：Q2_K~Q6_K、Q8_0、MXFP4、IQ1_S/IQ1_M 全格式覆盖，PP 批量（gemm nr≥16）加速最高 4.9×；其中 Q3_K/Q5_K/Q6_K/Q8_0 x86 内核与 IQ1_S/IQ1_M 全套（块布局+repack+内核）为本分支新增
- **融合算子**：dsv4 超连接 CUDA 内核、融合 MoE 路由器、RMS_NORM 吸收、GLM-DSA 闪电索引器
- **MTP 投机解码**（dsv4）
- **跨机 EP（已上线）**：激活派发（KB 级流量）而非权重传输，slave `llama-epd` worker 认领 MoE 层，RoCEv2 100G 直连（TCP/RDMA 双传输后端，RDMA opt-in）；MAX-EFFORT 层镜像（`GGML_REMOTE_EP_MIRROR`）把远程段从 decode 关键路径上重叠掉（TG +9~11%）；worker 固定开销已消除（DSV4 PP +75%）；支持 NUMA 加权交织（`GGML_EPD_NUMA=weighted`）；EP 规划器 `tools/epd/ep-plan.py` 按带宽实测给分层点

## 当前状态

早期开发阶段。`main` 分支 = 生产可用；双机 expert-parallel 已上线（DSV4 追平单机 + master 内存省 26%；GLM-5.2 见上方数据表），EPD worker / RDMA 后端 / 层镜像 / 规划器均在 `tools/epd`。

**完整改动清单见 [docs/CHANGES.md](docs/CHANGES.md)**（NUMA 体系、CPU 内核格式支持、融合算子、环境变量速查）。

主线跟踪：基于 llama.cpp `e8f19cc0a`（2026-07-16），`vendor` 分支保留基线，定期合并。

## 许可证与版权

本项目是 [llama.cpp](https://github.com/ggml-org/llama.cpp) 的分支。**原项目版权归 ggml-org 及 llama.cpp 全体贡献者所有**，采用 MIT 许可证发布。本分支的全部改动同样以 MIT 许可证发布，并保留原项目的版权声明与许可证全文（见 [LICENSE](LICENSE)）。
