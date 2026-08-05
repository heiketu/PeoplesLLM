# PeoplesLLM

> **[English README →](README.en.md)**

**让超大 MoE 模型在本地廉价硬件上跑起来。** llama.cpp 特化分支，专注：双路乃至多路 CPU + 消费级 GPU 上的 200B ~ 3T 参数级 MoE 推理。

## 前情提要

作者是硬件方向出身，大学里只教过简单的 C、Python 和嵌入式汇编，对本地跑大模型一直有两点不满：**速度慢、门槛高**。于是选择使用 Kimi K3 配合 Kimi Code CLI 进行本项目调优。

本人只对技术概念进行指定——EP（专家并行）、GEMM 内核、指令集优化（AVX512/VNNI/VBMI）、NUMA 优化等——其余设计、编码、测试、调参**全部由 AI 完成**。因此代码可读性可能不佳，且可能存在潜在 BUG。本人目前只对**个人平台**（双路 Xeon 8360Y + 2× RTX 3090）上的运行负责。

目前主要调优对象：**GLM-5.2** 与 **DeepSeek-V4**。有问题欢迎提 issue。

## 实测数据

DeepSeek-V4 284B（Q3_K 量化），双路 Xeon 8360Y（Ice Lake）+ 2× RTX 3090（GPU 卸载 14 层专家，72 线程，`--no-mmap`，2026-08-01 同口径复测）：

| 实现 | tg512 (t/s) | pp512 (t/s) | MoE 算子单次耗时 |
|---|---|---|---|
| 主线 llama.cpp（NUMA distribute） | 26.9-28.0 | 150.5-161.3 | ~102.5-125µs |
| PeoplesLLM（isolate 单路参照） | 25.7-25.9 | 363.6-370.2 | 133.4µs |
| **PeoplesLLM（NUMA 行窗 EP，内存减半）** | **30.34-30.48** | 318-333 | **88.4µs** |
| **PeoplesLLM（NUMA 镜像 mirror）** | **30.35-30.44** | **344-349** | 80.3µs |

> 注：以上 DeepSeek-V4 速度均为**未启用 MTP 投机解码**的实测值；开启 MTP 后 TG 还会更高（MTP 数据另行补充）。
> PP 优先或单路场景 isolate 反而最快（pp512 370）；内存充裕时 mirror 是 TG 最优；内存受限时行窗 EP 以 ~10% MoE 开销换内存减半。

GLM-5.2 745B（UD-Q2_K_MXFP4，单机 + 2×3090，9 层专家 GPU 卸载，2026-08-01 同口径 A/B）：

| 实现 | tg512 (t/s) | pp512 (t/s) | pp1020 (t/s) |
|---|---|---|---|
| PeoplesLLM 无 IQ traits | 11.23-11.27 | 56.8 | 98.2-99.1 |
| **PeoplesLLM + IQ2_XS/IQ3_XXS traits** | **11.95（+6%）** | **298-307** | **399-406（4.1×）** |

### 行窗口 EP + 线程亲和根因修复（2026-08-01）

- **行窗口 EP**：专家并行放置从整专家粒度改为**每专家内行窗粒度**（节点 n 拥有每专家第 n 段行窗，mbind 单份），dst 行单写者零归并，结构性消除 batch=1 尾部不均衡（旧专家粒度 EP 双路只能跑出单路成绩）。
- **线程亲和根因修复**：DISTRIBUTE 模式的钉核映射（`ith % n_nodes` 交错）与 EP 计算侧的窗口归属映射（块划分）不一致，**恰好一半线程被钉在错误节点、100% 跨 UPI 读"本地"行窗**（MoE 单次 155.7µs vs mirror 80.3µs）。修复后 88.4µs（收回差距的 89%），EP tg512 从 20.7-26.3 → **30.34-30.48，追平 mirror、反超主线 distribute**；正确性 mirror==EP 逐 token 一致。
- 残余 ~10% MoE 开销已归因于 claim 两阶段计算路径（非放置问题）；`GGML_NUMA_EP_STATIC/EP_CLAIM/EP_CHUNK` 诊断开关见参数手册。

### IQ2_XS/IQ3_XXS repack 内核 + mul_mat_id gemm 分流（2026-08-01）

- GLM-5.2 专家主力量化 **IQ2_XS/IQ3_XXS 补齐 8×8 交错 repack traits**（布局 + 转换 + generic/AVX512(VNNI+VBMI 门禁) gemv/gemm 全套）：微基准 gemm **2.3-2.5×** vec_dot、native==generic 位级一致、test-backend-ops 1996/1996 PASS。
- **mul_mat_id gemm 分流**：repack 的 MoE 批量路径此前全部逐行 gemv（Q2_K 同病），现 4 行一组 gather+量化进交错 tile 走 gemm——GLM 单机 **PP1020 99→406（4.1×）、tg512 +6%**，EP 与 mirror 逐 token 一致。

### 超长上下文 layer-major prefill + 双卡 MoE EP（2026-08-04，DSV4-Flash 0731）

新增实验接口 `llama_decode_layer_major()`：外层按层、内层按 token tile 执行，当前层专家权重驻留私有 CUDA slot、同层后续 tile 不重复 H2D，把长 prompt 的"每 ubatch 重复上传全模型权重"压到"每层上传一次"。16K prefill 从初版 161 tok/s 提升到 **604 tok/s**（精确 dense 基线，最终 logits 指纹逐位一致），稀疏 raw-KV 紧凑化（opt-in）达 **752 tok/s**：

![16K PP 演进](docs/benchmarks/longctx_pp_progression.png)

- **真双卡同层 expert-axis EP**（`GGML_CUDA_MOE_PP_EP`，opt-in）：同层专家沿专家轴切到 CUDA0/CUDA1 并行，2K 从串行正确基线 277 → **452 tok/s（+63%）**，16K 581 tok/s，输出逐位一致。
- **完整有序 K/V tile reuse**：FA 保持原 128+128 KQ 运算顺序扩大 shared-memory row stride 复用完整 tile，16K PP +3.9%，logits 指纹完全一致（改变归约分组的 256 合并变体已否决回退）。
- **batched top-k**（`GGML_CUDA_BATCHED_TOPK`）：DSA indexer 的 16384×4096 k=512 top-k 从 65.5ms → **3.03ms（21.6×）**，launch 数 139 万 → 1.9 万，e2e PP +8.2%。
- **稀疏 raw-KV 紧凑化**（opt-in，query batch ≥256）：raw KV 有效区间紧凑到 ≤512 行再追加 compressed top-k 并集，16K PP 596→**752 tok/s**；浮点归约分组变化故默认关闭，q1 decode 自动回 dense FA。

原生 MXFP4 版 DSV4-Flash（155GB，137GiB 专家单份 CPU_REPACK + 双路 NUMA EP，dense/attention/KV 在双 3090）经 CPU 审计（NUMA claim false sharing / 行窗错位 / 大 chunk scratch 越界三连修 + MXFP4 scale SIMD + Q8 流式 interleave）后：

![MXFP4 Hybrid CPU 审计与双卡 EP](docs/benchmarks/mxfp4_hybrid_cpu_audit.png)

长上下文 decode 的衰减根因已定位为 GPU attention/KV 的物理 dense 扫描（41 层 concat 全量处理），逐项修复中：

![16K TG 改进](docs/benchmarks/longctx_tg_improvements.png)

- **q1 FA 32-head MMA 实例**：DSV4 64 个 Q 头共享一个 512 维 K/V 头，K≥1024 自动选 32-head，16K 微基准 185.6→73.9µs，fixed TG64 **+17.0%**。
- **raw-SWA decode ring**（opt-in 验收中）：prefill 后仅搬最新 128 个 raw SWA cell，decode 物理宽度限 256，fixed TG512 11.30→**12.18（+7.8%）**，CUDA graph warmup reset 130→48 次。

16K GPU 侧时间分解（Nsight，604 tok/s 路径）——FA 与权重 H2D 是下一主战场：

![16K 热点分解](docs/benchmarks/longctx_hotspots.png)

> 复现与口径：`tests/test-layer-major.cpp`（`BENCH_GPU_STREAM` PP+TG 同次加载统一口径、`LLAMA_BENCH_FIXED_TG=1` 固定路由 A/B）；full tensor split 与跨 tile 双 scheduler 两条路线已实测否决（PP -44% / CPU threadpool 语义冲突），记录见 `docs/CHANGES.md`。

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
| 双机 15 层（RDMA） | 12.21 | **12.01-12.16** | 19.9 | ~~塌陷~~已修复 | ~~塌陷~~已修复 | ~~塌陷~~已修复 |
| 双机 15 层 + `MIRROR=1`（TCP，ABBA） | 12.76 | **12.53-12.93（+9.4%）** | 19.9 | 4.31-5.33（-31%） | 15.1（-19%） | 31.1-33.1（-6%） |
| 双机 32 层（3-34，TCP，规划器新最优点实测） | 9.24 | 9.46-10.26 | 16.4 | 8.30 | 19.2 | 31.7 |

> 注：本表为 IQ traits 合入前（2026-07-31）数据；traits 合并后单机 pp1020 已达 399-406（4.1×），双机复测数据待补充。

要点：

- **slave 带宽 ×1.8 后 decode 全面提速**：worker 每层 compute 0.85-0.94ms（旧带宽估计 2.6ms），双机 TCP TG512 10.71→11.8（+10%）、RDMA →12.1；单机 TG512 ~10→13.2。RDMA 对 decode（KB 级帧）再 +3%。
- **RDMA 大帧塌陷已修复（2026-07-31，根因=min_rnr_timer）**：rdma_cm 默认 RNR 重试定时器 ~80ms 级，大帧打空 8 槽接收环触发 RNR NAK 后每次重试睡一个定时器并级联塌陷（曾实测 6.2MB 帧 ~3MB/s、16MB 帧 p99 7.5s）；建连后 `ibv_modify_qp` 设 min_rnr_timer=0.01ms 后 **16MB 帧 5.5GB/s 零停顿（2× TCP），GLM 双机 PP 全面反超 TCP：PP63 14.4 / PP254 38.1 / PP1020 最高 76.1 t/s（同轮 TCP 7.4 / ~18.7 / 33.4）**，decode 小帧行为不变。另修复 rdma_cm 建连无超时（5s 上限 + 既有 TCP 回退，陈旧 worker 不再假死）。
- **MAX-EFFORT 层镜像（`GGML_REMOTE_EP_MIRROR=1`）**：decode 收益稳定（TG512 +9.4%，与旧带宽 +10.6% 相当）；但 PP 收益**反转**（PP1020 旧 +27% → 现 -6%，PP63 旧 -18.6% → 现 -31%）——master 本地 MoE prefill 有效带宽本次实测约减半（新旧二进制同现，与重启后环境相关，numa_balancing 已排除，待查），镜像把 prefill 计算搬回 master 由赚变亏。**decode 开镜像，PP 场景暂关**。
- **分层点维持 slave 15 层**：EP 规划器（`tools/epd/ep-plan.py`，新增 `--model glm` 预设）按新带宽预测 Ls* 15→32，实测 32 层 TG -13% 否定——decode 远程段近串行，远程每层成本仍高于本地；DSV4 预测 8→11 层（未实测）。
- worker 线程扫描：-t 70（12.16）> -t 36（11.66），维持物理核档位；worker `GGML_EPD_NUMA=weighted`（NUMA 加权交织）实测正常（slave 双节点对称，权重 1:1 等价 interleave）。
- 已知遗留：master 本地 MoE prefill 有效带宽偏低仍是小档 PP 主瓶颈；RDMA 大帧与 rdma_cm 重连无超时（worker 陈旧状态下 master warmup 会无限等待）列入待修。

双机快速上手与全部 env 见 [tools/epd/README.md](tools/epd/README.md)；测速脚本 `tools/epd/bench-glm-{master,worker}.sh` + `bench-glm-client.py`（TG96/TG512 + PP 摊销曲线 5/63/254/1020 + gen48 对拍采样）。

## 核心技术

- **NUMA 镜像**：非专家权重 + KV 双节点复制，线程绑核，UPI 流量归零
- **NUMA 行窗 EP**：专家单副本按**每专家内行窗口**跨插槽放置（mbind 策略级），mul_mat_id 本地优先计算 + per-(专家,节点) claim 派发 —— 内存减半、dst 行单写者零归并，TG 追平镜像；线程亲和块划分修复（半数线程跨 UPI 根因）
- **AVX512/VNNI 8×8 重排内核**：Q2_K~Q6_K、Q8_0、MXFP4、IQ1_S/IQ1_M/IQ2_XS/IQ3_XXS 全格式覆盖，PP 批量（gemm nr≥16）加速最高 4.9×；其中 Q3_K/Q5_K/Q6_K/Q8_0 x86 内核与 IQ1_S/IQ1_M/IQ2_XS/IQ3_XXS 全套（块布局+repack+内核）为本分支新增；MoE 批量 mul_mat_id 走 4 行 tile gemm 分流（GLM PP 4.1×）
- **融合算子**：dsv4 超连接 CUDA 内核、融合 MoE 路由器、RMS_NORM 吸收、GLM-DSA 闪电索引器
- **MTP 投机解码**（dsv4）
- **跨机 EP（已上线）**：激活派发（KB 级流量）而非权重传输，slave `llama-epd` worker 认领 MoE 层，RoCEv2 100G 直连（TCP/RDMA 双传输后端，RDMA opt-in）；MAX-EFFORT 层镜像（`GGML_REMOTE_EP_MIRROR`）把远程段从 decode 关键路径上重叠掉（TG +9~11%）；worker 固定开销已消除（DSV4 PP +75%）；支持 NUMA 加权交织（`GGML_EPD_NUMA=weighted`）；EP 规划器 `tools/epd/ep-plan.py` 按带宽实测给分层点

## 当前状态

早期开发阶段。`main` 分支 = 生产可用；双机 expert-parallel 已上线（DSV4 追平单机 + master 内存省 26%；GLM-5.2 见上方数据表），EPD worker / RDMA 后端 / 层镜像 / 规划器均在 `tools/epd`。2026-08-01：行窗 EP + 亲和修复、IQ2_XS/IQ3_XXS traits + mul_mat_id gemm 分流已合入（GLM PP 4.1×）；GPU 侧评估过上游 meta-backend 张量并行（`-sm tensor`），2×3090+NVLink 实测 **-20% 负收益**（MIRRORED 复制计算+allreduce 边界+launch 开销为结构性，数值正确性 KL=0 验证通过），不合入、保留在实验分支。2026-08-04：超长上下文 layer-major prefill（16K **604 tok/s** 精确基线 / **752 tok/s** 稀疏 opt-in，较初版 4.7×）、真双卡同层 MoE EP（+63%）、batched top-k（21.6×）、CPU repack/NUMA 审计（MXFP4 Hybrid 4K +144%）、CPU_REPACK offload 正确性修复已合入，见上方长上下文专节与 `docs/CHANGES.md`。

工程入口：

- **完整改动清单见 [docs/CHANGES.md](docs/CHANGES.md)**（NUMA 体系、CPU 内核格式支持、融合算子、分布式 EP）。
- **构建、测试与分支约定见 [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md)**。
- **1M 上下文内存预算与验收门槛见 [docs/LONG-CONTEXT-1M.md](docs/LONG-CONTEXT-1M.md)**。
- **全部本地参数与生产配方见 [docs/PEOPLESLLM-PARAMS.md](docs/PEOPLESLLM-PARAMS.md)**。

主线跟踪：基于 llama.cpp `e8f19cc0a`（2026-07-16），`vendor` 分支保留基线，定期合并。

## 许可证与版权

本项目是 [llama.cpp](https://github.com/ggml-org/llama.cpp) 的分支。**原项目版权归 ggml-org 及 llama.cpp 全体贡献者所有**，采用 MIT 许可证发布。本分支的全部改动同样以 MIT 许可证发布，并保留原项目的版权声明与许可证全文（见 [LICENSE](LICENSE)）。
