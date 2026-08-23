# PeoplesLLM

> **[English README](README.en.md)**

**让 200B-3T 级 MoE 模型在本地服务器上跑起来，并把多路 CPU、消费级 GPU 和多机内存真正叠加起来。**

PeoplesLLM 是面向异构本地推理的 llama.cpp 分支。项目围绕三条相互衔接的主线演进：多 NUMA/多设备执行架构、AVX512 与 GPU 内核、面向计算的 GGUF 权重格式；跨机部分提供独立的专家并行运行时 `tools/epd/`。

## 项目定位

主模型的 attention、dense、router 和 KV 可留在 GPU，routed MoE 专家由双路 CPU 或远程 NUMA worker 执行。目标不是让每个节点各跑一份模型或多个 slot，而是让多个 NUMA 节点共同计算**同一个 slot、同一层的专家任务**。

项目由作者确定架构和硬件方向，在 AI 辅助下完成实现、测试与调优。所有公开性能结论均来自同机、同口径的实测；项目仍处于研究和工程化阶段，生产使用前应在目标硬件上完成输出与容量验收。

主要测试平台：双路 Intel Ice Lake-SP（152 逻辑线程，251 GiB DDR4-3200）+ 2 x RTX 3090 24 GiB（NVLink P2P）以及一台通过 ConnectX-5 100 GbE/RoCEv2 直连的双路 Ice Lake-SP 从机。主要调优模型为 DeepSeek-V4 / DSV4-Flash 和 GLM-5.2。

## 已验证结果

下表中的每一行都是独立测试口径，不能将不同行的百分比直接相乘。

| 路线 | 配对基线 | 当前结果 | 变化 |
|---|---:|---:|---:|
| DSV4 pp2048，MXFP4 -> UDNL_MX | 307.58 tok/s | **384.87 tok/s** | **+25.1%** |
| DSV4 pp2048，同质量 MXFP4 -> E4A | 315.7 tok/s | **370.0 tok/s** | **+17.2%** |
| DSV4 DSpark decode，n2/p0 | 23.9 tok/s | **30.1 tok/s** | **+26%** |
| DSV4 raw decode，热专家分流配对 A/B | 26.0 tok/s | **28.5 tok/s** | **+9.7%** |
| DSV4 16K GPU MoE prefill | 213 tok/s | **334.5 tok/s** | **+57%** |
| GLM-5.2 pp512，2 -> 4 NUMA worker 真 EP | 24.13 tok/s | **40.59 tok/s** | **+68.2%** |
| DSV4 GGUF 体积，MXFP4 -> UDNL_MX | 145.3 GiB | **116.1 GiB** | **-20.1%** |

![三条性能路线的阶段结果](docs/img/evolution-staircase.png)

## 架构发展路线

### 1. NUMA 塌方：从测量开始

双路机器上的朴素 interleave 会让大量权重读取跨 UPI，同时所有线程在 barrier 上等待最慢者。实测内存流式读取为 **122 GB/s（interleave）对 313.5 GB/s（本地页 + 本地绑核）**，相差 2.57 x。MoE decode 恰好是短 burst、低算术强度的带宽型负载，因此普通的“多开线程”并不能解决问题。

### 2. Mirror：先消除跨路读取

`--numa mirror` 让每个 socket 持有一份完整权重副本，并把线程和页面绑定到同一节点。它将跨 UPI 的专家权重流量降为零，DSV4 tg512 达到 30.35 tok/s，对同机上游 distribute 的 26.9-28.0 tok/s 提升约 8%-13%。代价是专家权重占用双倍内存。

### 3. NUMA EP：内存减半，但静态专家归属不够

下一步把每个专家只放在一个节点。专家权重内存减半，pp512 与 mirror 基本持平（343-348 对 338-347 tok/s）；但 batch=1 时，一个专家是一条不可继续切分的 GEMV，静态“8 个激活专家分进 2 个节点”会产生尾部气泡，tg512 只有 23.9-24.2 tok/s。这个阶段给出了关键约束：**专家放置必须保留细粒度并行和动态调度自由度。**

### 4. NUMA 行窗 TP + DME

最终将每个专家平面按 128 行窗口交替放到两个节点，执行时再以 64 行 claim 量子动态领取任务。每个专家都由两路共同读取，输出行只有一个写入者，不需要跨节点归并。环境变量仍沿用 `GGML_NUMA_EP=1`，但执行语义已经是专家内部的 row-window tensor parallelism。

![NUMA 行窗 TP 开关 A/B](docs/img/numa-tp-onoff.png)

| 测试形态 | 关闭 | 开启 | 变化 |
|---|---:|---:|---:|
| 混合 tg512 | 16.59 | **25.01** | **+51%** |
| 混合 pp2048 | 267.05 | **298.98** | +12% |
| 纯 CPU tg128 | 7.23 | **11.65** | **+61%** |
| 纯 CPU pp512 | 101.41 | **107.91** | +6.4% |

在行窗 TP 之上，DME（Dynamic Matrix Execution）继续处理节点内的不规则 MoE 形状：

- claim 量子从 16 调到 64 行，微基准带宽由 145 提升到 165-179 GB/s；
- 按 `nrows` 和 batch 形状在 GEMV/GEMM 之间分派，并把 UDNL 同专家的 2-8 行尾部任务合并；
- 修正线程绑核与行窗归属不一致的问题，`MUL_MAT_ID` 从 155.7 降到 88.4 us/call；
- 两级 NUMA barrier 和重复专家感知调度降低尾部等待。

### 5. GPU 混合执行

在生产型混合路径中，GPU 处理 attention、dense、router 和 KV，CPU 处理 routed experts。针对不同阶段形成了三种互补机制：

- **热专家驻留**：重新采样后的 top-16 专家/层覆盖 49.1% 的选择，43 层紧凑 MXFP4 权重约 8.8 GiB。GPU 热分支与 CPU 冷分支在图内 fork/join，配对 tg512 从 26.1/25.9 提升到 28.4/28.6 tok/s，均值提升 9.7%。
- **NVLink P2P TP**：捕获安全的 P2P allreduce 和异步 D2H 回读把 TP 路径自身从 13.6 提升到 20.3 tok/s（+49%），每 token 的 `cudaStreamSynchronize` 约从 690 次降到 150 次。该数字是 TP 路径内部 A/B，不等于单卡最优路径。
- **MoE 流式 prefill**：专家权重保留在 host pinned memory，长 prompt 时按层上传、跨 tile 复用，并用双卡 expert-axis EP 和 3-slot 预取隐藏 H2D。

![GPU 流式 MoE prefill](docs/img/gpu-prefill-streaming.png)

| 16K prompt | tok/s |
|---|---:|
| chunked token-major | 213 |
| layer-major + pipe + device-HC | 248.5 |
| + 双卡 EP + `PREFETCH=2` | **334.5** |

### 6. 多机 EP：多个 NUMA worker 共同算一个 slot

`tools/epd/` 只传输 KB 级激活和结果，不在请求期间搬运专家权重。master 和 slave 的每个 NUMA node 都可以成为独立 worker；strict pure EP（`SCHED_KLOCAL=0`）下，master 不需要保留 routed experts 的完整副本。运行时根据真实 router 频率构建带热点副本的 holder map，再按在飞工作量动态派单。

![多机专家并行](docs/img/remote-ep.png)

| 测试 | 之前 | 之后 | 变化 |
|---|---:|---:|---:|
| GLM-5.2 MoE pp512，2 -> 4 worker | 24.13 | **40.59** | **1.682 x** |
| DSV4 16K PP，UB64 -> UB256 | 235.37 | **269.36** | +14.4% |
| 64 B RPC RTT，TCP -> RoCEv2 | 42-74 us | **10-13 us** | 约 4-6 x 降低 |

GLM 的 dense/attention 仍在 GPU；上表第一行只扩展 CPU MoE。75 层 MoE decode 合计由 86.71 降到 69.58 ms/token（1.246 x），128-token 输出逐字节一致。DSV4 四 worker pure EP + RDMA 的固定验收均值为 37.921 tok/s，峰值 38.423 tok/s，输出 SHA256 与单机参考一致。

## 内核发展路线

### AVX512 repack：先让所有 MoE 分支进入批量内核

早期审查发现 `mul_mat_id` 的多个分支仍逐行调用 `vec_dot`，大 batch 没有进入 GEMM；CUDA 构建中的 buffer-type 顺序还可能让 CPU 权重落入不支持 repack 的 pinned buffer。项目补齐 Q2_K-Q6_K、Q8_0、MXFP4 和 IQ 系列的 8 x 8 repack traits、加载期重排、GEMV/GEMM 分流与 EPD worker 路径。

### arec panel-stationary：权重只从 DRAM 流过一次

arec（activation record）把 Q8 激活和 scale 预计算一次，随后让权重 panel 驻留 L1/L2，并跨 8 行 tile 复用。端到端 pp2048：

- UDNL_W4：263.33 -> **366.58 tok/s（+39%）**；
- UDNL_MX：223.02 -> **384.87 tok/s（+73%）**。

### 融合和异步边界

- DSV4 router 的 5-kernel 链融合为单 warp kernel，小行 indexer top-k 合并为单 radix kernel；GPU busy 18.76 -> 18.23 ms/token，tg 25.15 -> 25.91；
- pinned staging + event drain 将 CPU input readback 从 13 降到 4.6 ms/token；
- MXFP4 repack GEMV 软件预取带来 +2.6% TG；
- 64 行 claim 与 UDNL 尾行批量化把 DSpark n2/p0 从 26.50 提升到 30.10 tok/s，同时保持接受率和输出一致。

## 格式发展路线

### “Q2_K 为什么比预期慢”的起点

最初一个名为 `Q2_K` 的 90.9 GiB 模型只有 223.60 pp2048 和 22.87 tg512。它的 TG 反而低于更大的 Q3_R（25.30 tok/s），虽然 PP 高于尚未充分优化 GEMM 的 Q3_R（174.60 tok/s）。读取 GGUF metadata 后又发现它实际是 IQ2_XXS（2.0625 bpw），并非 Q2_K。这个“Q2_K < Q3_K”现象最终被拆成格式标注和算子效率两个问题：文件更小不代表推理更快，码本查找、scale 展开、反量化长度以及是否能进入批量 GEMM，都会比名义 bpw 更早成为瓶颈。

于是格式设计从“最少比特”转向“存储布局服从内核访问顺序”：

- **UDNL_W4**：定长块，码本结果直接进入 AVX512 VNNI；
- **UDNL_MX**：用 imatrix 在 W2/W3/W4 之间混合分配，再复用同一 arec panel 内核；
- **E4A**：保持 MXFP4 数值 bit-exact，并让 row-block 的 nibble pairing 对齐 NR16 内核；加载时只做无算术的 byte-only panelization，不解码或重新量化。

![MoE 权重格式的体积、速度和质量](docs/img/quant-formats.png)

| 格式 | 体积 | pp2048 | tg512 | WikiText-2 PPL | 定位 |
|---|---:|---:|---:|---:|---|
| MXFP4 | 145.3 GiB | 307.58 | 26.26 | 3.5830 | 质量基线 |
| **UDNL_W4 + arec** | 146.4 GiB | 366.58 | 24.99 | 3.7997 | 固定 4-bit 计算亲和布局 |
| **UDNL_MX + arec** | **116.1 GiB** | **384.87** | 26.38 | 4.6274 | 最小体积和最高 PP |
| **E4A** | 约 147.2 GiB | **370.0** | 26.37 | ≈ MXFP4 | MXFP4 值 bit-exact，加载约 22 -> 11 s |
| Q3_R（MoE） | 114.1 GiB | 174.60 | 25.30 | 4.7636 | 小体积对照 |
| IQ2_XXS（文件名为 Q2_K） | 90.9 GiB | 223.60 | 22.87 | 6.6338 | 反量化成本对照 |

如果优先容量和 PP，选择 UDNL_MX；如果要求保持 MXFP4 数值与质量，选择 E4A。不同格式行来自固定 harness，但 E4A 的质量复验采用 5-chunk PPL；完整口径见 [BENCHMARKS](docs/BENCHMARKS.md)。

## 构建与使用

```bash
# CUDA 混合构建
cmake -B build-cuda -DGGML_CUDA=ON
cmake --build build-cuda -j

# 纯 CPU 构建
cmake -B build-cpu -DGGML_CUDA=OFF
cmake --build build-cpu -j

# DSV4 单机混合：GPU 跑 dense/attention/KV，双路 CPU 跑 routed experts
GGML_NUMA_EP=1 build-cuda/bin/llama-server \
  -m /path/to/model.gguf -ngl 99 -ncmoe 99 \
  -t 72 --threads-batch 72 --numa distribute -fa on -b 4096 -ub 1024
```

`tools/peoplesllm-run.sh` 提供 `dsv4-prod`、`dsv4-dual`、`glm-dual` 和 `cpu-pure` profile。双机 worker、RDMA/TCP 回退、专家 map 和完整参数请参阅 [快速开始](docs/QUICKSTART.md)、[参数手册](docs/PARAMETERS.md) 与 [EPD 文档](tools/epd/README.md)。线程数和 ubatch 必须按目标 CPU、内存通道、显存和 prompt 长度重新标定。

## 测量与正确性

- 性能 A/B 使用同一模型、prompt、线程与 offload 配置；关键结论用 ABBA/反序重复，避免约 25% 的运行顺序效应。
- 保持归并顺序的路径使用固定 seed、贪婪解码的逐字节输出或 SHA256 对拍。
- 跨设备归并可能因浮点加法顺序在近似并列 logit 上产生单 token 翻转，因此另做 run-to-run 确定性和接受率检查；不会把这种路径写成 bit-exact。
- 图中和表中的 `pp`、`tg` 是 llama.cpp benchmark 口径，不代表任意并发数、上下文长度或硬件上的 SLA。

完整实测数据和口径见 [docs/BENCHMARKS.md](docs/BENCHMARKS.md)，技术改动见 [docs/CHANGES.md](docs/CHANGES.md)。

## 文档

- [快速开始](docs/QUICKSTART.md)：构建、单机/双机配方与常见问题
- [参数手册](docs/PARAMETERS.md)：环境变量、CLI 参数和 profile
- [性能档案](docs/BENCHMARKS.md)：详细 A/B 数据和图表
- [改动清单](docs/CHANGES.md)：NUMA、内核、GPU 与 EP 改动
- [双机 EP](tools/epd/README.md)：worker、传输和部署

## 许可证

本项目是 [llama.cpp](https://github.com/ggml-org/llama.cpp) 的分支。原项目版权归 ggml-org 及 llama.cpp 贡献者所有，并采用 MIT 许可证发布。本分支的改动同样使用 MIT 许可证，保留原版权声明与 [LICENSE](LICENSE)。
