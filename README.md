# PeoplesLLM

**让超大 MoE 模型在本地廉价硬件上跑起来。** llama.cpp 特化分支，专注：双路 CPU + 消费级 GPU 上的 200G ~ 3T 级 MoE 推理。

Run giant MoE models (200GB – 3TB) locally on cheap hardware: dual-socket CPUs + consumer GPUs. A specialized llama.cpp fork.

## 实测数据 / Benchmarks

DeepSeek-V4 284B (Q3_K, 94.6GB GGUF)，双路 Xeon 8360Y (Ice Lake) + 2× RTX 3090：

| 实现 | TG (t/s) | PP (t/s) |
|---|---|---|
| 主线 llama.cpp (b10173) | 5.85 | 23.6 |
| **PeoplesLLM (NUMA mirror)** | **~33** | — |
| **PeoplesLLM (NUMA-EP)** | **28.5** | **310** |

GLM-5.2 (UD-Q2_K, 236GB GGUF)：TG 12.0 t/s（EP + GPU 专家卸载）。

## 核心技术 / What's inside

- **NUMA mirror**：非专家权重+KV 双节点复制，线程绑核，UPI 流量归零
- **NUMA-EP**：专家单副本按 socket 放置（mbind 策略级），mul_mat_id 本地优先计算 —— 内存减半，超大模型可装载
- **AVX512/VNNI 8×8 repack 内核**：Q2_K~Q6_K、Q8_0、MXFP4、IQ1_S/IQ1_M 全格式 PP 加速 2-4×
- **融合算子**：dsv4 hyper-connection CUDA kernel、fused MoE router、RMS_NORM 吸收、GLM-DSA Lightning Indexer
- **MTP 投机解码**（dsv4）
- **跨机 EP（开发中）**：激活 dispatch（KB 级流量）而非权重传输，InfiniBand EDR 互联可扩展 CPU MoE 节点，目标 2.8T 级模型

## 状态 / Status

早期开发阶段。`main` 分支 = 生产可用；跨机 EP 传输层（`tools/epd`）已就绪，master 集成进行中。

跟踪主线：基于 llama.cpp `e8f19cc0a` (2026-07-16)，vendor 分支保留 base，定期 merge。

## License

MIT（与上游 llama.cpp 一致）
