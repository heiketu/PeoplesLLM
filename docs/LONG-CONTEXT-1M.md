# 1M 上下文显存与内存预算

本文规定 PeoplesLLM v2 的长上下文资源边界。目标是模型能够配置并稳定运行
`n_ctx = 1,048,576`，不是用短上下文测速结果推断 1M 配置。

## 基准平台与口径

- 2× RTX 3090 24GB，NVLink `NV4`，每卡空闲显存约 24,112MiB。
- 主机内存 251GiB；从机关闭，不计入可用容量。
- 模型：DeepSeek-V4 Flash Q4_K/MXFP4，GGUF 155,089,895,772 bytes。
- 配置：`-c 1048576 -fa on -ngl 99 -ncmoe 99`，双卡 layer split。
- 数字来自 `llama-fit-params --fit-print on` 的静态分解，单位 MiB：
  `model + context + compute`。它是 OOM 前置门槛，不替代真实运行峰值采样。

## ubatch 上限

F16 KV 的估算如下：

| ubatch | CUDA0 | CUDA1 | 结论 |
|---:|---:|---:|---|
| 8192 | 3,304 + 3,467 + 24,344 = **31,115** | 3,712 + 3,779 + 23,144 = **30,635** | 必然 OOM |
| 4096 | 3,304 + 3,379 + 12,140 = **18,823** | 3,712 + 3,695 + 11,636 = **19,043** | 可用，约 5GiB 余量 |
| 2048 | 3,304 + 3,335 + 6,598 = **13,237** | 3,712 + 3,653 + 5,930 = **13,295** | 安全回退档 |
| 1024 | 3,304 + 3,313 + 3,324 = **9,941** | 3,712 + 3,632 + 3,089 = **10,433** | 保守档 |

结论：1M 默认上限暂定 `-b 4096 -ub 4096`。短上下文上最快的 ubatch 8192
不能用于 1M。运行前要求每卡至少保留 2GiB；低于该余量自动降至 2048，不能靠
CUDA OOM 后重试来做正常控制流。

## KV 量化与预取预算

ubatch 4096 时：

| KV 类型 | CUDA0 总计 | CUDA1 总计 | 相对 F16 节省 |
|---|---:|---:|---:|
| F16 | 18,823 | 19,043 | — |
| Q8_0 | 17,242 | 17,567 | 约 1.5GiB/卡 |
| Q4_0 | 16,399 | 16,645 | 约 2.4GiB/卡 |

多槽 MoE H2D 预取若使用 3 个约 1.14GB 的设备窗口，会额外消耗约 3.4GB/卡。
F16 KV + ubatch 4096 在扣除这部分后无法稳定满足 2GiB 保留线，因此必须满足以下之一：

1. 将 ubatch 降至 2048；
2. 使用已经过正确性与质量验收的 Q8/Q4 KV；
3. 根据实时可用显存减少预取槽数，并无损回退到当前串行 staging。

预取实现不得静态假设 24GB 全部可用，也不得一次分配完整模型专家权重到显存。

## 主机内存边界

静态估算的 host model 为 140,888MiB；ubatch 4096 的 host compute 为 4,322MiB。
实机 `--no-mmap` 加载时曾观测约 145GiB RSS。251GiB 主机可以容纳单份模型，但不能
容纳完整双 NUMA 权重镜像：仅两份 model 估算就约 275GiB，尚未计 KV、compute 和页缓存。

1M 验收阶段禁止默认的全组件 `--numa mirror`。纯 CPU 路径可用
`GGML_NUMA_EP=1 --numa mirror --numa-mirror weights`：EP 会把 routed experts 排除在镜像外，
只复制非专家权重，KV 也不复制。2x38C/251GiB 实机在 1M/Q8 KV 配置下观测
约 165GiB RSS、0 swap，仍有约 78GiB available；更小内存机器必须重新证明峰值安全。
从机恢复前，不把远端内存算入任何预算。

### 双路纯 CPU 验证配方（2026-08-05）

```sh
CUDA_VISIBLE_DEVICES="" \
GGML_NUMA_EP=1 \
GGML_NUMA_HIER_BARRIER=1 \
GGML_NUMA_EP_GATE_UP_PARALLEL=1 \
llama-server \
  -m DeepSeek-V4-Flash-Q4-mxfp4-0731.gguf \
  -c 1048576 -b 512 -ub 512 -t 76 -tb 76 \
  --numa mirror --numa-mirror weights \
  -ngl 0 -nkvo -ctk q8_0 -ctv q8_0 -fa on \
  --no-mmap -dio -np 1
```

`GGML_NUMA_EP_CHUNK` 故意不设置，让 decode 用 16、prefill 用 64。相同模型与输出下，
相对原始 distribute + EP/chunk16 基线的实测如下；TG512 两次复测为 12.37/12.38 tok/s。

| 指标 | 基线 | 优化后 | 变化 |
|---|---:|---:|---:|
| 浅提示 TG | 8.94 tok/s | 12.38 tok/s | +38.5% |
| 4.8k 提示后 TG | 8.48 tok/s | 11.40 tok/s | +34.4% |
| PP4801 | 81.79 tok/s | 86.00 tok/s | +5.1% |

固定逐线程物理核 pin 在同一常驻服务内 A/B 回退约 4%–5%，因此保持当前节点级亲和；
`GGML_NUMA_EP_STATIC` 与 THP 实验开关也保持关闭。

### 双卡 16K 吞吐配方（2026-08-05）

16K 吞吐与真实 1M 容量使用不同配置。16K 档把 context 限制为 32K，以释放显存并让
每张 3090 完整常驻 5 个 routed-expert layers；dense、attention、KV 和输出层仍全部在 CUDA，
其余 33 个 routed-expert layers 使用 CPU_REPACK + 双路 NUMA EP。

```sh
GGML_NUMA_EP=1 \
GGML_NUMA_HIER_BARRIER=1 \
GGML_NUMA_EP_GATE_UP_PARALLEL=1 \
GGML_CUDA_DSV4_KV_REUSE=1 \
llama-server \
  -m DeepSeek-V4-Flash-Q4-mxfp4-0731.gguf \
  -c 32768 -b 512 -ub 512 -t 76 -tb 76 \
  --numa mirror --numa-mirror weights \
  -ngl 99 -ncmoe 99 \
  -ot 'blk\.(17|1[8-9]|2[0-1])\.ffn_(up|gate|down)_exps\.weight=CUDA0,blk\.(38|39|4[0-2])\.ffn_(up|gate|down)_exps\.weight=CUDA1' \
  -ctk q8_0 -ctv q8_0 -fa on --no-mmap -dio -np 1
```

固定 16,384-token prompt、TG512、固定输出 token 的 5 次服务实测为
23.94/24.00/23.56/23.31/24.16 tok/s，均值 23.79、中位数 23.94 tok/s；冷 prompt
为 260.42 tok/s。相同固定输出下，4 层/卡的 5 次均值为 23.58 tok/s，因此第 5 层带来
约 0.93% TG 提升。纯 decode scheduler 计时由 37.93 降至 37.08ms/token，graph splits
由 73 降至 69。部分 offload 不保留：CPU 就地 expert reduction 为 38.70ms/token，额外
offload 4 个 `down_exps` 为 38.89ms/token，分别回退约 2.0% 和 2.5%。

PCM 在一次 TG512（24.32 tok/s）中的系统读/写带宽约 66.9/1.73GB/s；两 socket 读带宽
约 33.8/33.1GB/s，本地命中率 97%/98%。UPI incoming 约 1.8GB/s、每链路约 1%，
outgoing 约 7.5GB/s、每链路 4%–5%。进程 NUMA 内存约 56.25/53.89GiB，显存峰值约
20.95/21.04GiB，均未出现双路失衡或 UPI 饱和。

这组数据确认旧实验路径的 16K TG 问题不在当前生产图中：旧 layer-major/host-stream
路径曾只有 9.55–12.66 tok/s，并出现 89 个 graph splits；当前图的全部 43 层
dense/attention 都在 CUDA，剩余限制是 CPU-MoE 与逐层 CPU/GPU 同步。5 层/卡档仅用于
32K context；不要把它用于 1M 容量验收。真实 1M 档应移除 expert residency override，
按本文件的显存门槛重新验证 KV 峰值。

## 运行纪律与验收

所有大模型进程必须持有：

```sh
flock -x /tmp/xllama-bench.lock <command>
```

1M release gate 至少包含：

1. 启动前记录 RAM、swap、两卡显存和残留模型进程；不满足 2GiB/卡保留线则不启动。
2. 先通过静态 fit，再按 1K → 2K → 4K ubatch 阶梯验证，禁止直接跳到未知大档。
3. 记录加载后、首个 ubatch、稳态和退出前的 RAM/VRAM 峰值；swap 持续增长视为失败。
4. 固定 seed、temperature=0 做非空输出对拍；KV 量化还需单独做长距离质量验收。
5. 性能使用 ABBA 顺序复测；任何吞吐提升都不能以缩短实际 context 或静默降档获得。

## Prefill 执行模式

PeoplesLLM 保留 llama.cpp 的通用 token-major 图作为兼容基线，并按能力与资源选择优化路径：

| 模式 | 适用范围 | 权重行为 | 回退条件 |
|---|---|---|---|
| 通用 chunked | 所有模型、短/中等 prompt | llama.cpp 原调度 | 始终可用 |
| MoE 流式 | 已识别的 host MoE 权重、长 prompt | pinned RAM 异步 H2D，多槽前取 | 后端无异步能力或显存不足 |
| Weight-stationary | 结构已验收的超长 MoE prompt | 当前层权重常驻，按 token tile 扫完整 prompt | 图不支持分层执行或 RAM/IO 预算不足 |

Weight-stationary 的“1M batch”是逻辑批次，不是单次 CUDA kernel。设备端仍以安全 tile
执行；层间 hidden state 由 NUMA 本地 host buffer 保存并通过小型 pinned ring 搬运。
该模式只有在以下不变量都可证明时才能启用：

1. 每个 tile 按位置顺序更新当前层 KV，注意力结果与通用图一致；
2. 当前层的全部 tile 完成后才进入下一层；
3. residual / hyper-connection state 的 host 表示无精度或布局变化；
4. 最终 KV 直接位于 decode 使用的 cache，不发生隐式重建；
5. 任一能力检测失败时，在计算前回退，不能运行中途生成半份 KV 后继续。

这一路径先作为 DeepSeek-V4 的 opt-in 原型验证，再把“分层图能力”抽象到模型接口；
不通过能力检测的架构不会改变现有行为。参考 Lvllm 的可移植原则是阈值分派、pinned
权重、独立 stream/event、有限预取窗口和可配置常驻层，不引入 PyTorch 运行时或固定显存占位。
具体 session、分层图、状态提交和失败清理方案见
[Weight-stationary 超长 Prefill 设计](WEIGHT-STATIONARY-PREFILL.md)。
