# 性能实测与进展档案

> 本文档收录详细 benchmark 数据与迭代进展。项目优势与特色概览见 [README.md](../README.md)；复现口径见 [CHANGES.md](CHANGES.md) 与 [benchmarks/](benchmarks/)。

**Decode 口径约定**：所有 TG/decode 数字逐项标为 `raw/no-DSpark` 或 `DSpark speculative`。PP 是 prefill，不包含 draft 接受收益。两种 decode 口径不得相加或互作基线。GPU-hot + CPU-remote-EP 目前只在 raw/no-DSpark、单 slot、四 worker strict cover、同步 REQ4、非 MAX_EFFORT 下通过，不能把该结果外推到 DSpark/多 slot，也不能与两条独立路径的百分比相乘。

## DSV4-Flash 全格式统一矩阵（2026-08-24，`performance`）

表中 16 个原矩阵点使用同一二进制、单机混合配置和固定命令；UDNL_MX corrected 在修复后用同硬件、参数和 `performance` 配方补测，替换旧污染行。固定参数为 `-ngl 99 -ncmoe 99 -fa 1 -dev CUDA0 -sm layer -t 72 --numa distribute -b 4096 -ub 1024 --load-mode none -p 2048 -n 512 -r 3`。运行时 152/152 CPU 的 governor/EPP 均为 `performance`，turbo 开启，`numa_balancing=1` 已记录。这是单 slot raw 口径，未启用 DSpark 或热专家。

| 格式 | 体积 GiB | pp2048 | tg512 raw/no-DSpark | WikiText-2 PPL |
|---|---:|---:|---:|---:|
| UD-IQ1_S | 76.87 | 251.18 | 29.45 | 7.7533 |
| UD-IQ1_M | 80.93 | 241.38 | 29.03 | 6.8899 |
| UD-IQ2_XXS | 84.62 | 247.17 | 28.89 | 5.9844 |
| UD-IQ2_M | 84.68 | 246.83 | 29.24 | 6.0004 |
| UD-Q2_K_XL | 90.18 | 232.22 | 28.03 | 5.4675 |
| UD-IQ3_XXS | 97.05 | 208.96 | 26.74 | 4.7101 |
| UD-IQ3_S | 108.10 | 254.48 | 23.99 | 4.5707 |
| **UDNL_MX corrected**[^udnl-mx-corrected] | **116.13** | **418.95±3.50** | **26.90±0.25** | 4.6047±0.1782 |
| UD-Q3_K_M | 119.28 | 207.20 | 25.49 | 4.0242 |
| UD-Q3_K_XL | 119.40 | 207.69 | 25.34 | 4.0189 |
| UD-IQ4_XS | 127.28 | 215.93 | 21.10 | 3.8633 |
| UD-IQ4_NL | 127.28 | 223.53 | 21.36 | 3.8633 |
| UD-Q4_K_XL[^q4-loader] | 144.44 | 314.15 | 25.65 | 3.4363 |
| MXFP4 | 145.26 | 312.48 | **26.87** | 3.5830 |
| **E4A** | 145.26 | **362.92** | 24.83 | 3.5830 |
| **UDNL_W4** | 146.36 | **370.87** | 25.10 | 3.7997 |
| UD-Q8_K_XL | 150.75 | 306.17 | 21.39 | 3.4440 |

UDNL_MX corrected 在修复 imatrix 契约后从确认的源重新量化，随后用完全相同的 `performance` 配方补测三轮 PP/TG；PPL 使用 20×512 WikiText-2 历史口径。其余 PPL 沿用同一权重此前的质量测量；CPU 频率不会影响 PPL。`UD-IQ4_XS` 的正式值按重复次数合并：原 3 轮为 219.09/21.62，补充 5 轮为 214.03/20.78，加权后 PP/TG 为 **215.93/21.10**。

公开图中，普通 UD 点用圆形，MXFP4 用作锚点，自研 UDNL_W4、UDNL_MX corrected、E4A 用大星。红色空心菱形表示该格式虽然更小，却被至少一个更大格式以 3% 以上速度反超。趋势线只拟合普通 UD 点与 MXFP4 锚点，不用自研格式参与拟合。单独的 tradeoff 图把 UDNL_MX 与 Q3_K_XL 的体积、PP、TG 和 PPL 并列，避免只看速度隐藏质量代价。

[^q4-loader]: 当前 loader 实际把 `UD-Q4_K_XL` 的 routed MoE tensor 识别为 MXFP4；该点保留为实测文件结果，但不能作为纯 Q4 专家端点。
[^udnl-mx-corrected]: 旧 UDNL_MX 文件由错误的 imatrix 路径生成，其质量与 mode 分布已永久撤回。当前行来自修复后重新量化的新文件；完整 raw 结果为 116.1284 GiB、PP2048 418.95±3.50、TG512 26.90±0.25、PPL 4.6047±0.17815。

## 2026-08-24 strict hot-expert 与量化路线收口

### UDNL_MX corrected raw/no-DSpark endpoint

| 对比 | 体积 | PP2048 | TG512 raw/no-DSpark | PPL |
|---|---:|---:|---:|---:|
| UDNL_MX corrected | **116.13 GiB** | **418.95±3.50** | **26.90±0.25** | 4.6047±0.1782 |
| UD-Q3_K_XL | 119.40 GiB | 207.69 | 25.34 | **4.0189** |

UDNL_MX corrected 比 Q3_K_XL 小约 2.7%，PP 约 2.0×、TG 高约 6.2%，但 PPL 高 **14.6%**。因此它是容量与 PP 的研究点，**不是推荐的 Q3_K_XL 替代格式**。

![UDNL_MX corrected 与 Q3_K_XL 的权衡](img/udnl-mx-tradeoff.png)

### UDNL_MX corrected + MXFP4 K24 strict-hot raw/no-DSpark pilot

| 指标 | UDNL_MX no-hot | UDNL_MX + K24 hot | 变化 |
|---|---:|---:|---:|
| TG512 raw/no-DSpark（单组 paired pilot） | 26.9 tok/s | **31.3 tok/s** | **+16.36%** |
| PPL（同命令 b1/ub1，5 chunks） | 3.5614 | **3.5218** | -1.1119% |
| 权重驻留 | 116.1284 GiB | **128.9780 GiB** | +12.8496 GiB hot weights |

strict-hot 改善了 UDNL_MX 自身的 paired PPL 和 TG，但没有证明追平 Q3。Q3_K_XL 的旧 20-chunk PPL 为 4.0189，其中 checkpoint `[5]=3.0808`；它与本次 b1/ub1 paired 命令不同，不能直接作为质量门。即使仅作提示，hot B=3.5218 仍比该旧 `[5]` 高 14.31%，而组合权重驻留又比 Q3_K_XL 大 8.02%。因此该结果只作为速度/负面质量消融，不进入 quality headline；需要同命令的 Q3 20-chunk 对照才能重新判断。

### E4A + MXFP4 K24 hot-expert（单 slot，raw/no-DSpark）

最终 strict 路径让 GPU 回传六个 router-slot 输出，CPU 按 slot 0→5 恢复 baseline 左折叠；AVX512 合并保持逐 slot 独立乘加、不使用 FMA。12 次 TG 使用 3 组 ABBA，PP 为同条件 A/B，PPL 为 batch=1、5 chunks；43 层 marker、GPU 轨迹与固定 seed 响应重复性全部通过。

| 指标 | E4A no-hot | E4A + K24 hot | 变化 | 判定 |
|---|---:|---:|---:|---|
| TG512 raw/no-DSpark（6 轮均值） | 25.02 tok/s | **30.23 tok/s** | **+20.85%** | ≥27.70，通过 |
| PP2048 | **366.85 tok/s** | 361.47 tok/s | -1.47% | ≥98% A 且 ≥350，通过 |
| WikiText-2 PPL（batch=1，5 chunks） | 2.7758 | **2.7548** | -0.76% | ≤1.003×A，通过 |

这组结果证明的是同一 E4A cold 模型在单 slot 下的 strict hybrid 路径；不能外推到多 slot，因为当前 hot staging/event 是每层单份状态。

UDNL_W4 + MXFP4 K24 只完成一组 raw/no-DSpark 负面 pilot：TG 23.5→30.5 tok/s，但 PPL 2.9559→2.9719（+0.54%），超过 0.5% 门。W4 hot 会把命中的 UDNL_W4 专家替换为数值不同的 MXFP4 权重，因此该结果仅作为质量负面消融，不升级为正式配置或里程碑。

### DSV4 matched 2→4 pure EP（raw/no-DSpark）

同一 MXFP4 模型、master 命令、CUDA placement、prompt 和 512-token 输出下，master 设置 `KLOCAL=0`，CPU routed experts 由 worker 做互斥 modulo cover；这是一个 slot 的 expert parallelism，不是四份模型或四个 slot。

| topology | 31-token prompt throughput | TG512 raw/no-DSpark | 2→4 变化 |
|---|---:|---:|---:|
| 单机 2 NUMA worker | 64.3 tok/s | 22.1 tok/s | — |
| 双机 4 NUMA worker | **75.6 tok/s** | **25.2 tok/s** | prompt +17.57%，TG +14.03% |

两种拓扑的完整生成响应一致。该数据是一次 matched topology 对比，不声称理想 2×，也不与下文 GLM-5.2 的 PP512 workload 混算。

### DSV4 GPU-hot + CPU-remote-EP strict bridge（raw/no-DSpark）

这是联合路径的首次端到端验收，不是把两个历史百分比相加。A、B 使用同一 MXFP4 cold 模型、prompt、seed、二进制和四路 NUMA worker；`KLOCAL=0` 关闭 master CPU routed-expert 副本。A 让全部 slot 进入 CPU remote EP，B 在 CUDA1 驻留每层 K24 热专家，只把 cold slot 通过同步 REQ4 发给四路 strict modulo cover。测试为单 slot，未启用 DSpark 或 MAX_EFFORT。

TG512 顺序为 ABBA 后接 BAAB：

| 模式 | 四个有效样本，tok/s | 均值 | 标准差 |
|---|---|---:|---:|
| A：四路 CPU remote-only | 24.9 / 25.2 / 25.5 / 25.4 | 25.25 | 0.265 |
| B：CUDA1 K24 hot + 四路 CPU remote cold | 29.6 / 28.3 / 28.6 / 28.9 | **28.85** | 0.557 |

端到端提升为 **+14.26%**。同命令 WikiText-2 b1/ub1、5 chunks 的 PPL 为 A 2.7647、B **2.7412（-0.85%）**。A 的四轮生成 hash 内部稳定，B 的四轮也内部稳定，但 A/B hash 不同，因为 GPU 与 CPU expert kernel 不是 bit-identical；PPL 改善是 within-command 质量门，不是逐值一致声明。所有 B 轮均记录 43/43 hot-fork 和 43/43 remote-bridge marker。

下表来自额外的八 token verbose 结构诊断，共 301 次单 token MoE 调用；它**不是性能运行**，因此仅用于证明 cold-only 派发、四端点覆盖和 strict merge 的结构行为：

| 结构指标 | remote-only A | hot + remote B | 变化/说明 |
|---|---:|---:|---|
| CPU assignment | 1,806 | **916** | -49.28% |
| 四端点 assignment | — | 239 / 220 / 213 / 244 | 聚合工作保持分散 |
| 全四路 fanout 调用 | 117 / 301 | **18 / 301** | hot slot 不发给 worker |
| remote wait mean | 0.444 ms | **0.310 ms** | 仅结构诊断 |
| mixed merge mean | 0.0070 ms | 0.0296 ms | B 恢复严格的六 slot GPU/CPU 左折叠 |

当前限制是 `n_seq_max=1`、KLOCAL=0、non-pipe 同步调度和 weight-on-master REQ4；每层只有一份 staging/event，原子 in-flight gate 会拒绝并发使用。多 slot 需要 per-context hot buffer/event，MAX_EFFORT 需要额外副本与负载门，DSpark 需要独立验收。

### Q2/Q3/Q4 proxy 与物化门禁状态

新 Q2/Q3/Q4 首模尚未生成，因而没有 PPL、raw TG 或 DSpark TG。Phase A v1 的 129-tensor gate/up 原子计划为 `Q2_K/Q3_K/Q4_K = 61/58/10`，dry-run **108.4046 GiB**、large-scan proxy loss **0.898640× all-Q3**、expert traffic **0.903594× all-Q3**。完整物化在 `blk.21.ffn_gate_exps.weight` 的 Q3 fp16 super-scale 变为 `inf` 时被验证门拒绝，没有输出模型；这不是 PPL 失败。

随后对 129 个 expert tensor 做全量只读 MXFP4 representability 扫描，只有 `blk.21` 的 gate/up 两个 tensor 含异常范围，可能让 Q2_K/Q3_K/Q4_K 的 fp16 super-scale 溢出。v2 不 clamp、不重解释源值，而是把完整 gate/up 原子对保留为 MXFP4；类型计数为 `Q2_K/Q3_K/Q4_K/MXFP4 = 61/56/10/2`：

| 计划 | Q2_K / Q3_K / Q4_K / MXFP4 | dry-run | proxy loss / all-Q3 | expert traffic / all-Q3 | 状态 |
|---|---:|---:|---:|---:|---|
| v1 | 61 / 58 / 10 / 0 | 108.4046 GiB | 0.898640 | 0.903594 | 物化被 scale-inf 门拒绝，无模型 |
| v2 | 61 / 56 / 10 / 2 | **108.8109 GiB** | 0.888917 | 0.907259 | dry-run 通过，尚未转换 |

这些仍只是 reconstruction/traffic 代理与 quantizer 容量结果，不是模型质量或速度。即使 v2 未来成功生成，也必须先通过 PPL、确定性和完整输出门，才能测 raw TG；DSpark 需另行测量。

<details>
<summary>历史矩阵：2026-08-21，<code>powersave</code>（仅供追溯）</summary>

同一单机混合配置：双 RTX 3090 执行 attention/KV，双路 Ice Lake CPU 执行全部 routed experts，`-t 72 -fa 1 -b 4096 -ub 1024 --load-mode none`，测试 `pp2048/tg512 raw/no-DSpark`。下列 16 行均采集于 CPU governor 切换前，只用于追溯格式异常，不能继续作为当前排名或 README 主图的解释口径。

| 格式 | 体积 GiB | pp2048 | tg512 raw/no-DSpark | PPL（20 x 512） |
|---|---:|---:|---:|---:|
| UD-IQ1_S | 76.9 | 256.86 | 27.94 | 7.7533 |
| UD-IQ1_M | 80.9 | 242.84 | 26.44 | 6.8899 |
| UD-IQ2_XXS | 84.6 | 251.75 | 27.17 | 5.9844 |
| UD-IQ2_M | 84.7 | 244.99 | 26.60 | 6.0004 |
| UD-Q2_K_XL | 90.2 | 236.31 | 26.56 | 5.4675 |
| UD-IQ3_XXS | 97.1 | 209.67 | 24.77 | 4.7101 |
| UD-IQ3_S | 108.1 | 265.20 | 22.97 | 4.5707 |
| mix-IQ2XS/IQ3XXS/MXFP4 | 112.1 | 233.71 | 25.14 | 4.2027 |
| UD-Q3_K_M | 119.3 | 209.77 | 24.05 | 4.0242 |
| UD-Q3_K_XL | 119.4 | 209.62 | 23.92 | 4.0189 |
| UD-IQ4_XS | 127.3 | 226.92 | 21.05 | 3.8633 |
| UD-IQ4_NL | 127.3 | 229.91 | 20.74 | 3.8633 |
| UD-Q4_K_XL | 144.4 | 318.49 | 25.17 | 3.4363 |
| mix-MXFP4-MoE/Q8 | 145.6 | 313.28 | 24.49 | 3.4295 |
| UD-Q8_K_XL | 150.8 | 307.48 | 20.78 | 3.4440 |
| mix-MXFP4-MoE/BF16 | 150.8 | 306.81 | 20.59 | 3.4440 |

</details>

## 最新进展（2026-08-06~13）

- **PP repeat-affinity 小幅提速并进入生产配置**：PP dealer 现在可让同一批次里重复命中的热点专家尽量留在同一 holder，减少重复权重流；`GGML_REMOTE_EP_SCHED_PP_REPEAT_COST=250` 对默认 1000 的同组 6 轮 A/B 为 263.555 vs 261.108 tok/s，**+0.94%**。默认值仍是 1000 以保持兼容，当前 DSV4 生产显式设 250。
- **MXFP4 5–8 行 shared-weight 内核只保留能力，不进入生产热路**：AVX512 内核和 NR2..8 bit-exact 回归均通过；但真实 expert-first 派发从 NR2..4 放宽到 NR2..8 后，排除首个冷请求的热态均值 267.128，对配对旧路径 268.443 tok/s 为 **-0.49%**。因此 expert-first 已恢复只合并 2–4 行，不能把早先约 +0.9% 的非配对波动当成收益。
- **DSpark + 远程 CPU EP 的 layer-major 已验证为负收益并默认关闭**：所有 decode 数字均为 DSpark speculative；普通 UB256 为 263.17 PP / 37.70 TG，layer-major host-HC 为 185.97 / 29.71，device-HC 为 259.09 / 30.15 tok/s，整套 `-b/-ub 2048` device-HC 也只有 228.51 / 21.77。原因是 GPU layer-major 权重驻留无法减少远端 CPU 专家权重流，反而增加 HC 边界搬运并受 raw-SWA decode 状态约束。框架入口保留给 GPU-local/未来 HBM 实验，`LLAMA_LAYER_MAJOR_SPECULATIVE` 默认关，生产未设置。
- **1M×1 的 PP 默认从 UB64 升到 UB256**：同一 F16 KV、NMAX=3、四路 KLOCAL=0 EP 下，16K 独立长提示三轮 PP 为 270.01/267.12/270.95 tok/s，均值 **269.36 tok/s**；旧 UB64 PP 三轮均值 235.37，提升 **14.4%**。UB2048 PP 虽达到 298.20 tok/s，但同配置 DSpark speculative TG 热态下降约 2.8%，故生产选择 UB256。UB4096/8192 在 1M 配置下因 GPU0 compute buffer OOM 被正确拒绝，而不是再被 RDMA 12MiB 接收环误拦。
- **EP 拓扑热路缓存与重连一致性**：pure EP 在建图时一次生成 dealer holder 表，不再每个 MoE 调用重扫四份 CAP bitmap；固定请求 2386 次调用的 `deal+send` 累计 177.6→172.1ms（-3.1%），DSpark speculative 复测后四轮均值 36.84 tok/s，输出 SHA256 与 draft/accepted 145/78 不变。CAP/稀疏 bitmap 已拆到独立可测试模块，worker 重连若 expert map、kernel ID 或能力位变化会拒绝继续执行。
- **DSV4 单 slot 四 NUMA 真 EP 的 DSpark speculative TG 达到 37.921 tok/s**：43 层 routed MoE 由 master/slave 各两个 NUMA worker 共同计算，target dense/attention/router 与 DSpark draft 在双 RTX 3090；NMAX=3、四 worker `-t 36`、e128 CPU-worker 热点副本图（不是 GPU-hot）、真 RDMA、仅 master CQ spin 的热态六轮为 38.016/37.429/37.873/37.695/38.423/38.092 tok/s，输出 SHA256 全部一致。worker 也 spin 的 DSpark mean 37.659，已否决。
- **Q8 KV 容量档达到每槽完整 1M × 3**：target 43 层与 target KV 连续放 CUDA0，DSpark 与 draft KV 放 CUDA1，共用 embedding/output 也放 CUDA1；GPU0/1 实占 20,456/12,304 MiB，三个 slot 均为 1,048,576。标准 128-token DSpark speculative 请求热态六轮均值 34.260 tok/s，只比同布局 `1M×2` 的 34.436 低 0.51%；`1M×4` 在 CUDA0 申请 2.73GiB PP compute buffer 时真实 OOM。三槽并发 DSpark aggregate 约 41.6 tok/s，并不高于旧 F16 双槽的 42.998，因此第三槽是容量档而非总吞吐优化，不是当前 F16 默认。单槽固定请求 7/7 输出一致，三个独立算术 chat 顺序/并发均返回 2/4/6；raw continuation 的近似并列 logits 会随合批路径分叉，不能拿 raw 字节哈希误判 slot 污染。
- **AVX512 MXFP4 重复专家路径**：同一次请求 2/3/4 个 assignment 命中同一专家时复用权重流，真实 worker 微门 gate/up 约 +24%、down 约 +22%；请求紧凑 gather 与 repeat-aware dealer 继续降低小 batch 固定开销。
- **EP 紧凑激活量化并行修正**：REQ2/REQ4 把 assignment 放在张量 `ne12` 维，而旧 CPU_REPACK 只按 `ne11` 分线程；生产形状 `ne11=1` 时 F32→Q8 激活量化因此长期只有线程 0 工作。现在按完整 `(ne12,ne11)` 行空间分摊，单行 TG 行为不变。六行微基准 gate/up（K=7168）由 31~33 µs 降到 19~20 µs、down（K=2048）由 17~20 µs 降到约 15 µs；两次部署 DSpark speculative 复测六轮均值为 36.573/36.902 tok/s，对旧 worker 36.280 为约 +0.8~1.7%，输出与 draft 接受数不变。
- **e144 副本图没有升级为生产默认**：DSpark speculative 下每路 144 个专家的两组六轮均值为 37.075/36.843 tok/s，12 轮总均值 36.959；相邻 e128 六轮 36.902，仅 +0.16%，低于抖动且四路内存峰值升到约 97.6/99.9/96.5/96.8G，故恢复 e128，也不再浪费时间测试 e160。
- **pure EP worker 重启不再立即打崩 master**：`GGML_REMOTE_EP_RECONNECT_TIMEOUT_MS` 允许 SCHED 在断连后等待 worker 重新监听并重发暂存请求，健康热路径无额外工作。生产设为 90000 ms；真实重启 slave w2 后，请求等待约 66 秒并正确完成，正文 hash 与 draft 接受数不变，master PID 未变、`NRestarts=0`。systemd unit 另设 `Restart=on-failure` 作为最终兜底。

- **AVX512 EPD 路径完整化**：修复零字节 CPU_REPACK trait 探测导致真实模型静默退回 raw kernel 的问题；兼容的 separate gate/up 在加载期融合，mmap 原始页在每个张量转换完成后立即回收，线程标定改测每路实际紧凑 assignment。GLM-5.2 layer 3、64 专家受控 A/B 中，warm REQ2 PP128 从 33.14~36.73 ms 降至 31.53~31.60 ms，worker 内 raw/no-DSpark TG kernel 从 0.505~0.595 ms 降至 0.460~0.477 ms；输出一致。该数据仅代表 CPU MoE worker 层内计算，不混入 GPU dense/attention。
- **异构 worker 动态负载反馈**：热点副本 dealer 不再假定四个 NUMA 等速；它分别学习每路 TG/PP 的空队列服务时间，并用在飞虚拟工作量派单。纯策略、溢出与 dealer 回归已通过；该改动尚未计入下述 40.59 tok/s，需在四路安静窗口复测。
- **单 slot 四 NUMA 真 EP（GLM-5.2）**：master 与 slave 的每个 NUMA node 各一个 worker，`KLOCAL=0` 下 master 不复制专家权重；ragged 分配紧凑执行、PP 图缓存分层、真实 PP 频率画像 + 76 份热点副本后，PP512 从 2 NUMA 24.13 提升到 4 NUMA 40.59 tok/s（+68.2%，延迟 -40.6%），128-token 输出逐字节 MATCH。
- **raw-SWA decode ring 默认启用**（分支 fa-decode-fix，raw/no-DSpark）：q1 decode 图宽与 prompt 长度解耦，fixed TG64 8.894→11.837/12.085 tok/s（+33~36%）、TG512 +4~8%；多 slot 场景自动回退全宽语义。
- **q8 compact 稀疏 FA 默认启用**（raw/no-DSpark）：q8_0 KV 下 q1 decode 只物化 top-k 选中行，TG64 9.75→10.69（+10%）、TG512 +1.3%，成为最快 q1 decode 路径（超 f16 dense）；f16 fused sparse 保持 opt-in（16K 实测 -12%，长上下文再评估）。
- **多流（小 q）稀疏 FA**（raw/no-DSpark，opt-in，`LLAMA_DSV4_FUSED_INDEXED_FA=3/4`、`LLAMA_DSV4_Q8_SPARSE_FA=2`）：多 slot 并发 decode 的稀疏扩展，正确性三方 byte-identical 验证；≤16K 性能持平（权重带宽主导），收益主场在 256K+ 长上下文。
- **GPU 流式 prefill（方案A，server 集成）**：长 prompt 整 tile 路径 PP 63→127.6 tok/s（2×）；紧凑环形 SWA 缓存下按资格门自动回落 chunked，`--swa-full` 可启用；指纹四组逐位命中。
- **IQ2_XXS AVX512 kernel**：gemv 微基准 **17×**；raw/no-DSpark 全模型 A/B 持平（3.78→3.71，瓶颈在专家 mul_mat_id 未覆盖）——微基准提升、模型级收益待扩展。

## 性能实测

### 对标主线 llama.cpp（同机同口径 A/B）

DeepSeek-V4 284B（Q3_K），GPU 卸载 14 层专家，72 线程：

![DSV4 vs upstream](benchmarks/dsv4_vs_upstream.png)

行窗 EP 与 mirror 双双反超主线：**PP 2.1-2.2×、raw/no-DSpark TG +9%**；行窗 EP 额外省一半专家内存。PP 优先或单路场景 isolate 配置 pp512 可达 370 tok/s。

### vs 主线全档复测（2026-08-07，DSV4-Flash 284B mxfp4 生产形态）

![vs mainline 2026-08-07](benchmarks/vs_mainline_20260807.png)

主线 `e9fa0781f` vs 本分支（含 Q2_K/mxfp4 VNNI dpbusd 化），同机同口径 llama-bench A/B。**GPU 卸载（-ngl 99 -ncmoe 99 EP，生产形态）：PP 全档 +40~59%（pp2048 265.1 vs 166.5、pp8192 257.7 vs 164.5、pp16384 227.5 vs 163.1）、raw/no-DSpark TG +20%（23.3 vs 19.5）**；纯 CPU（-ngl 0）：PP +48%（135.1 vs 91.3），raw/no-DSpark TG 已知回归（3.7 vs 8.1，−54%，EP 无关系，隔离定位中——GPU 卸载为生产场景，不受影响）。Q2_K repack 经 dpbusd 化后 PP 164→218（+33% 端到端），仍略落后 no-repack（244），Q2_K 建议维持 `--no-repack`。

### GLM-5.2 745B：IQ traits + gemm 分流

![GLM traits](benchmarks/glm_traits.png)

### CPU 内核：8×8 repack vs 主线 legacy vec_dot

![CPU kernel speedup](benchmarks/cpu_kernel_speedup.png)

gemv（raw kernel decode，与 DSpark 无关）≈1× 是内存带宽物理上限；prefill gemm 提升来自 8×8 重排摊销权重读带宽。完整 300 格数据可用 `tests/test-repack-kernels --perf` 复现。

### NUMA 局部性：为什么 EP/mirror 必然赢过主线 distribute

![NUMA locality](benchmarks/numa_locality.png)

主线 `--numa distribute` 把权重页 interleave 摊到两路，每次权重读约 50% 跨 UPI；实测跨路读带宽只有本地的 ~38%（53.7-54.6 vs 136-143 GB/s，membw2 76 线程/路）。行窗 EP 与 mirror 结构性做到 ~100% 本地读，双路合计有效带宽 279.2 GB/s，比主线 interleave 模式（177.6 GB/s）高 **57%**。带宽矩阵全表见 `CHANGES.md`。

### 纯 CPU 推理（-ngl 0，同机 A/B）

![Pure CPU vs upstream](benchmarks/pure_cpu_vs_upstream.png)

DSV4 284B Q3_K 纯 CPU（72 线程 interleave）：**PP +38%、raw/no-DSpark TG +17-18%**。2026-08-05 负载环境复测，安静窗口定稿后更新。

### 多 slot 并发与混合配置复测（2026-08-05）

![Multi-slot concurrency](benchmarks/multislot_concurrency.png)

llama-server 8 槽 raw/no-DSpark 并发压测（每槽 512 tok）：**EP（行窗）配置全并发档领先主线——1 槽 +22%、8 槽 +18%（74.5 vs 63.2 tok/s）**；mirror 作为兼容性选项在 8 槽被主线反超 7%（高并发下其每令牌权重流量摊薄、结构性带宽优势贬值），生产多并发请用 `GGML_NUMA_EP=1` 配置。混合配置（14 层专家上双卡）同口径 llama-bench：EP pp512 **227.8 vs 主线 158.4（+44%）**、raw/no-DSpark tg512 31.3 vs 29.1（+7.5%）；mirror pp512 200.2（+26%）。

### 超长上下文：layer-major prefill（DSV4-Flash，16K）

![16K PP 演进](benchmarks/longctx_pp_progression.png)

![MXFP4 Hybrid CPU 审计与双卡 EP](benchmarks/mxfp4_hybrid_cpu_audit.png)

原生 MXFP4 版（155GB，137GiB 专家单份 CPU_REPACK + 双路 NUMA EP）经 CPU 审计三连修后 4K PP +144%。

### 长上下文 decode（16K，固定负载 A/B）

![16K TG 改进](benchmarks/longctx_tg_improvements.png)

长上下文 raw/no-DSpark TG 衰减根因已定位为 GPU attention/KV 的物理 dense 扫描，逐项修复（raw-SWA ring 已于 08-06 默认化，见下节）。16K GPU 侧时间分解（Nsight）：

![16K 热点分解](benchmarks/longctx_hotspots.png)

### FA decode 结构性修复：TG64 累积演进（2026-08-06/07）

![FA decode TG64 累积演进 / progression](benchmarks/fa_decode_tg64_progression.png)

本图全部为 raw/no-DSpark。左：f16 KV，raw-SWA decode ring 默认化把 fixed TG64 从 8.894 推到 12.085 tok/s（+36%），多 slot 自动回退。右：q8_0 KV，compact 稀疏 FA 只物化 top-k 选中行，9.75→10.69（+9.6%），超越 f16 dense（9.66）成为最快 q1 decode 路径。多流稀疏 FA（opt-in）≤16K 持平，收益主场在 256K+ 长上下文。

### 双机 expert-parallel（GLM-5.2，100G RoCEv2 直连）

![双机 EP](benchmarks/dual_machine_ep.png)

最新单 slot 真 EP 口径（2026-08-10，GLM-5.2 UD-Q2_K_MXFP4，MoE 层 3–77，dense/attention 在双 3090，CPU 只跑 routed experts）：

| 对比 | 单机 2 NUMA worker | 双机 4 NUMA worker | 4 / 2 |
|---|---:|---:|---:|
| MoE 阶段 raw/no-DSpark decode 时间（75 层合计，扣除 prompt） | 86.71 ms/token | 69.58 ms/token | **1.246×** |
| PP512（b512/ub256，同代码三轮均值） | 24.13 tok/s（21.22s） | **40.59 tok/s（12.62s）** | **1.682×** |

PP 映射来自该 workload 的 `layer,expert,count` 画像：四路各保留 64 个 primary + 16–23 个热点副本（80–87/256 专家），在线 dealer 在 holder 集合内逐 token 选最轻端点。逐层关键路径负载/理想平均从 1.374× 降到 1.091×；worker RSS 约 71–78GiB/NUMA，未使用 swap。MoE decode 与 PP 均完成 128-token 逐字节参考对拍。

### 生产 server：DSV4-Flash 每槽 1M 上下文

![Flash PP/TG 曲线](benchmarks/flash_pp_tg_curve.png)

当前后台是 **F16 KV、单槽完整 1M、UB256**。连续布局让 target 43 层与 target KV 留在
CUDA0，DSpark、draft KV 及共用 embedding/output 位于 CUDA1。16K PP（prefill，无 draft 接受收益）三轮均值
269.36 tok/s；本轮恢复生产 2–4 行派发前的配对三轮为 267.114/269.287/268.928，均值
268.443 tok/s。最终四 worker 回退后，固定 128-token DSpark speculative 请求三轮为
35.642/36.898/36.386，均值 36.309 tok/s，三轮 draft/accepted 均为 145/78 且输出一致。
历史最佳短上下文配置
37.921 tok/s 保留为 DSpark speculative 性能上限，不与 1M context 口径混算。

PP 极限档把 `-b/-ub` 提到 2048 时，热态约 318.08 tok/s，相对同组 UB256 约 +20.7%，
但 DSpark speculative TG 36.092 对 36.707 tok/s 下降 1.68%，所以没有替换当前 TG/质量优先的 UB256。
现阶段 `TG 40+ / PP 1000+` 目标尚未达到；README 只记录已复测结果，不把实验峰值或失败路径
写成生产性能。

容量档另行验证过 Q8 KV `1M×3`：GPU0/1 实占 20,456/12,304MiB，三个 slot 均由
`/slots` 确认为 `n_ctx=1048576`，单请求 DSpark speculative 热态 34.260 tok/s；`1M×4` 在 CUDA0 申请
2.73GiB PP compute buffer 时 OOM。因此三槽是双 3090 的 Q8 容量上限，不是当前质量优先
默认。历史 F16 `1M×2` 的 DSpark speculative 35.526 tok/s、并发总计 42.998 tok/s 继续作为对照。

> 所有数字均为实测，口径与复现方式见 `CHANGES.md` 与 `benchmarks/`（绘图脚本同目录）。已实测否决的路线（full tensor split -44%、meta-backend TP -20%、跨 tile 双 scheduler）也记录在案。

## 上游合并回归（2026-08-17，merge 4df29be4f / 96 提交）

同机同口径（DSV4-Flash mxfp4，-ngl 99 -ncmoe 99 -t 72 NUMA_EP+HIER_BARRIER+PREFETCH，--load-mode none，-b 4096 -ub 1024）：合并后 raw/no-DSpark tg512 **24.24±0.14**（合并前 24.4-25.0，持平）；pp2048 **282.99±4.39**（合并前 ≈299，**-5.4%**，差异来自上游图/调度改动，本分支路径未动）。AVX2-only 构建（build-avx2）纯 CPU 烟测通过：pp512 73.22 / raw/no-DSpark tg128 6.63。
