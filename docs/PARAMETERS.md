# x-llama.cpp 参数手册（Parameters Reference）

> 适用范围：本仓库 `local` 分支新增的环境变量（env vars）与命令行参数（CLI flags）。
> 「默认」指不设置该变量 / 不传该开关时的代码路径。除特别注明外，环境变量置 `1` 开启、置 `0` 或不设置关闭。
> 所有语义均对照源码读取点核实，不确定处直接标注「见源码 <file:line>」。
> 上手配置见 [QUICKSTART.md](QUICKSTART.md)。旧文档 PEOPLESLLM-PARAMS.md 已被本文档取代（仅保留重定向页）。

目录：

1. 单机 NUMA（CPU / CPU-MoE 推理）
2. 跨机 RPC Expert-Parallel（master 侧）
3. 跨机 EP worker（`llama-epd`）与 RDMA
4. GPU 混合推理与流式 prefill
5. 调度器（dspark / server 调度）
6. 调试与观测
7. CLI 参数（server / llama-bench / llama-epd）
8. 已废弃或建议避免
9. AMX 加速路径（Sapphire Rapids+）
10. 附录：L3 实验/否决留档

---

## 1. 单机 NUMA（CPU / CPU-MoE 推理）

### 1.1 NUMA expert-parallel（EP）核心

源码：`ggml/src/ggml-cpu/ggml-cpu.c`、`ggml/src/ggml-cpu/repack.cpp`、`src/llama-model.cpp`。

`GGML_NUMA_EP=1` 的语义：每个 routed-expert 张量（`*_exps`）保持**单份**，但每个专家平面内按 128 行对齐的行窗（row window）切分到各 NUMA 节点并 `mbind` 放置；计算侧每个节点先从本地窗口认领（claim）行，大批量时追加跨节点窃取（steal）相位。只在 NUMA 节点数 > 1 时生效（`ggml_cpu_numa_ep_active()`）。

| 变量 | 默认 | 作用 | 何时开 / 关 |
|---|---|---|---|
| `GGML_NUMA_EP` | 关 | 单机 NUMA EP 总开关（行窗放置 + claim 计算）。需要多 NUMA 节点；权重需可迁移（见 `_MMAP`）。`ggml/src/ggml-cpu/ggml-cpu.c:963` | 双路/多路 Xeon/EPYC 上跑 MoE（纯 CPU 或 `-ncmoe` CPU 专家）必开。单节点机器无效 |
| `GGML_NUMA_EP_CLAIM` | 开 | `=0` 时保留行窗页放置但旁路 EP 计算路径，落回专家优先路径。`ggml-cpu.c:2300`、`repack.cpp:6219` | **仅诊断**：用于归因「放置」与「claim 路径」各自的贡献。生产勿设 |
| `GGML_NUMA_EP_CHUNK` | 自动：token≤8 用 16，否则 64 | 行窗 claim 的块行数。显式值必须是内核列块（NB_COLS）的倍数且整除 128，否则回落 16。`repack.cpp:6246`、`ggml-cpu.c:2320` | 一般不设；模型相关扫参时才显式给 16/32/64/128 |
| `GGML_NUMA_EP_STATIC` | 关 | 单相位（小批）时用静态连续行窗划分替代动态 claim：零原子、每线程长顺序流，数值不变。`repack.cpp:6228` | 实验开关；实测端到端不优于 claim，保持关 |
| `GGML_NUMA_EP_STEAL_MIN_TOKENS` | `32` | 批内 token 数 > 此值才跑「本地 + 窃取」两相位；≤ 阈值只做本地相位。`ggml-cpu.c:976` | 一般不调；与分级 barrier 的 TG 判据一致 |
| `GGML_NUMA_EP_GATE_UP_PARALLEL` | 关 | 小批 decode 时把每个节点内线程对半分给独立的 gate/up 专家投影并融合 clamp/GLU；要求 2 节点、线程数可被 4 整除、repack 权重、token 维 ≤8，不满足自动回退。`ggml-cpu.c:5452` | 配合 `GGML_NUMA_EP=1` + `GGML_NUMA_HIER_BARRIER=1` 的双路纯 CPU decode；混合模式 PP 自动走原路径 |
| `GGML_NUMA_EP_DEBUG` | 关 | 行窗 EP claim 诊断：每 256 次调用打印 wall/spread/busy/两节点认领行数/窃取统计。`repack.cpp:6021` | 调试专用；共享计数器热行有明显海森堡效应（TG 约 -18%），只看相对结构 |
| `GGML_NUMA_EP_MMAP` | 关 | 允许对 mmap 加载的模型做 **policy-only** 专家放置（`mbind(MPOL_BIND, flags=0)`，只设 VMA 策略、不迁移已缓存页）。不设时 mmap + EP 直接跳过放置并打日志。`src/llama-model.cpp:1501` | **避免**：冷 page cache 下首触页按 interleave 落两节点后被钉死在错位节点，PP 减半。EP 生产配置请用 `--no-mmap` |
| `GGML_NUMA_EP_PLACE` | 行窗（默认） | `=block` 时专家权重改按 2 MiB 块交替 mbind 到各节点，计算侧按同一字节网格推导节点本地行区间。`src/llama-model.cpp`、`repack.cpp` | **否决留档**：端到端零收益（HANDOVER:1377-1383），保持默认 |

### 1.2 NUMA barrier 与 mirror

| 变量 | 默认 | 作用 | 何时开 / 关 |
|---|---|---|---|
| `GGML_NUMA_HIER_BARRIER` | **开**（2026-08 起默认开，设 `0` 回退 flat barrier） | 纯自旋两级 NUMA 分级 barrier（先节点内、再跨节点），替代全局 flat barrier。仅在 `--numa mirror`，或 `--numa distribute` + `GGML_NUMA_EP=1` 的块划分下生效。`=2` 追加节点内小组级（三级 barrier 脚手架，配合 `GGML_NUMA_BARRIER_GROUP`）。`ggml-cpu.c:1117,1206` | 双路机上配合 EP/mirror 开启（实测浅层 TG +8.6%、深提示 +6.2%）；其他拓扑自行 A/B；`=2` 见下行，已否决 |
| `GGML_NUMA_BARRIER_GROUP` | `8` | 三级 barrier（`GGML_NUMA_HIER_BARRIER=2`）下节点内每小组的线程数；<2 忽略。`ggml-cpu.c:1222` | **否决留档**（Slice 10，commit 35aa515b9）：无净收益，保持两级 |
| `GGML_NUMA_PIN_CORE` | 关 | `=1` 把每个线程钉到按 mesh 中心性排序的单核（置换表来自 Ice Lake SP 实测 c2c 矩阵，仅 38 物理核/节点的布局生效，否则回退节点 cpuset）。`ggml-cpu.c:3135` | **否决留档**（Slice 10）：实测无净收益，仅实验复现用 |
| `GGML_NUMA_MIRROR_THREADS` | `hardware_concurrency()`（上限 32） | `--numa mirror` 加载期节点间权重复制的线程数。`src/llama-model.cpp:1264` | 加载太慢时调大；一般默认即可 |
| `GGML_NUMA_MIRROR_BUDGET_GB` | `12` | PARTIAL 镜像每节点预算硬上限（GiB）；实际预算还受「最紧节点空闲内存 − 6 GiB 保留」约束，防 OOM-kill。`src/llama-model.cpp:1311` | 仅实验调参 |
| `GGML_NUMA_MIRROR_PARTIAL` | 关 | 只镜像「热张量」（排除 `_exps` 与 `token_embd`，按大小优先、预算内尽量多）。`src/llama-model.cpp:1727` | 内存不够全量 mirror 时的折中；全量镜像内存不足会自动回落此模式 |
| `GGML_NUMA_MIRROR_MOE` | 关 | 只镜像含 routed 专家权重（`_exps`）的 buffer。`src/llama-model.cpp:1735` | 混合 CPU+GPU 模式：注意力在 GPU、只有 MoE 专家在 CPU 时避免为 GPU 侧张量浪费一倍内存 |
| `GGML_NUMA_FAKE_NODES` | 关 | 单 NUMA 节点机器伪造 N 个节点（1 < N ≤ 8）。`ggml-cpu.c:887` | **仅开发/测试** mirror/EP 代码路径，生产勿设 |

### 1.3 repack / CPU 内核

| 变量 | 默认 | 作用 | 何时开 / 关 |
|---|---|---|---|
| `GGML_REPACK_GEMV_PREFETCH` | **开**（2026-08 起默认开，设 `0` 关闭） | repack MXFP4 GEMV 内核的权重流软件预取（约 2 KB 提前量，8 个 block 迭代）。非空且非 `0` 即开。`ggml/src/ggml-cpu/arch/x86/repack.cpp:1915` | AVX-512 机器上 decode 提速（生产配置开启）；AVX2 机器无此内核、无效 |
| `GGML_REPACK_MXFP4_AVX512_GEMV` | 开 | MXFP4 GEMV 使用 AVX-512 内核；`=0` 回退 AVX2 实现。`arch/x86/repack.cpp:2018` | 仅 A/B 诊断用，生产保持默认 |
| `GGML_CPU_DISABLE_FUSION` | 关（上游已有） | `=1` 禁用 CPU op fusion 框架。`ggml-cpu.c:5448` | 调试用 |
| `GGML_REPACK_MMID_GEMM_TILE` | `32` | MoE `mul_mat_id` 每次 GEMM 调用 staging 的 src1 行数上限；仅接受 4/8/16/32，其他值回落 32。`repack.cpp:6254` | 扫参/A/B 专用 |
| `GGML_CPU_FP16_INTERMEDIATE` | 关 | `=1` MoE 块内激活走 f16 中间态（优先于 q8 边界）。`src/llama-graph.cpp:2501`、`repack.cpp:41` | 实验岛，默认关 |
| `GGML_CPU_INT8_INTERMEDIATE` | 关 | `=1` MoE 块内激活走 q8_0 中间态；与 FP16 同设时 INT8 赢。`src/llama-graph.cpp:2508`、`ops.cpp:664` | **否决留档**：比 FP16 差一个量级（HANDOVER:1227 ②），别用 |
| `GGML_MOE_HOT_STATS` | 关 | `=1` 经 gate 投影的 mmid ids 统计每层每专家命中数（每个 (token, expert) 选中计一次），退出时 atexit 落盘 TSV；上限 128 层 × 1024 专家。`repack.cpp:81` | 采集热专家画像；产出即 Slice 12 `GGML_HOT_EXPERT_TABLE` 的输入格式（§4.4） |
| `GGML_MOE_HOT_STATS_PATH` | `/tmp/expert-hot.tsv` | 上者 TSV 的落盘路径。`repack.cpp:60` | 配合上者 |
| `GGML_MOE_HOT_TRACE` | 关 | `=1` 记录 gate router 的时序选择，每个 token row 输出 `step layer expert...`；`step` 是捕获序号。事件先写入有界内存，退出时批量落盘。`xllama-hot-trace.cpp` | 仅用于单槽、hot-expert-off 的 temporal LRU 回放；与 `GGML_HOT_EXPERT=1` 同开时会拒绝启用，避免把已掩码 sentinel 写成伪缺失 trace |
| `GGML_MOE_HOT_TRACE_PATH` | `/tmp/expert-hot-trace.tsv` | temporal trace 输出路径 | 配合上者；`replay-hot-expert-cache.py --format trace` 可直接读取 |
| `GGML_MOE_HOT_TRACE_MAX_MB` | `64` | trace 内存上限，接受 1–4096 MiB；满后停止追加并在 footer 记录 `dropped_capacity` | 长时间采集按预计 token 数调整；出现 drop 的回放不标记为 exact |

#### 1.3.1 repack 格式开关（读取点集中在 `repack.cpp:7646-7846`，均为 A/B 旋钮，生产保持默认）

| 变量 | 默认 | 作用 |
|---|---|---|
| `GGML_REPACK_Q2_K` | AVX512-VNNI 机器**关**（该平台上成熟 row-major vec-dot 更快），其他架构开 | Q2_K 8x8 repack 内核；`=1` 强制开、`=0` 强制关 |
| `GGML_REPACK_IQ2_XXS` | 开 | IQ2_XXS 8x8 内核（需 AVX512+VNNI+VBMI）；`=0` 回退 |
| `GGML_REPACK_IQ4_XS` | **关** | IQ4_XS x8 LUT 布局（无原生内核，x86 上比 vec_dot 慢）；`=1` 仅为原生内核开发留门 |
| `GGML_REPACK_Q3_R` | 开 | Q3_R repack（恒等 memcpy，热内核需 VNNI+VBMI+BW）；`=0` 强制 row-major vec_dot 回退 |
| `GGML_REPACK_UDNL_W4` | 开 | UDNL_W4 NR16xK4 panel 重排（需 AVX512F+BW+VNNI）；`=0` 回退标量 vec_dot |
| `GGML_REPACK_UDNL_MX` | 开 | UDNL_MX panel 重排 + 16 行共享 mode word 折叠（需 F+BW+VNNI+VBMI）；`=0` 回退 |
| `GGML_REPACK_E4A` | 开 | E4A NR16xK4 panel 重排（需 AVX512F+BW+VNNI）；`=0` 回退 |
| `GGML_REPACK_Q3_R_GEMV_INT` | 开 | Q3_R GEMV 整数 scale 变体；`=0`（或空串）回落 bit-exact 变体。**每次调用读取**（非 static），测试可动态切换。`arch/x86/repack.cpp:2482` |
| `GGML_REPACK_Q3_R_GEMM_MR` | `8` | Q3_R GEMM 行分块；`=4` 恢复旧分块。`arch/x86/repack.cpp:5994` |
| `GGML_REPACK_Q3_R_GEMM_MADD` | 开 | Q3_R GEMM maddubs+madd 整数 tile（仅 nc≥8 启用）；`=0` 强制 fp32-finalize tile。`arch/x86/repack.cpp:6002` |

### 1.4 E4A 有界流式加载

源码：`src/llama-e4a-stream-bake.cpp`、`src/llama-model-loader.cpp`、`ggml/src/ggml-cpu/repack.cpp`。

| 变量 | 默认 | 作用 | 何时开 / 关 |
|---|---|---|---|
| `GGML_E4A_STREAM_BAKE` | 关 | `=1` 时，非 mmap、非 direct-I/O 且目标为 CPU_REPACK 的 E4A tensor 使用双 staging `pread` + NR16 panel byte-bake；不分配完整 raw tensor。非 E4A、非法 shape、非 CPU_REPACK 或能力不匹配在写入前自动走原 loader。读取/验证错误终止加载，进度取消返回失败，不执行半成品 tensor | E4A + AVX512/VNNI 的 `--load-mode none` 实验；当前保持 opt-in，生产启用前先跑 `test-e4a-stream-bake` 和真实模型 smoke |
| `GGML_E4A_STREAM_BAKE_CHUNK_MIB` | `1` | requested staging chunk MiB，合法范围 1–1024；运行时向完整 `(16 rows, all K blocks)` panel 边界取整，双 staging 峰值约为 2x effective chunk。非法值会 warning 并关闭 stream bake | 当前 NVMe/E4A pure-byte 实测 1 MiB 最优；不同磁盘、direct I/O 或未来 requantization baker 需重新扫描 |

---

## 2. 跨机 RPC Expert-Parallel（master 侧）

源码：`src/llama-remote-ep.cpp`（`parse_env()` 起始于 `:406`）。所有变量只在 `GGML_REMOTE_EP` 置非 0/非空后读取。

### 2.1 连接与分层

| 变量 | 默认 | 作用 | 何时开 / 关 |
|---|---|---|---|
| `GGML_REMOTE_EP` | 关 | 双机 EP 总开关：认领层的 MoE FFN 经 RPC 发给 worker；不满足条件的图自动回退本地。`src/llama-remote-ep.cpp:407` | 跨机部署必开 |
| `GGML_REMOTE_EP_HOST` | `127.0.0.1` | worker 地址。`:413` | 指向 slave 直连 IP；**slave 关机降级到本机 worker 时用 `127.0.0.1`**（见 QUICKSTART 坑 b） |
| `GGML_REMOTE_EP_PORT` | `29200` | worker 端口。`:416` | 与 `llama-epd --port` 一致 |
| `GGML_REMOTE_EP_LAYERS` | 全部层 | 远端认领的层区间，格式 `A-B`、`A` 或逗号列表 `3-7,22-42`。区间内 `ffn_*_exps.weight` 在 master **不分配不读入**（真分片，TENSOR_SKIP）。`:419` | 分层点由两端带宽/算力比决定（DSV4 实测 7–8 层最优）；区间必须落在 master 已 offload 给 CPU 的层内 |

### 2.2 层镜像（MIRROR）与动态调度（SCHED），二者互斥

`GGML_REMOTE_EP_SCHED=1` 与 `GGML_REMOTE_EP_MIRROR=1` 同设时 SCHED 赢、MIRROR 自动禁用并打 warning（`:598`）。

| 变量 | 默认 | 作用 | 何时开 / 关 |
|---|---|---|---|
| `GGML_REMOTE_EP_MIRROR` | 关 | 层镜像 + 专家 slot 维拆分：master 镜像远程层专家权重，MoE 沿 slot 维切 `[0,k_r)` 发 worker、`[k_r,k)` 本地算。`:432` | master 内存有空闲且 decode/长 PP 为主（TG +9~11%、PP1020 +11~27%）；**小档 PP（≤256 tok）回归，关掉** |
| `GGML_REMOTE_EP_MIRROR_LAYERS` | = `_LAYERS` | 镜像层子区间。`:437` | 内存不够镜像全部远程层时缩小范围 |
| `GGML_REMOTE_EP_MIRROR_KREMOTE` | `n_expert_used/2` | 发远端的 slot 数 k_r（clamp 到 [1, n_expert_used-1]）。`:447` | 实测 2/3/4 差异在噪声内，默认即可 |
| `GGML_REMOTE_EP_SCHED` | 关 | 专家级动态调度：多端点 CAP 协商（协议 v2）+ dealer 纯函数派单，逐 slot 不求和。`:464` | 多 worker（含同机多 NUMA worker）时开；单 worker 经典模式不需要 |
| `GGML_REMOTE_EP_SCHED_ENDPOINTS` | = `HOST:PORT` | 多端点列表，逗号分隔 `host:port,...`。`:506` | 多于一个 worker 时显式列出 |
| `GGML_REMOTE_EP_SCHED_KLOCAL` | `2` | m*：每 token 本地保留的 slot 数。`0` = 严格 pure EP：master 不加载目标层 routed-expert 权重，worker 所有权必须完整覆盖（有缺口直接中止，无静默回退）。`:533` | 真 EP（权重全在 worker 上）设 0 |
| `GGML_REMOTE_EP_SCHED_MAX_EFFORT` | 关 | 配合 `KLOCAL=0` 允许 worker 所有权位图重叠（热点专家副本）；仍要求每个专家至少一个持有者。KLOCAL≠0 时自动禁用。`:541` | 用热点副本的画像派单时开 |
| `GGML_REMOTE_EP_SCHED_PP` | 关 | 允许 n_tokens > 1 的多 token 派单（PP）；默认仅 decode。`:549` | 需要 PP 也走远端时开（生产开启） |
| `GGML_REMOTE_EP_SCHED_TG_ACTIVATION_COST` | `0` | TG 首次启用某 endpoint 的 fanout 虚拟成本（1000 ≈ 一个中位专家 assignment）。`:559` | 仅影响多 holder 专家的取舍；调优用 |
| `GGML_REMOTE_EP_SCHED_TG_REPEAT_COST` | `250` | 同一次 TG 请求再次命中同一专家时的虚拟成本（0–1000；250 = 按新 stream 的 1/4 计）。`:569` | 对应 worker 的 shared-weight 内核路径；DSV4 生产值 250 |
| `GGML_REMOTE_EP_SCHED_PP_REPEAT_COST` | `1000` | 同一 PP batch 再次命中同一专家时的边际虚拟成本（0–1000）。`:579` | 生产显式设 250（同组 A/B +0.94%） |
| `GGML_REMOTE_EP_SCHED_REPEAT_ACCOUNTING` | 关 | 把 endpoint 在飞队列与服务率样本统一成「新专家 + repeat 边际成本」单位。`:589` | 多 slot 调度实验；单 slot 实测 -1.45%，生产保持关 |
| `GGML_REMOTE_EP_WEIGHT_ON_MASTER` | 关 | 同步 SCHED 用 REQ4/RESP4：worker 返回未乘 router weight 的逐 slot 向量，master 加权合并。与异步 `_PIPE=1` 不兼容（自动关闭）。`:552` | 减少 worker 端操作，生产开启 |
| `GGML_REMOTE_EP_RECONNECT_TIMEOUT_MS` | `0`（一次立即重连） | SCHED endpoint 断连后等待 worker 重启并重发暂存请求的最长毫秒（0–300000）。重连后强制核对 expert map/kernel ID/CAP，不一致即拒绝。`:75` | 常驻服务建议 `90000`（覆盖 worker ~70 s 的权重加载/repack） |

### 2.3 传输与合并

| 变量 | 默认 | 作用 | 何时开 / 关 |
|---|---|---|---|
| `GGML_REMOTE_EP_PIPE` | 关 | REQ3/RESP3 异步请求号协议，每 endpoint 后台接收线程；隐式启用 SCHED。`:477` | 跨 slot 流水实验；当前生产 decode 用同步 REQ4（`_PIPE=0`） |
| `GGML_REMOTE_EP_PIPE_MAX_MIB` | `512` | 异步 PIPE 所有在飞请求的总响应 staging 字节上限。`:483` | 只配合 `_PIPE=1`；过小回压，过大升峰值内存 |
| `GGML_REMOTE_EP_PIPE_MAX_REQUESTS` | `256` | 异步 PIPE 在飞请求数上限。`:494` | 同上 |
| `GGML_REMOTE_EP_PARALLEL_IO` | 关 | 对独立 endpoint 并行 send/recv；响应仍按全局 slot 次序合并，浮点结合顺序不变。`:61` | 多 worker 同步 REQ4 生产开启 |
| `GGML_REMOTE_EP_MERGE_THREADS` | `8` | SCHED 响应按 token 并行、按 slot 原序合并；token <64 自动单线程。`:47` | 只影响 PP merge；本机扫频 8 最优 |
| `GGML_REMOTE_EP_PIPELINE` | 关 | 单层 token 维切块 + W=1 滑动窗口流水线投递。K<2（decode）自动走原路径。`:106` | 实测净收益仅 +0.7~1.4%，保持关 |
| `GGML_REMOTE_EP_PIPELINE_CHUNK` | `256` | 流水线切块 token 数；再按帧字节封顶（TCP ≤3 MiB / RDMA ≤12 MiB 防死锁）。`:114` | 仅配合 PIPELINE |
| `GGML_REMOTE_EP_FREQ` | 关 | 统计每个图中 router 对 `(layer, expert)` 的选择次数（只计数，不改派单）。`:469` | 采集真实 workload 画像，配合 `tools/epd/ep-map-from-freq.py` |
| `GGML_REMOTE_EP_FREQ_FILE` | 空（退出时打印逐层摘要） | 设置 CSV 路径则落盘完整 `layer,expert,count` 表。`:472` | 画像采集时设 |

### 2.4 master 侧诊断

| 变量 | 默认 | 作用 |
|---|---|---|
| `GGML_REMOTE_EP_DEBUG` | 关 | 每次 RPC 打 send/wait/compute 分段计时（master 与 worker 两侧均读此变量）。`src/llama-remote-ep.cpp:35`、`tools/epd/llama-epd.cpp:82` |
| `GGML_REMOTE_EP_TRACE_ROUTER` | 关 | 在 `_DEBUG=1` 时输出小批次 router 专家共激活 trace，供 `ep-map-from-trace.py`。`:87`。日志量大，生产关 |

---

## 3. 跨机 EP worker（`llama-epd`）与 RDMA

### 3.1 RDMA（master 与 worker 两侧同名变量）

源码：`tools/epd/llama-ep-transport.cpp`、`tools/epd/llama-ep-rdma.cpp`。编译需检测到 libibverbs + librdmacm（CMake 打印 `EP RDMA transport (RoCEv2): ON`），否则置 1 也回退 TCP。

| 变量 | 默认 | 作用 | 何时开 / 关 |
|---|---|---|---|
| `GGML_REMOTE_EP_RDMA` | 关（TCP） | RoCEv2 transport（rdma_cm 自动 GID，RC QP）。建连失败自动回退 TCP + warning。**master 与所有 worker 必须同设**（协议不同，不互通）。不存在 `GGML_EPD_RDMA`。`tools/epd/llama-ep-transport.cpp:245` | 有 RoCE 网卡必开：跨机 64 B RTT 42–74 µs → 10–13 µs |
| `GGML_EP_RDMA_SPIN` | 关 | CQ busy-poll 代替 completion channel，持续占核。`tools/epd/llama-ep-rdma.cpp:124` | Ice Lake + ConnectX-5 实测仅 master 开有净收益（+2.7%），worker 同时开反而降；逐机 A/B |
| `GGML_EP_RDMA_COALESCE` | 开 | 发帧时把 header 与分散 payload 合并到最少的已注册 SEND slot；`=0` 逐段发送。`tools/epd/llama-ep-transport.cpp:22` | 单 slot ABBA 无可测增益，默认开是为多 slot 的 WR 容量与稳定性；保持默认 |

### 3.2 EPD worker（`tools/epd/llama-epd.cpp`）

| 变量 | 默认 | 作用 | 何时开 / 关 |
|---|---|---|---|
| `GGML_EPD_AUTOTUNE` | 开 | 未显式 `-t` 时对 {16,24,32,36,物理核} 阶梯实测专家 FFN，取全局最优 3% 内最小线程数（<1 s）。`llama-epd.cpp:1518` | 保持默认；`--no-autotune` 或 `=0` 关闭（关闭后默认 8 线程）。显式 `-t` 时不运行 |
| `GGML_EPD_AUTOTUNE_ROWS` | 自动（按 `ceil(top_k × owned/full)` 估算） | 覆盖标定用的每端点 decode assignment 行数。`:1582` | 一般不设 |
| `GGML_EPD_NUMA` | `off` | worker 权重页 NUMA 策略：`interleave`（MPOL_INTERLEAVE）/ `weighted`（MPOL_WEIGHTED_INTERLEAVE，内核 ≥6.9），在任何权重分配前对进程生效。`:1960` | **单个 worker 跨两路 CPU + `--no-mmap` 时设 `weighted`**；每 NUMA 一个 worker 时用外部 `numactl --membind=N`，此变量保持 off |
| `GGML_EPD_NUMA_WEIGHT` | 启动实测（带宽探针，~150 ms/节点） | weighted 模式的节点权重比，`a:b` 或 `a,b`。`:2005` | 节点带宽不对称且自动标定不准时手动指定 |
| `GGML_EPD_REPACK` | 开 | worker 侧专家权重启动转 CPU_REPACK 交错布局，启用 repack GEMV/GEMM 内核。`:127` | 保持默认开（修复前 PP 慢 ~4.5×）；`=0` 仅诊断 |
| `GGML_EPD_CPP_GATHER` | 开 | 小 TG 请求用串行紧凑 memcpy 收集 ragged hidden，避免一次 GET_ROWS 图调度。`:94` | `=0` 仅 A/B |
| `GGML_EPD_FUSE_GATE_UP` | 开 | 兼容的 separate gate/up 加载期融合为同一 repacked 张量（总字节不变，少一次 MMID 调度与一次激活量化）。`:138` | `=0` 仅诊断 |
| `GGML_EPD_FUSE_CLAMP_SWIGLU` | 开 | DSV4 clamped SWIGLU 在 CPU 单 op 内完成 clamp + 激活。`:146` | `=0` 回三节点参考图，正确性对拍用 |
| `GGML_EPD_SHARED_Q8_MIN_TOKENS` | `2` | gate/up 不能融合时，从 N token 起共享一次 Q8 激活量化；`0` 关、`1` 强制所有 batch。`:1264` | 保持默认 |
| `GGML_EPD_POLL` | `50` | persistent CPU threadpool 混合轮询等级（0–100）。`:210` | worker 独占 NUMA 节点保持默认；与 master 同机共享 CPU 时从 `0` 起测，避免空闲忙等争核 |
| `GGML_EPD_GRAPH_CACHE_MAX_ROWS` | `64` | 只缓存 ≤ 该行数的 MoE 计算图；大 PP ragged 形状重建图但复用 grow-only allocator。`:1322` | 保持默认 |
| `GGML_EPD_GRAPH_CACHE_MIB` | `512` | worker 计算图缓存总预算（LRU）。`:1214` | 保持默认 |
| `GGML_EPD_NO_GRAPH_CACHE` | 关 | 置 1 完全禁用 worker 图缓存。`:1318` | 仅诊断 |
| `GGML_EPD_HUGEPAGES` | 关 | repack 完成后对大匿名专家 buffer 做 `MADV_HUGEPAGE`。`:159` | 实测无净收益且曾触发 systemd-oomd，保持关 |
| `GGML_EPD_MAX_SESSIONS` | `64`（1–4096） | worker 同时持有的 client session 上限（非 llama-server slot 数）。`:2904` | 多 server 共享 worker 时调 |
| `GGML_EPD_OP_TIMING_EVERY` | `0` | 每 N 个请求输出一次 worker op timing。`:107` | 诊断专用，输出与计算互斥 |
| `GGML_EP_PREFAULT` | 关 | 启动时多线程预触认领层专家权重的全部 mmap 页，消除冷专家首次命中的多 ms 页入尖峰（30%→~1%，启动 +4.5 s）。`--no-mmap` 时自动跳过。`:200` | mmap 模式的生产 worker 建议开；用 `--no-mmap` 就不需要它 |
| `GGML_EP_PREFAULT_THREADS` | `16` | prefault 线程数。`:1160` | 配合上者 |

worker CLI 参数见 §7.3。

---

## 4. GPU 混合推理与流式 prefill

### 4.1 GPU 流式 MoE prefill（`GGML_CUDA_MOE_PP_*`）

源码：`ggml/src/ggml-backend.cpp:1805-1822`（scheduler 侧）、`src/llama-graph.cpp:2397-2414`、`src/llama-layer-major.cpp`。核心思路：开启后 `*_exps` 专家权重保持原始布局的 CUDA pinned host buffer，达到 token 阈值的 `mul_mat_id` 用异步 H2D 把权重流到 GPU 上算，低于阈值留在 CPU（decode 不受影响）。

| 变量 | 默认 | 作用 | 何时开 / 关 |
|---|---|---|---|
| `GGML_CUDA_MOE_PP_MIN_TOKENS` | `0`（关） | GPU 流式 MoE prefill 的 token 阈值与总开关；同时解除「CPU_REPACK 图钉在 CPU」的限制。`ggml/src/ggml-backend.cpp:1805`、`src/llama-context.cpp:1494` | 长 prompt、大 ubatch 的 PP 专项，建议从 `2048` 起测。开启后专家权重不做 CPU_REPACK，不适合作为 decode 默认 |
| `GGML_CUDA_MOE_PP_PREFETCH` | `0`（关） | 0–4 个私有设备槽跨 split 预取专家权重（独立 H2D/commit stream + event 衔接）。`ggml-backend.cpp:1809` | 双卡 ub4096 实测深度 3 约 +20%；显存不足自动缩为 2/1/0 |
| `GGML_CUDA_MOE_PP_DEFER_PREFETCH` | `0`（关） | 预取延后到 scheduler 解析真实 view 权重后启动。`ggml-backend.cpp:1813` | 当前 true-EP 基准使用 |
| `GGML_CUDA_MOE_PP_PIPE` | `0`（关） | 有界 copy-stream MoE 流水线：content-addressed staging slot + 设备侧 commit event 代替 host drain。`ggml-backend.cpp:1819` | 配合 PREFETCH 的实验开关 |
| `GGML_CUDA_MOE_PP_UNREPACK_THREADS` | `16`（clamp 1–256） | CPU_REPACK → 原始布局 unrepack 线程池大小。`ggml/src/ggml-cuda/ggml-cuda.cu:2850` | 一般默认 |
| `GGML_CUDA_MOE_PP_EP` | `0`（关） | 真双卡同层 expert-axis EP：同层 routed experts 沿 expert 轴拆到两张 GPU 分别计算，融合 `MOE_WREDUCE` 按升序 slot 归并，只跨卡传 `[n_embd, n_tokens]` partial。要求 `n_expert` 偶数、MXFP4 等条件。`src/llama-graph.cpp:2401` | 双卡/NVLink 长 prefill（2K PP +63%）；必须配 `MOE_PP_MIN_TOKENS`；单卡勿设 |
| `GGML_CUDA_MOE_PP_EP_MIN_TOKENS` | `0` | true-EP 的最小 query batch；生效门限为 `max(ep_min, pp_min)`。`src/llama-graph.cpp:2397`、`src/llama-layer-major.cpp:241` | q1 decode 不进入 GPU EP；建议显式设 2048 |
| `GGML_CUDA_MOE_PP_EP_OWNER_EXPERTS` | `n_expert/2` | rank0 持有的专家数（覆盖 50/50 拆分）。`src/llama-layer-major.cpp:32` | 双卡算力不均时调 |
| `GGML_CUDA_MMQ_MOE_J` | `0`（自动） | 强制 MoE MMQ tile 宽度 J（8..128 步进 8）；自动模式按每专家典型行数选择。`ggml/src/ggml-cuda/mmq.cuh:1486` | 仅 A/B 测量 |
| `GGML_CUDA_P2P` | 关 | 允许双卡 backend 用 peer/NVLink copy。`ggml-cuda.cu:400` | 有 NVLink/P2P 时开；无能力勿设 |
| `GGML_CUDA_MOE_PP_RESERVE_MB` | `2048` | MoE 预取 staging 槽扩容时必须保留的显存余量（MB），不足则槽不增长。`ggml-cuda.cu:2527` | 显存紧张时调 |
| `GGML_CUDA_ALLREDUCE` | 平台默认（Linux 尝试 NCCL，未编译 NCCL 时回退） | 多卡 TP AllReduce 实现选择：`nccl` / `internal`（自研 P2P 流水线，`allreduce.cu`）/ `none`；未知值 warning + none。`ggml-cuda.cu:1277` | **实验级**：Slice 11 二轮 TP tg +49%（13.6→20.3）仍低于 layer 模式 25.9；生产保持 layer split |
| `GGML_CUDA_AR_P2P` | 开（有 P2P 授权时） | internal AllReduce 的 P2P direct-read 路径；`=0` 退出。前置：`GGML_CUDA_P2P` 已设（或编译 NCCL）且双卡双向 peer 可达。`allreduce.cu:714` | 仅 A/B 诊断 |
| `GGML_SCHED_ASYNC_READBACK` | **开**（设 `0` 关闭） | scheduler 对跨 backend split 的结果用异步 D2H readback（Slice 4a）。`ggml-backend.cpp:2559` | 生产默认；`=0` 仅 A/B 回归 |

### 4.2 attention / top-k / KV（CUDA 内核）

| 变量 | 默认 | 作用 | 何时开 / 关 |
|---|---|---|---|
| `GGML_CUDA_BATCHED_TOPK` | **开**（2026-08 起默认开，设 `0` 关闭；`k==512` 门限仍在） | `k=512, nrows>=32` 时用每行一个 block 的 stable batched radix top-k（边界 ties 按较低索引稳定选择）。`ggml/src/ggml-cuda/top-k.cu:190` | DSV4 Lightning Indexer launch storm 场景：16K PP +8%；生产配置开启 |
| `GGML_CUDA_DSV4_KV_REUSE` | **开**（2026-08 起默认开，设 `0` 关闭；仅命中 DSV4 形状门） | 对 DSV4 `K=V` alias、512 维、64 列 FA specialization 保留完整 K shared tile 供 V 相位复用（约 100 KB dynamic shared memory）。`ggml/src/ggml-cuda/fattn-mma-f16.cuh:2186` | 16K prefill +3.9%，logits 逐位一致；生产配置开启 |
| `GGML_CUDA_DSV4_SPARSE_RAW_COMPACT` | `0`（关） | sparse FA 内仅保留每组有效 raw span（≤512 行）再追加 compressed union；要求同时开 sparse FA 且 query batch ≥256。`ggml/src/ggml-cuda/fattn-common.cuh:1492` | 16K sparse 596→754 tok/s；`=2` 仅强制小形状单测，生产勿用 |
| `GGML_CUDA_DSV4_Q1_HEADS` | `0`（自动：Ampere MMA 且 K 行数 ≥1024 用 32，否则 8） | q1 FA 的 head 分组覆盖（>0 生效，要求 Q/K head 比整除分组）。`ggml/src/ggml-cuda/fattn.cu:201` | 仅实验 |
| `LLAMA_DSV4_SPARSE_FA` | `0`（关） | 8-query union 组装 sparse physical rows + 逐 query mask；只在 query batch ≥256 建图，q1 自动 dense。**改变浮点归约分组，非 bit-exact**。`src/models/deepseek4.cpp:1050` | 长上下文 PP 实验 |
| `LLAMA_DSV4_FUSED_INDEXED_FA` | `0`（关） | raw/compressed 两段 KV 不经全宽 concat 直接进 sparse kernel。仅 F16 KV。`=1` decode+prefill，`=2` 仅 q1 decode，`=3/4` 追加多流 decode。`deepseek4.cpp:314` | 16K 实测 q1 sparse 输 dense MMA（TG64 -12%），暂不默认开；长上下文再评估 |
| `LLAMA_DSV4_Q8_SPARSE_FA` | `1`（开） | 量化 KV（如 q8_0）q1 decode 走 compact gather：只物化选中 compressed 行。`=0` 回落全扫，`=2` 追加多流 decode。`deepseek4.cpp:330` | Q8 KV 的默认最快 q1 路径，保持开 |
| `LLAMA_DSV4_COMPACT_KV` | `0`（关） | 对任意 KV 类型（含 F16）强制 compact gather（`=1` 单流 q1，`=2` 追加多流）。`deepseek4.cpp:1055` | Q8 KV 已由上者覆盖，一般不设 |
| `LLAMA_DSV4_COMPACT_DECODE_SWA` | `1`（开） | layer-major 大 prefill 完成后把 raw SWA 最后 128 行搬到 256-cell 物理环，q1 decode 图宽与 prompt 长度解耦；decode 有微小数值差（非 bit-exact）。`src/llama-layer-major.cpp:1122` | 默认开（TG64 +22%）；多 slot/cache 被占用时自动回退 |
| `LLAMA_DSV4_HCA_INDEXED_FA` | `0`（关） | HCA 层对完整 compressed state + raw SWA 窗口的 indexed FA。`deepseek4.cpp:341` | 实验 |

### 4.3 GPU 流式 prefill（layer-major 执行器，server 集成）

server 侧资格门：全新单序列 prompt、token 数 ≥ 阈值、无 cache 复用/LoRA/mtmd/prob 才走 layer-major；其余零行为变化回落 chunked 路径（`tools/server/server-context.cpp:3394-3430`）。

| 变量 | 默认 | 作用 | 何时开 / 关 |
|---|---|---|---|
| `LLAMA_GPU_PREFILL_MIN_TOKENS` | `4096` | server 侧 GPU 流式 prefill 的 prompt token 阈值；`0` = 关闭该 fast path。`tools/server/server-context.cpp:3401` | 调阈值用；想禁用整条路径设 0 |
| `LLAMA_LAYER_MAJOR_DEVICE_HC` | `0`（host HC） | `=1` 把 DSV4 layer-major 的 F32 HC layer-boundary state 放首个 GPU，层间 D2D/P2P 传递（分配前保留 ≥4 GiB 或 20% 显存）；`=2` 追加跨 backend 迁移。`src/llama-layer-major.cpp:486` | 16K/tile4096 需 1 GiB device state；配 stable top-k 实测 209→269 tok/s。显式实验开关 |
| `LLAMA_LAYER_MAJOR_SPECULATIVE` | `0`（关） | 允许 dspark speculative full-batch prefill 进入 layer-major 执行器。`tools/server/server-context.cpp:3405` | **远程 CPU EP 实测为负收益，生产不得开启** |
| `LLAMA_LAYER_MAJOR_UBATCH` | `0`（跟随运行时 ubatch） | 覆盖 layer-major 自分块尺寸（≤16384）。`server-context.cpp:3409`、`src/llama-layer-major.cpp:661` | 受控实验；不能越过 KV/SWA 物理容量 |
| `LLAMA_LAYER_MAJOR_SCHED_COPIES` | `0`（关） | layer-major 请求 scheduler pipeline copies；仅在 `n_ubatch ≤ 512` 时生效，否则打 warning 禁用。`src/llama-context.cpp:451` | 短 prefill 实验 |
| `LLAMA_LAYER_MAJOR_RING_COMPAT` | `1`（开） | 紧凑环形 raw SWA cache 的 replay 兼容（按首趟逐 tile n_kv 回放，图输入逐字节一致）；`=0` 恢复旧行为：紧凑环上拒绝 layer-major。`src/llama-layer-major.cpp:710` | 仅诊断/回归对比 |

### 4.4 热专家 GPU 驻留（实验）

源码：`src/llama-hot-expert.{h,cpp}`。机制：按 `GGML_MOE_HOT_STATS`（§1.3）画像把每层 top-K 专家以紧凑 MXFP4 常驻一张 GPU。decode（小 n_tokens）时 MoE 块在图内 fork/join：custom op 把 router ids 重映射进紧凑 slot 空间，GPU 异步算热专家 FFN，CPU 链只算冷 slot。默认 merge 等 backend event 后取回逐 router-slot 输出，并按 slot 0..5 恢复基线左折叠顺序；`GGML_HOT_EXPERT_SLOT_ORDER=0` 才使用旧的 cold/hot 两 partial 相加。热侧 CUDA FFN 与 CPU repack FFN 仍不是逐位等价，最终质量需由配对 PPL 验证。

| 变量 | 默认 | 作用 |
|---|---|---|
| `GGML_HOT_EXPERT` | 关 | 总开关（`=1` 开启） |
| `GGML_HOT_EXPERT_TABLE` | 空（开启时**必填**，缺失报错） | 热专家 TSV 路径（`GGML_MOE_HOT_STATS` 产出格式） |
| `GGML_HOT_EXPERT_GGUF` | 空（开启时**必填**） | 模型 GGUF 路径（读取热专家权重来源） |
| `GGML_HOT_EXPERT_K` | `16` | 每层钉驻的热专家数；DSV4 最终单槽质量门使用 K24（12.8496 GiB 热权重），不得在未做显存与 PPL 门禁时机械照搬 |
| `GGML_HOT_EXPERT_DEV` | `CUDA1` | 热专家驻留设备 |
| `GGML_HOT_EXPERT_LAYERS` | `all` | 生效层：逗号表/区间（如 `42`、`30-42`） |
| `GGML_HOT_EXPERT_MAX_TOKENS` | `1` | GPU fork 处理的最大 n_tokens；当前 compact graph 固定单 token，设置 >1 会 warning 并钳制到 1；`0` 禁用 fork |
| `GGML_HOT_EXPERT_PACKED_IO` | 开 | 把 hidden、ids 和 router weights 按 256 B 对齐后合成一次 H2D；`=0` 回退三次独立 H2D。layer 42 callback ABBA 为 137.560 -> 134.432 us（+2.33%），43 层端到端收益仍待测 |
| `GGML_HOT_EXPERT_SLOT_ORDER` | **开** | `=1` 让 GPU 回传逐 router-slot 专家输出，由 CPU 按 slot 0..5 严格左折叠，恢复 baseline 求和顺序；`=0` 回退 cold/hot 两 partial 最后相加的旧路径，用于 A/B。当前严格路径限定 `MAX_TOKENS=1` |
| `GGML_HOT_EXPERT_SLOT_MERGE_AVX512` | **开** | x86 GCC/Clang 且 CPU 支持 AVX512F 时，跨 16 个 hidden rows 向量化 strict slot merge；slot 0..5 仍逐 slot 独立 `mul` 后 `add`，不使用 FMA。`=0` 强制 scalar 回退；其它平台自动 scalar |
| `GGML_HOT_EXPERT_REMOTE_EP` | 关 | `=1` 将CUDA hot slots与scheduled remote-EP合并；当前只接受单slot、`n_seq_max=1`、`KLOCAL=0`、同步REQ4、`WEIGHT_ON_MASTER=1`、`PIPE=0`。dealer只向CPU派cold slots，master按原slot顺序合并 |
| `GGML_HOT_EXPERT_MARKERS` | 关 | `=1` 输出稳定的 `[hotmarker] init/fork` 记录；init 显式包含 `slot_order=1/0`，供正式 benchmark meta 校验实际路径 |

当前已验收范围是单槽 decode。DSV4 K24 本地CPU cold路径的12轮结果为25.0167→30.2333 tok/s，PP2048=361.47 tok/s，5-chunk PPL 2.7758→2.7548。四路remote CPU cold桥接的raw/no-DSpark八轮为25.25→28.85 tok/s（+14.26%），paired PPL 2.7647→2.7412。后者仍是strict modulo cover，不是MAX_EFFORT。多槽并发尚未验收，因为 staging/event 状态仍需拆成per-context/per-slot所有权；当前in-flight原子门会拒绝并发覆盖。`SLOT_ORDER=0`只用于旧路径消融，不应作为生产提速开关。

---

## 5. 调度器（dspark / server 调度）

dspark（DSpark sidecar draft 投机解码）通过通用 speculative CLI 配置；调度旋钮是**命令行参数**而非环境变量。

| 参数 / 变量 | 默认 | 作用 | 何时用 |
|---|---|---|---|
| `--spec-type draft-dspark` | — | 使用 DSpark sidecar（带额外 Markov head 的 draft）。`common/arg.cpp` | DSV4 + dspark 生产配置；配 `-md dspark.gguf` |
| `--spec-draft-n-max N` | `3`（`common/common.h:326`） | 每步投机起草的 token 数。`common/arg.cpp:4108` | dspark 生产值 `2`（n-max 2 p-min 0 实测最优档） |
| `--spec-draft-p-min P` | `0.0`（`common/common.h:330`） | 贪婪接受的最小投机概率。`common/arg.cpp:4134` | 生产值 `0` |
| `--spec-draft-device DEV` | — | draft 模型放置设备。与 `-dev/-sm/-ts` 配合做双卡布局 | 双卡：target 层 + target KV 在 CUDA0，dspark 主体与 draft KV 在 CUDA1 |
| `LLAMA_DSPARK_CONF_PROFILE` | 关 | 设置即开：dspark 接受率/置信度画像日志。`common/speculative.cpp:942` | 调 n-max/p-min 时观测 |
| `LLAMA_SPC_PROF` | 关 | 设置（任意值）即开：speculative 流水线各阶段逐次计时，stderr 打 `SPC_PROF kind=<proc|draft> t_us=.. dur_us=.. n_tokens=..`。`common/speculative.cpp:2625` | 投机管线分段耗时观测 |
| `LLAMA_SERVER_PREFILL_CHUNK_SIZE` | `0`（adaptive） | server prefill 调度器固定 chunk 大小；0 = 自适应。`tools/server/server-context.cpp:1298` | 一般保持自适应 |

跨机 EP 的动态调度（dealer、端点、热点副本）变量全部在 §2.2（`GGML_REMOTE_EP_SCHED_*`）。

---

## 6. 调试与观测

全部为诊断开关，生产保持关闭（部分有明显性能扰动）。

| 变量 | 默认 | 作用 | 源码 |
|---|---|---|---|
| `GGML_OP_TIMING` | 关 | per-op 墙钟 profiler，退出时按耗时排序汇总 | `ggml/src/ggml-cpu/ggml-cpu.c:4552` |
| `GGML_MM_PHASE` | 关 | mul_mat 分阶段计时（src1 转换 vs gemm 本体） | `ggml-cpu.c:1873` |
| `GGML_COPY_TRACE` | 关 | 跨 backend 拷贝 ≥256 KB 打 `[copy-trace]` 日志 | `ggml/src/ggml-backend.cpp:493` |
| `GGML_SCHED_PROFILE` | 关 | scheduler 总时间、各 backend 输入/图时间、split 与跨 backend 传输汇总 | `ggml-backend.cpp:1767` |
| `GGML_SCHED_PROFILE_INPUTS` | 关 | 在上者中额外打印较大传输 tensor 名（日志量大） | `ggml-backend.cpp:1771` |
| `GGML_SCHED_DEBUG` | 关 | scheduler 调试输出 | `ggml-backend.cpp:2407` |
| `GGML_SCHED_DEBUG_REALLOC` | 关 | 强制 scheduler 重分配诊断路径 | `ggml-backend.cpp:2550` |
| `GGML_SCHED_TIMING` | 关 | 按 split 汇总 scheduler input / graph submit / post 墙钟时间 | `ggml-backend.cpp:891` |
| `GGML_SCHED_TIMING_SPLITS` | `0`（全部） | 只记录 split 数等于指定值的图 | `ggml-backend.cpp:898` |
| `GGML_SCHED_TIMING_SKIP` | `0` | 跳过前 N 次符合条件的图 | `ggml-backend.cpp:903` |
| `GGML_SCHED_TIMING_MIN_SAMPLES` | `1` | 少于该样本数时不输出 timing summary | `ggml-backend.cpp:908` |
| `GGML_CUDA_GRAPH_TIMING_INCLUDE_INPUTS` | 关 | CUDA graph timing 同时包含 scheduler 输入传输 | `ggml-backend.cpp:1961` |
| `GGML_CUDA_GRAPH_DEBUG` | 关 | CUDA graph 捕获、兼容性与 launch timing 诊断 | `ggml-cuda.cu:3263` |
| `GGML_CUDA_GRAPH_OPT` | 关（仅值 `1` 开） | CUDA graph stream-context 优化实验开关 | `ggml-cuda.cu:5231` |
| `GGML_OP_OFFLOAD_MIN_BATCH` | `32` | CUDA op-offload 的最小 batch；仅用于 A/B 和 scheduler 诊断 | `ggml-cuda.cu:6436` |
| `GGML_CUDA_AR_COPY_THRESHOLD` | `1048576` bytes | internal AllReduce copy 路径阈值 | `allreduce.cu:593` |
| `GGML_CUDA_AR_COPY_CHUNK_BYTES` | `0`（自动） | internal AllReduce 固定 copy chunk；非零时下限 256 KiB | `allreduce.cu:596` |
| `GGML_CUDA_AR_BF16_THRESHOLD` | `1` | F32 internal AllReduce 使用 BF16 round-trip 的字节阈值；`0` 关闭 | `allreduce.cu:605` |
| `GGML_HOT_EXPERT_DEBUG` | 关 | 热专家初始化、buffer 与 callback 诊断日志 | `src/llama-hot-expert.cpp:245` |
| `GGML_RPC_DEBUG` | 关 | RPC transport 层调试日志 | `ggml/src/ggml-rpc/transport.cpp:39` |
| `GGML_NUMA_EP_DEBUG` | 关 | EP claim 统计（见 §1.1，有 -18% TG 扰动） | `repack.cpp:6021` |
| `GGML_REMOTE_EP_DEBUG` / `_TRACE_ROUTER` / `_FREQ` | 关 | 跨机 EP 分段计时 / router trace / 频率画像（见 §2.4、§2.3） | `src/llama-remote-ep.cpp` |
| `GGML_EPD_OP_TIMING_EVERY` | 0 | worker op timing（见 §3.2） | `tools/epd/llama-epd.cpp:107` |
| `LLAMA_DECODE_TIMING` | 关 | 每次 decode 打 ctx 类型/token 数/耗时 | `src/llama-context.cpp:1556` |
| `LLAMA_NAN_DEBUG` | 关 | ubatch 级 NaN 检查 | `src/llama-context.cpp:1724` |
| `LLAMA_DSV4_STATE_DEBUG` | 关 | DSV4 状态 dump（同步 sched） | `src/llama-context.cpp:1759` |
| `LLAMA_DSV4_2KV` | 关 | `=old` 强制旧 `flash_attn_ext_2kv` 对比路径（CUDA 上已坏，仅 CPU 调试） | `src/models/deepseek4.cpp:965` |
| `LLAMA_TRACE` | 关 | server/loader 详细 trace | `tools/server/server-context.cpp:1281` |
| `LLAMA_SERVER_SLOTS_DEBUG` | 关 | server slot 调度调试 | `server-context.cpp:1290` |
| `LLAMA_DSPARK_CONF_PROFILE` | 关 | dspark 置信度画像（见 §5） | `common/speculative.cpp:942` |
| `LLAMA_SPC_PROF` | 关 | 投机流水线分段计时（见 §5） | `common/speculative.cpp:2625` |
| `GGML_MOE_HOT_STATS` | 关 | 专家命中画像（见 §1.3） | `repack.cpp:81` |
| `GGML_MOE_HOT_TRACE` | 关 | 有界 temporal router trace（见 §1.3） | `xllama-hot-trace.cpp` |
| `GGML_OFFLOAD_TRACE` | 关 | 设置即开：`_exps` 张量 offload 决策与 layer-major 路由 trace（`[lm-trace]` engaged/decline 原因、rollback 等） | `src/llama-model-loader.cpp:1236`、`src/llama-layer-major.cpp:766`、`src/llama-graph.cpp:2232` |
| `LLAMA_LAYER_MAJOR_PROFILE` | 关 | `=1` layer-major 执行器逐调用分段计时（store/sync/rewind/tile 计数） | `src/llama-layer-major.cpp:852` |
| `LLAMA_LAYER_MAJOR_DEBUG_SUM` | `0` | 逐层 hc checksum 定位非确定性漂移；`=2` 追加逐 tile checksum，`=3` 再 dump 层 1/tile 0 交接处原始 float | `src/llama-layer-major.cpp:906` |
| `LLAMA_DSV4_INPUT_PROFILE` | 关 | `=1` DSV4 `set_input` 各阶段（plan/raw 等）计时 | `src/llama-graph.cpp:1001` |
| `LLAMA_DSV4_COMPRESS_DEBUG` | 关 | `=1` 打印 DSV4 compressed KV plan 的 state 持久化/写入位置 | `src/llama-graph.cpp:841`、`src/llama-kv-cache-dsv4.cpp:692` |
| `GGML_CUDA_TOPK_OVERLAP_PROFILE` | 关 | `=1` 起 3×256 诊断 kernel 统计 batched top-k 结果在 8 行组间的重叠度 | `ggml/src/ggml-cuda/top-k.cu:302` |

---

## 7. CLI 参数

### 7.1 通用（common，llama-cli / llama-server / llama-bench 共用）

| 参数 | 状态 | 说明 |
|---|---|---|
| `--numa distribute/isolate/numactl` | 上游已有 | 线程 NUMA 分布策略；EP 生产配置用 `distribute`（配合 `GGML_NUMA_EP=1` 时钉核改为块划分） |
| `--numa mirror` | **本项目新增取值** | 每 NUMA 节点复制一份权重/KV，线程按节点 pinning，只读本地内存；耗 N 倍 RAM。**隐含 `--no-mmap`** |
| `--numa-mirror LIST` | **本项目新增** | mirror 复制内容：`weights,kv,all,none`（默认 `all`）；设置即隐含 `--numa mirror`。env 等效 `LLAMA_ARG_NUMA_MIRROR` |
| `-ncmoe, --n-cpu-moe N` | 上游已有 | 把 MoE 专家权重留在 CPU 的层数（99 = 全部）；与 `-ngl 99` 组合即「attention 在 GPU、专家在 CPU」的混合推理 |
| `--experts` 相关 | 上游已有 | 见上游 `llama-server --help` |
| `--repack` / `-nr, --no-repack` | 上游已有（common）；**llama-bench 的 `--no-repack` 为本项目新增** | 权重 repacking 开关（`no_extra_bufts`）。单机混合 TG 测速用 `--no-repack`（repack 反慢 ~13%）；PP / 双机 / 纯 CPU 保持默认开 |

### 7.2 llama-server / 投机解码（本项目常用组合）

`--spec-type draft-dspark`、`--spec-draft-n-max`、`--spec-draft-p-min`、`--spec-draft-device`、`-md/--model-draft` 等见 §5。其余为上游参数，不重复。

### 7.3 llama-epd（EP worker，整个工具为本项目新增，`tools/epd/llama-epd.cpp`）

```
llama-epd -m model.gguf --port 29200 --layers 3-42 [--experts 0-255 | --expert-mod R/N | --expert-list SPEC] [-t N] [--no-autotune] [--no-mmap]
llama-epd -m model.gguf --selftest [--selftest-layer N] [--selftest-tokens N]
```

| 参数 | 默认 | 说明 |
|---|---|---|
| `-m, --model PATH` | 必填 | GGUF 模型文件（与 master 同内容） |
| `--port N` | `29200` | 监听端口 |
| `--layers A-B` | 全部 | 认领的层区间（与 master `GGML_REMOTE_EP_LAYERS` 一致） |
| `--experts A-B` | 全部 | 认领的连续专家区间（闭区间，如 `0-63` 共 64 个） |
| `--expert-mod R/N` | — | 稀疏认领 `expert_id % N == R`；`0/4`…`3/4` 是无画像时的四路基线 |
| `--expert-list SPEC` | — | 任意稀疏专家集合（逗号 + 闭区间，如 `0,4,8-11,19`）；CAP 附带所有权位图。由 `tools/epd/ep-map-from-freq.py` 生成 |
| `-t, --threads N` | autotune；关闭时 `8` | 计算线程。**勿超物理核数**（超物理核严重劣化） |
| `--no-autotune` | 关 | 关闭启动线程自动标定（同 `GGML_EPD_AUTOTUNE=0`） |
| `--no-mmap` | 关 | 认领层专家权重启动一次性 pread 进匿名内存：RSS 全量常驻、零页入、免疫页缓存驱逐。**生产推荐**；需内存装得下认领层 |
| `--selftest` | 关 | 本地直算 vs loopback TCP 数值一致性检查后退出 |
| `--selftest-layer N` | 首个认领 MoE 层 | selftest 用层 |
| `--selftest-tokens N` | `4` | selftest token 数 |

配套工具（`tools/epd/`，不在本文档逐项展开，见该目录 README.md）：

- `ep-plan.py`：部署前按实测带宽/延迟给分层点
- `ep-map-from-freq.py`：从 `GGML_REMOTE_EP_FREQ_FILE` 画像生成各 worker 的 `--expert-list`（支持 `--extra-per-worker` 热点副本 + `--json`）
- `ep-map-from-trace.py`：从 `GGML_REMOTE_EP_TRACE_ROUTER` trace 生成 decode/verify 热点副本
- `bench-glm-master.sh` / `bench-glm-worker.sh`：双机启动脚本范例（含 env 组合）
- `ep-topo-run.sh` / `ep-topo-probe.c` / `ep-topo-gpu.cu`：拓扑探测
- 各 `ep-*-test.cpp`：协议/调度/拓扑单元测试

---

## 8. 已废弃或建议避免

| 参数 | 状态 | 原因 |
|---|---|---|
| `GGML_NUMA_EP_MMAP=1` | 避免 | 冷 page cache 下专家页错位钉死、PP 减半；改用 `--no-mmap`（`src/llama-model.cpp:1501`） |
| `GGML_REMOTE_EP_PIPELINE(+_CHUNK)` | 实验性 | 净收益 +0.7~1.4%，默认关 |
| `GGML_REMOTE_EP_SCHED_REPEAT_ACCOUNTING` | 多槽实验 | 单 slot 实测 -1.45%，默认关 |
| `GGML_EP_RDMA_SPIN` | 硬件相关 | 仅 master 开有 +2.7%；占核，必须逐机 A/B |
| `GGML_NUMA_FAKE_NODES` | 测试 only | 单节点机伪造 NUMA 拓扑 |
| `LLAMA_DSV4_2KV=old` | 调试 only | 旧 2kv 对比路径，CUDA 上已坏 |

## 9. AMX 加速路径（Sapphire Rapids+，Ice Lake 不可用）

构建（默认 OFF）：`-DGGML_AMX_TILE=ON -DGGML_AMX_INT8=ON -DGGML_AMX_BF16=ON`，需同时显式开 AVX512 家族（非 NATIVE 构建时）：`-DGGML_NATIVE=OFF -DGGML_AVX512=ON -DGGML_AVX512_VNNI=ON -DGGML_AVX512_VBMI=ON -DGGML_AVX512_BF16=ON -DCMAKE_CXX_FLAGS="-mavx512vpopcntdq -mgfni" -DCMAKE_C_FLAGS="-mavx512vpopcntdq -mgfni"`。

- 运行时门控：`arch_prctl(ARCH_REQ_XCOMP_PERM, XFEATURE_XTILEDATA)`——非 AMX 硬件上内核直接拒绝，AMX buft 自动缺席，权重落回 CPU_REPACK，行为与未编译 AMX 完全一致。
- 类型覆盖：F16/BF16/Q4_0/Q4_1/Q8_0/Q4_K/Q5_K/Q6_K/IQ4_XS（上游）+ **MXFP4、IQ2_XXS、IQ2_XS、IQ3_XXS、Q2_K**（本分支补全，含 MUL_MAT/MUL_MAT_ID 双算子）。
- 算子覆盖：MUL_MAT（上游）+ **MUL_MAT_ID MoE 专家路由**（本分支，per-expert tile 打包 + (slot,token) 路由映射）。
- 布局策略：MXFP4 保持 4bit packed（tile 加载时 LUT 解码，带宽敏感场景体积不涨）；IQ 系列/Q2_K 为 pack 时解码到 int8（体积约 4x 原格式，换内核零改动）。
- gemv/gemm 分流：M==1 自动走 AVX512-VNNI 内核（decode 带宽受限，AMX 无收益），M>1 走 tile gemm；无需手工阈值。
- 验证手段（无 AMX 硬件）：Intel SDE `sde64 -spr` 下 bit-exact/tol 对比，harness 参考 `test-amx-smoke.cpp` 模式。
- 未接：NUMA EP 行窗（GGML_NUMA_EP）对接 AMX buft（代码内有 TODO 对接点；128 行窗与 AMX 32 行块对齐已确认兼容）。

---

## 10. 附录：L3 实验/否决留档

集中留档**实验脚手架与已否决方向**（对应硬化方案 PROJECTHARDENING-PLAN §1.3 处置表）。约定：无特殊说明时代码保留、default-off，仅诊断/复现用，生产不开。

| 项 | 状态 | 证据与结论 |
|---|---|---|
| `GGML_NUMA_EP_PLACE=block` | 否决留档，default-off | 2 MiB block placement 端到端零收益（HANDOVER:1377-1383，commit 752c435fd；§1.1） |
| 三级 barrier（`HIER_BARRIER=2`）+ `GGML_NUMA_PIN_CORE` + `GGML_NUMA_BARRIER_GROUP` | 否决留档，default-off | Slice 10 否决（commit 35aa515b9，HANDOVER:1491-1497；§1.2） |
| `pocs/udnl-grid-mmid.cpp` | 否决留档 | Slice 6 的 32 核网格 GEMM 原型，胜率 0（HANDOVER:1395-1400） |
| `GGML_CPU_INT8_INTERMEDIATE`（INT8 激活岛） | 否决留档，default-off | 比 FP16 差一个量级（HANDOVER:1227 ②；§1.3） |
| `GGML_CUDA_FA_PV_Q8` | 否决留档，default-off | q8_0 V 的替代 P*V 路径，负结果（HANDOVER:1227 ④；`fattn-common.cuh:1692`） |
| `GGML_REMOTE_EP_PIPELINE(+_CHUNK)` | 留档，default-off | 净收益仅 +0.7~1.4%（§2.3） |
| `LLAMA_LAYER_MAJOR_SPECULATIVE` | 留档，default-off，**生产禁开** | 远程 EP 下负收益（§4.3） |
| `GGML_REMOTE_EP_SCHED_REPEAT_ACCOUNTING` | 留档，default-off | 单 slot -1.45%（§2.2） |
| `GGML_CUDA_ALLREDUCE=internal`（TP AllReduce） | 实验级保留 | 二轮 TP tg +49%（13.6→20.3）仍低于 layer 模式 25.9；联合 graph 捕获已否决（天花板 +3，HANDOVER:1503；§4.1） |
| 整层 `-ot` 跨设备钉卡 | 否决，无代码残留 | Slice 9：跨设备 -10%（HOT-EXPERT-GPU.md §4） |
| dspark 流水线化 / 半确定预取 / e144 / MXFP4 NR5..8 / signed-bias dealer / GLU-down fusion | 各 Slice 否决，代码已回退 | 无残留，仅留档 |
