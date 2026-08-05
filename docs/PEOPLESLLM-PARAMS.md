# PeoplesLLM 参数手册（local 分支相对 vendor 主线的新增可调参数）

> 适用范围：`llama-src` 仓库 `local` 分支（基线 `vendor`）。本文档由 `git diff vendor..local` 全量盘点生成，
> 只收录**本分支新增**的环境变量与 CLI 开关；主线（vendor）已有的参数在「主线已有、勿混淆」一节列出以便对照。
> 数据与实测结论出自 `/home/heiketu/x-llama.cpp/HANDOVER.md`（2026-07-31）。
>
> 约定：「默认」指不设置该环境变量 / 不传该开关时的行为。除特别注明外，环境变量 `=1` 开启、`=0` 或不设置关闭。

---

## 1. 快速配方表（按场景）

| 场景 | 推荐配置 | 要点 |
|---|---|---|
| 双路单机纯 CPU（DSV4/1M） | `CUDA_VISIBLE_DEVICES="" GGML_NUMA_EP=1 GGML_NUMA_HIER_BARRIER=1 GGML_NUMA_EP_GATE_UP_PARALLEL=1` + `--numa mirror --numa-mirror weights --no-mmap -ngl 0 -nkvo -ctk q8_0 -ctv q8_0`，线程数取物理核总数，`GGML_NUMA_EP_CHUNK` 不设 | EP 专家仍是单份行窗，只有非专家权重镜像；2x38C/251GiB 实测 RSS ~165GiB，TG512 12.38 tok/s。**CVD="" 是铁律**；不要固定逐核 pin |
| 单机混合 GPU+CPU（TG 为主） | `-ngl 99 -ncmoe 99 --no-repack`，线程 `-t 72` | 混合模式 TG 开 repack 反慢 ~13%；repack 的双向效应见「已知坑」 |
| 单机混合 GPU+CPU（PP 为主） | `-ngl 99 -ncmoe 99`（repack 默认开）`-t 72 --threads-batch 72` | PP 大批次 repack gemm 内核 ~3 倍速（实测 pp512 103→318） |
| 双 3090 超长 GPU Prefill | `GGML_CUDA_MOE_PP_MIN_TOKENS=2048 GGML_CUDA_MOE_PP_PREFETCH=3 GGML_CUDA_MOE_PP_DUAL=1` + `-ngl 99 -ncmoe 99 -fa 1 -b 4096 -ub 4096 --no-mmap` | 4096-token PP 同版本实测约 +20%；1M + F16 KV 必须允许按显存余量自动缩槽或降 `ub=2048`，禁止使用 `ub=8192` |
| 双机 DSV4（生产） | master：`GGML_REMOTE_EP=1 GGML_REMOTE_EP_HOST=10.0.0.2 GGML_REMOTE_EP_LAYERS=36-42`（有 RoCE 加 `GGML_REMOTE_EP_RDMA=1`）+ `-ngl 99 -ncmoe 99 -t 72 --numa mirror -fa 1 -b 4096 -ub 1024`，repack 保持默认开；slave：`llama-epd -m dsv4.gguf --port 29200 --layers 36-42 -t 72 --no-mmap` + `GGML_EPD_NUMA=weighted` | 分层 7 层（36-42）实测最优；**双机 repack 双项全胜不要关**；worker `--no-mmap` 必须配 `GGML_EPD_NUMA=weighted` |
| 双机 GLM-5.2 | master 同上但 `GGML_REMOTE_EP_LAYERS=3-17 -t 70 --no-mmap`；slave：`--layers 3-17` | slave 15 层（43.5G）是内存上限；**master 一律 `--no-mmap`**（mmap 冷缓存页错位钉死，PP 减半）；GLM 与 DSV4 不能同时跑 |
| 双机 decode / 长 PP 加速 | 在双机配置上加 `GGML_REMOTE_EP_MIRROR=1`（可选 `_LAYERS`/`_KREMOTE` 调层数与比例） | TG +9~11%、PP1020 +11~27%，代价 master 内存 +17~46G；**小档 PP（≤256 tok）回归，关掉即可** |
| 测速 bench | `llama-bench ... --no-mmap`，跑前 `sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'`，全程 `flock -x /tmp/xllama-bench.lock`，基线统一 `numactl --interleave=all` 口径 | A/B 必须同会话反序复测（~25% 运行顺序效应）；`--numa mirror` 双倍内存，连跑前必 drop_caches |

---

## 2. 环境变量详解

### 2.1 NUMA 系（16 个）

| 变量 | 默认 | 作用 | 何时用 / 注意 |
|---|---|---|---|
| `GGML_NUMA_EP` | 关 | 单机 NUMA 专家并行（行窗口）：每个 routed-expert 张量（`*_exps`）的**每个专家平面内**按行窗划分，节点 n 拥有 `[n*win, (n+1)*win)` 行（128 行对齐 mbind 单份放置）；repack mul_mat_id 计算侧按 per-(专家,节点) 计数器认领本节点窗口的行（`ggml/src/ggml-cpu/repack.cpp` + `src/llama-model.cpp: numa_ep_place_experts`）。需要 `--no-mmap` 加载（匿名内存才可迁移页）；**必须配合 80fe44321 起的亲和修复**（EP 下 DISTRIBUTE 钉核改块划分，否则半数线程全远程读） | 多 NUMA 节点纯 CPU / CPU-MoE 场景。可与 `--numa mirror --numa-mirror weights` 组合：专家仍单份行窗，非专家权重按节点复制 |
| `GGML_NUMA_EP_MMAP` | 关 | 允许在 mmap 加载的模型上做 **policy-only** 专家放置（`mbind(MPOL_BIND, flags=0)`，只设 VMA 策略、不迁移已缓存页） | **已知坑**：冷 page cache 下首次缺页按 interleave 落两节点后被钉死在错位节点，PP 减半。不要用，改用 `--no-mmap` |
| `GGML_NUMA_EP_STEAL_MIN_TOKENS` | `32` | NUMA EP 工作窃取的 token 阈值：批内 token 数 > 此值才跑「本地 + 窃取」两阶段协议，否则只做本地相位（静态按节点划分） | 一般不用调；与分层 barrier 的 TG 判据一致 |
| `GGML_NUMA_EP_STATIC` | 关 | 单相位（小批）时用**静态连续行窗划分**替代动态 claim：选中专家的本节点窗口拉平后按线程等分连续切片（NB_COLS 对齐），零原子、每线程长顺序流 | 实验开关；数值不变（每 dst 行仍恰算一次）。实测 op wall 更低但 barrier 等待更高，端到端不优于 claim，保持默认关 |
| `GGML_NUMA_EP_CHUNK` | 自动：TG `16`、PP `64` | 行窗 claim 的块行数（显式值须为 kernel 列块的倍数且整除 128，否则回落 16）。未设置时，`MUL_MAT_ID` 的 token 维 `<= 8` 使用 16，较大 batch 使用 64 | 原生 MXFP4 4K Hybrid：16/64/128 分别为 200.99/270.23/254.10 tok/s；显式环境变量仍可用于模型相关扫参 |
| `GGML_NUMA_EP_CLAIM` | 开 | `=0` 时保留行窗页放置但旁路 EP 计算路径（落回专家优先/默认路径） | **仅诊断**：归因 claim 路径 vs 页放置用，生产勿设 |
| `GGML_NUMA_EP_DEBUG` | 关 | 行窗 EP claim 诊断：每 256 次调用打印 wall/spread/busy/两节点认领行数/窃取统计到 stderr | **海森堡效应明显**（共享调试计数器热行，TG 约 -18%），只看相对结构别看绝对值 |
| `GGML_NUMA_HIER_BARRIER` | 关 | 纯自旋两级 NUMA 分级 barrier（先节点内、再跨节点）；在 `--numa mirror`，或 `--numa distribute` + `GGML_NUMA_EP=1` 的块状线程划分下生效 | 非 OpenMP 2x38C DSV4 实测应开启：浅层 TG 约 +8.6%，4.8k 深提示 TG 约 +6.2%；其他构建/拓扑仍需 A/B |
| `GGML_NUMA_EP_GATE_UP_PARALLEL` | 关 | DSV4 小 batch decode 下，将每个 NUMA 节点内线程对半分给独立的 gate/up 专家投影，并融合后续 clamp/GLU；要求 2 节点、线程数可被 4 整除、repack 权重且 token 维 ≤8，不满足自动回退 | 配合 EP + 分级 barrier；2x38C 实测 TG 再提升约 4.6%–6.2%，PP 自动走原路径 |
| `GGML_NUMA_MIRROR_THREADS` | `hardware_concurrency()` | `--numa mirror` 加载期做节点间权重复制的线程数 | 加载太慢时调大；一般默认即可 |
| `GGML_NUMA_MIRROR_BUDGET_GB` | `12` | PARTIAL 镜像（见下）每节点的镜像预算硬上限（GiB）；实际预算还受「最紧节点空闲内存 − 6GiB 保留」约束，防 MPOL_BIND 迁移触发 OOM-kill | 仅实验调参 |
| `GGML_NUMA_MIRROR_PARTIAL` | 关 | 只镜像「热张量」（排除 `_exps` 与 `token_embd` 的 host 张量，按大小优先、预算内尽量多），而不是全量镜像 | 内存不足以全量 mirror 时的折中；全量镜像内存不足时会自动回落到本模式并打日志 |
| `GGML_NUMA_MIRROR_MOE` | 关 | 只镜像含 routed 专家权重（`_exps`）的 buffer | 混合 CPU+GPU 模式：注意力在 GPU、只有 MoE 专家在 CPU 时，避免为 GPU 侧张量浪费一倍内存 |
| `GGML_NUMA_FAKE_NODES` | 关 | 单 NUMA 节点机器上伪造 N 个节点（1 < N ≤ 8），切分 node0 的内存范围 | **仅开发/测试** mirror/EP 代码路径用，生产勿设 |
| `GGML_NUMA_THP` | 关 | `=collapse` 时对 NUMA EP 绑定的专家区间做 `MADV_COLLAPSE` 同步折叠成 2MiB 大页 | **实测零收益**（EP 权重不在覆盖范围 + 碎片导致 MADV_COLLAPSE 全部 EAGAIN），保留但别用 |
| `GGML_KV_THP` | 关 | `=collapse` 时对 ≥2MiB 的 host KV buffer 做 THP 折叠（`src/llama-kv-cache.cpp`） | **实测负收益**（TG -6%/PP -55%），且 KV 本就在 GPU 上；别用。注意 HANDOVER 旧记录写 `GGML_KV_THP=1`，**现代码只认 `=collapse`** |

### 2.2 远程 EP 系（master 侧，`src/llama-remote-ep.cpp`，15 个）

| 变量 | 默认 | 作用 | 何时用 / 注意 |
|---|---|---|---|
| `GGML_REMOTE_EP` | 关 | 双机 expert-parallel 总开关。`build_moe_ffn` 对认领层经 custom op 发阻塞 RPC 给 EPD worker；未命中守卫自动回退本地，单机零影响 | 双机必开 |
| `GGML_REMOTE_EP_HOST` | `127.0.0.1` | worker 地址 | 双机设为 slave 直连 IP（如 `10.0.0.2`） |
| `GGML_REMOTE_EP_PORT` | `29200` | worker 端口 | 与 `llama-epd --port` 一致 |
| `GGML_REMOTE_EP_LAYERS` | 全部层 | 远端认领的层区间，格式 `A-B`、`A` 或逗号列表 `3-7,22-42`。区间内 `ffn_*_exps.weight` 在 master **不分配不读入**（真分片） | 分层由两端算力比决定而非内存（实测：DSV4 7 层 36-42 最优，GLM 15 层 3-17）；层区间必须落在 master 已 offload 给 CPU 的层内 |
| `GGML_REMOTE_EP_MIRROR` | 关 | 层镜像 + 专家 slot 维拆分：master 用空闲内存镜像远程层专家权重，MoE 沿 slot 维切 [0,k_r) 发 slave（只发不等）、[k_r,k) 本地算，wait op 按基线结合序合并 | decode/长 PP 大幅净胜（TG +9~11%、PP1020 +11~27%）；**小档 PP 回归**；与 SCHED 互斥（SCHED 优先） |
| `GGML_REMOTE_EP_MIRROR_LAYERS` | = `_LAYERS` | 镜像层子区间（`A-B` 或 `A`） | 内存不够镜像全部远程层时缩小范围 |
| `GGML_REMOTE_EP_MIRROR_KREMOTE` | `n_expert_used/2` | 发远端 slot 数 k_r | GLM 实测 kr 2/3/4 差异 ~1.5% 在噪声内，默认即可 |
| `GGML_REMOTE_EP_SCHED` | 关 | 专家级动态调度（P0）：多端点 CAP 协商（协议 v2，旧 worker 回 ERR 自动落回 classic），dealer 纯函数派单 + REQ2 逐 slot 不求和协议，slot 升序左结合合并（bit 级=基线） | 与 MIRROR **互斥**（同设时 SCHED 赢并打 warning）；slave 带宽降级期比 baseline 敏感 |
| `GGML_REMOTE_EP_SCHED_ENDPOINTS` | = `HOST:PORT` | 多端点列表，逗号分隔 `host:port,...` | 多 worker 时显式列出 |
| `GGML_REMOTE_EP_SCHED_KLOCAL` | `2` | m*：每 token 本地保留的 slot 数（≥1） | 负载均衡调参；P1 遗留 m* 扫描 |
| `GGML_REMOTE_EP_SCHED_PP` | 关 | 允许 n_tokens > 1（P4）；默认仅 decode（逐 token） | 实验性，未验收 |
| `GGML_REMOTE_EP_SCHED_DEAL` | — | `static` / `balance` | **当前两种模式用同一个确定性 dealer（P0），设置无实际差异** |
| `GGML_REMOTE_EP_PIPELINE` | 关 | 流水线分块投递：单层 token 维切块 + W=1 滑动窗口，worker 计算与 master 收发重叠；K<2（decode）自动走原路径 | 实测净收益仅 +0.7~1.4%（worker 修复后可重叠的不多），保持默认关 |
| `GGML_REMOTE_EP_PIPELINE_CHUNK` | `256` | 流水线切块大小（token 数）；自动封顶使单块 hidden 在飞行窗口内（TCP 3MiB / RDMA 1.5MiB，防死锁） | 仅配合 PIPELINE 使用 |
| `GGML_REMOTE_EP_DEBUG` | 关 | 每次 RPC 打 send/wait/compute 分段计时（master 与 worker 两侧） | 定位分层/延迟问题的第一手工具 |

### 2.3 RDMA 系（2 个）

| 变量 | 默认 | 作用 | 何时用 / 注意 |
|---|---|---|---|
| `GGML_REMOTE_EP_RDMA` | 关（=TCP） | RoCEv2 transport（rdma_cm 自动 GID，RC QP Send/Receive + 256KB 收发环）。三层 TCP 兜底：CMake 无 libibverbs 不编译 / 建连失败自动回退 TCP+warning / 默认零变化。master 与 worker 两侧都要设 | 有 RoCE 网卡（ConnectX-5 等）必开：跨机 64B RTT 42-74µs→10-13µs，尾延迟 ~1/4，GLM TG512 +7%、PP +10%。大帧 RNR 塌陷已修复（min_rnr_timer=0.01ms，16MB 帧 5.5GB/s 零停顿） |
| `GGML_EP_RDMA_SPIN` | 关 | busy-poll CQ 代替 completion channel | **debug knob**，勿在生产设置 |

### 2.4 EPD worker 系（`tools/epd/llama-epd`，6 个）

| 变量 | 默认 | 作用 | 何时用 / 注意 |
|---|---|---|---|
| `GGML_EPD_AUTOTUNE` | 开 | 启动时未显式 `-t` 则对 {16,24,32,48,物理核} 阶梯实测专家 FFN 取 knee（边际增益 <3% 即停），<0.1s | 实测 knee=72 或 48；`--no-autotune` 或 `=0` 关闭（关闭后默认 8 线程） |
| `GGML_EPD_NUMA` | `off` | worker 权重页 NUMA 放置策略：`interleave`（MPOL_INTERLEAVE）/ `weighted`（MPOL_WEIGHTED_INTERLEAVE，内核 ≥6.9），在任何权重分配/首触之前对进程生效 | **`--no-mmap` worker 必须 `=weighted`**，否则权重全落单节点、计算腰斩；不支持 weighted 的内核自动降级为 interleave + warning |
| `GGML_EPD_NUMA_WEIGHT` | 启动实测 | weighted 模式的节点权重比，`a:b` 或 `a,b`（每在线节点一个值）。不设则启动时做 ~150ms/节点 的带宽探针自动标定，再退回 sysfs 值 | 节点带宽不对称时手动指定 |
| `GGML_EPD_REPACK` | 开 | worker 侧专家权重启动时转 CPU_REPACK 交错布局，启用 repack gemv/gemm 内核；不匹配 traits 的张量保留原始布局 | 修复前 worker 权重是 vec_dot，双机 PP 慢 ~4.5 倍。`=0` 回退原始布局 |
| `GGML_EP_PREFAULT` | 关 | 启动预触专家权重页（消除冷缓存 mmap 磁盘页入尖峰） | 冷缓存尖峰率 30%→~1%，启动 +4.5s；**`--no-mmap` 开启时自动跳过（无意义）**，二者选 `--no-mmap` |
| `GGML_EP_PREFAULT_THREADS` | `16` | prefault 线程数 | 配合 PREFAULT 使用 |

### 2.5 融合与链式系（13 个）

**fused op 开关**（`src/llama-context.cpp`，全部默认开；设为 `0` 禁用单个融合 op，调试用）。加载时 probe 自动校验设备支持，不支持自动禁用并打日志：

| 变量 | 对应融合 op |
|---|---|
| `LLAMA_FUSED_GDN_AR` | Gated Delta Net（autoregressive 部分） |
| `LLAMA_FUSED_GDN_CH` | Gated Delta Net（chunk 部分） |
| `LLAMA_FUSED_LID` | Lightning Indexer（DSA 稀疏注意力） |
| `LLAMA_FUSED_DSV4_HC_PRE` | DeepSeek V4 HC 前段 |
| `LLAMA_FUSED_DSV4_HC_COMB` | DeepSeek V4 HC combine |
| `LLAMA_FUSED_DSV4_HC_POST` | DeepSeek V4 HC 后段 |
| `LLAMA_FUSED_DSV4_MOE_ROUTER` | DeepSeek V4 MoE router |

**微 op 链式执行阈值**（`ggml/src/ggml-cpu/ggml-cpu.c`：连续可链 op 串成一串由单线程执行，省掉每 op ~6-8µs 的线程分摊 + barrier 固定开销；元素数超过阈值则不链接、走并行路径）。设为正整数覆盖默认：

| 变量 | 默认 | 适用 op 族 |
|---|---|---|
| `GGML_CHAIN_MAX_DST_ELEMS` | `4096` | 流式 elementwise / reduction |
| `GGML_CHAIN_MAX_MATH_ELEMS` | `1024` | UNARY / GLU / SOFT_MAX（exp  bound，~4-8ns/elem） |
| `GGML_CHAIN_MAX_COPY_ELEMS` | `8192` | CONT / CPY / CONCAT（纯 memcpy） |
| `GGML_CHAIN_MAX_GATHER_ELEMS` | `2048` | GET_ROWS / SET_ROWS（离散行） |
| `GGML_CHAIN_MAX_SRC_ELEMS` | `65536` | 链节点任意计算型 src |
| `GGML_CHAIN_MAX_ROPE_ELEMS` | `8192` | ROPE / ROPE_BACK |

> 主线已有的 `GGML_CPU_DISABLE_FUSION`（=1 禁用 CPU fusion 框架）在 vendor 已存在，非本分支新增。

### 2.6 调试观测系（9 个）

| 变量 | 默认 | 作用 |
|---|---|---|
| `GGML_OP_TIMING` | 关 | `=1` 开 per-op 墙钟 profiler（线程 0 计时，含 barrier 等待），进程退出时（destructor）按耗时排序打印汇总。定位「op 消失/退化」问题（如 fused_lid 误伤时 FLASH_ATTN_EXT 从图里消失） |
| `GGML_MM_PHASE` | 关 | `=1` 打 mul_mat 分阶段计时（src1 转换 vs gemm 本体），仅线程 0 |
| `GGML_COPY_TRACE` | 关 | 设任意值即开：跨 backend 拷贝 ≥256KB 时打 `[copy-trace]` 日志（`ggml-backend.cpp`，标注 temporary instrumentation） |
| `GGML_SCHED_PROFILE` | 关 | `=1` 汇总 scheduler 总时间、各 backend input/graph 时间、split 数及跨 backend 传输方向/字节/时间 |
| `GGML_SCHED_PROFILE_INPUTS` | 关 | `=1` 在 scheduler profile 中额外打印较大传输 tensor 名称；日志开销较高，只用于定位 |
| `LLAMA_DECODE_TIMING` | 关 | 设任意值即开：每次 decode 打 ctx 类型 / token 数 / 耗时（ms） |
| `LLAMA_NAN_DEBUG` | 关 | `=1` 开 ubatch 级 NaN 检查（`src/llama-context.cpp`） |
| `LLAMA_DSV4_STATE_DEBUG` | 关 | `=1` 且 arch=DEEPSEEK4 时同步 sched 并 dump DSV4 状态 |
| `LLAMA_DSV4_2KV` | 关 | `=old` 强制走旧 `flash_attn_ext_2kv` 路径做对比（**CUDA 上已坏，仅 CPU 调试用**） |

---

### 2.7 GPU MoE prefill 流式（3 个）

| 变量 | 默认 | 作用 | 何时用 / 注意 |
|---|---|---|---|
| `GGML_CUDA_MOE_PP_MIN_TOKENS` | `0`（关） | 开启后把 `*_exps` 专家权重保存在原始布局的 CUDA pinned host buffer；达到阈值的 `mul_mat_id` 用单次全张量异步 H2D 后在 CUDA 上计算，低于阈值留在 CPU | 长 prompt、大 ubatch 的 PP 专项模式，建议从 `2048` 起测。会放弃专家权重 CPU_REPACK，因此不适合作为 decode 默认配置。未开启时 CUDA 会拒绝 offload `CPU_REPACK` 权重，避免 GPU 按原始布局读取交错数据 |
| `GGML_CUDA_MOE_PP_PREFETCH` | `0`（关） | `0..4` 个私有设备槽跨 split 预取专家权重；独立 H2D/commit stream 用 event 与计算流衔接 | 双 RTX 3090、ub4096 实测深度 1/2/3 相对关闭约 `+12.8%/+14.0%/+20.1%`。建议目标 `3`；每次分配保留至少 2GiB/卡，空间不足自动缩为 2/1/0，不能替代 1M 的 ubatch 预算 |
| `GGML_CUDA_MOE_PP_DUAL` | 关 | 流式模式下按层号在多张 GPU 间轮转专家计算 | 无独立预取时曾回归 1%-7%；配合 `PREFETCH=3` 后可利用两条 PCIe/NUMA H2D 路径。仍为显式开关，单卡勿设 |

### 2.8 CUDA / DSV4 执行实验（11 个）

| 变量 | 默认 | 作用 | 何时用 / 注意 |
|---|---|---|---|
| `GGML_CUDA_BATCHED_TOPK` | `0`（关） | `=1` 时对 NVIDIA CUDA 的 `k=512,nrows>=32` 使用每行一个 block 的 stable batched radix top-k；边界 ties 按较低索引稳定选择，其他形状和后端保持原实现 | 面向 DSV4 Lightning Indexer launch storm。`16384x4096,k=512` 单 op 21.59x，16K PP +8.24%，CUDA 456/456、非 CUDA 构建和重复 logits 已通过；补齐 8K/32K、生成文本和更多架构前仍不得设为全局默认 |
| `LLAMA_LAYER_MAJOR_DEVICE_HC` | `0`（关） | `=1` 时把 DSV4 layer-major 的完整 F32 HC layer-boundary state 保存在首个 GPU backend，层间用 D2D/P2P 传递；分配前保留至少 4 GiB 或 20% 显存，不满足时自动使用原 host HC | 16K/tile4096 需要 1 GiB device state；配合 stable top-k 和 FA KV lower bound，实测 209.00 -> 269.12 tok/s。当前使用每 tile 同步保证 split-copy 顺序，8K/32K/生成验收前保持显式实验开关 |
| `GGML_CUDA_MOE_PP_EP` | `0`（关） | 将同层 routed experts 沿 expert 轴拆到 CUDA0/CUDA1，两支各自在所属 backend 计算后归并；每 rank 用融合的 `MOE_WREDUCE` 按升序 slot 归并本地专家（跳过越界 slot 的 zero-fill 与读取），只跨卡传 `[n_embd,n_tokens]` partial | 双 3090/NVLink true-EP；必须配 `_EP_MIN_TOKENS`，单卡/P2P/OOM fallback 产品验收前保持显式开启 |
| `GGML_CUDA_MOE_PP_EP_MIN_TOKENS` | `2048` | true-EP 的最小 query batch | q1 decode 不进入 GPU EP；长 prefill 建议从 2048 起测 |
| `GGML_CUDA_MOE_PP_DEFER_PREFETCH` | `0`（关） | 将 expert slot 预取延后到 scheduler 已解析真实 view 权重后启动 | 当前 3-slot true-EP 基准使用；必须保留 slot 生命周期和失败回退 |
| `GGML_CUDA_MMQ_MOE_J` | `0`（自动） | 强制 MoE `MUL_MAT_ID` MMQ 的 tile 宽度 J（8..128 步进 8），仅用于 A/B 测量；自动模式按每专家典型行数（+25% 方差余量，向上取 16 倍）选择，n<=2048 的 MoE 调用单 op 快 1.1--2.3 倍 | 实验开关，不保证跨版本保留；n=4096 以上自动选择与原 J=128 一致 |
| `GGML_CUDA_P2P` | `0`（关） | 允许双卡 backend 使用 peer/NVLink copy | 本机 RTX3090 间为 NVLink；没有 P2P 能力时不得假定可用 |
| `GGML_CUDA_DSV4_KV_REUSE` | `0`（关） | 对 DSV4 `K=V` alias、512 维、64 列 FA specialization 保留完整 K shared tile供 V 阶段复用 | 16K true-EP 581.47 -> 604.23 tok/s，保持原 KQ 运算顺序与精确 logits；约需 100480 bytes dynamic shared memory |
| `LLAMA_DSV4_SPARSE_FA` | `0`（关） | 用 8-query union 组装 raw/compressed sparse physical rows 和逐 query mask | 只在 query batch `>=256` 建图，q1 自动 dense；会改变浮点归约分组，不能当作 bit-exact 优化 |
| `GGML_CUDA_DSV4_SPARSE_RAW_COMPACT` | `0`（关） | 在 sparse FA 内仅保留每组有效 raw span（最多 512 行），再追加 compressed union | `=1` 要求同时开启 sparse FA 且 query batch `>=256`；16K sparse 596.08 -> 754.60 tok/s，TG 不命中。`=2` 只用于强制小形状 CUDA 单测，生产勿用 |
| `LLAMA_DSV4_COMPACT_DECODE_SWA` | `0`（关） | layer-major 大 prefill 完成后，把 raw SWA 的最后 128 行搬到 256-cell 物理环，使 q1 decode 的 raw KV 图宽与 prompt 长度解耦 | 16K fixed TG64 `9.666 -> 11.801 tok/s`（+22.08%）；TG512 `11.298 -> 12.184 tok/s`（+7.84%）且完成环绕写回。prefill logits bit-exact，decode 会因 FA/stream-K 归约分组变化产生小数值差，完成 greedy/多序列/fallback 验收前仅作实验开关 |

### 2.9 Hybrid CPU-MoE 边界（2 个）

| 变量 | 默认 | 作用 | 何时用 / 注意 |
|---|---|---|---|
| `LLAMA_LAYER_MAJOR_CPU_MOE_REDUCE` | `1` | layer-major DSV4 命中 CPU_REPACK MoE 时，将 down 后 weighting 与 top-k reduction 保留在 CPU，只向 GPU 回传归约后 activation | 不命中模型/布局时无作用；`=0` 恢复 scheduler 默认放置。top-k=6 时每层主回传从 384 MiB 降到 64 MiB |
| `LLAMA_CPU_MOE_Q8_BOUNDARY_MIN_TOKENS` | x86-64: `128`；其他: `0` | 达到阈值时，在源 backend 把 DSV4 MXFP4 CPU_REPACK gate/up activation 精确量化为 Q8_0，CPU 两个投影复用同一份 Q8 | `=0` 关闭。CUDA 路径使用与 CPU 一致的 nearest-even 语义；16K 主 GPU->CPU activation 从 64 MiB/层降到约 17 MiB/层，最终 logits 指纹精确一致 |

## 3. CLI 开关详解

### 3.1 common（llama-cli / llama-server / llama-bench 等共用，`common/arg.cpp`）

| 开关 | 状态 | 说明 |
|---|---|---|
| `--numa mirror` | **本分支新增取值**（`distribute/isolate/numactl` 为主线已有） | 每 NUMA 节点复制一份模型权重/KV，各节点线程只读本地内存、线程按节点 pinning；耗 N 倍 RAM（N=节点数）。**隐含 `--no-mmap`**。加载前生效（`llama_numa_init` 提前到模型加载前调用） |
| `--numa-mirror LIST` | **本分支新增** | 选择 mirror 复制内容：逗号列表 `weights,kv,all,none`（默认 `all`）。设置即隐含 `--numa mirror`。等效 env：`LLAMA_ARG_NUMA_MIRROR` |

### 3.2 llama-bench

| 开关 | 状态 | 说明 |
|---|---|---|
| `--no-repack` | **本分支新增** | 关闭权重 repacking（`mparams.use_extra_bufts = false`）。单机混合 TG 测速用（repack 反慢 13%）；双机/PP 测速保持默认开 |
| `--numa mirror` | **本分支新增取值** | 同 common |

### 3.3 llama-epd（EPD worker，整个工具为本分支新增，`tools/epd/llama-epd.cpp`）

```
llama-epd -m model.gguf --port 29200 --layers 3-42 [--experts 0-255] [-t N] [--no-autotune] [--no-mmap]
llama-epd -m model.gguf --selftest [--selftest-layer N] [--selftest-tokens N]
```

| 开关 | 默认 | 说明 |
|---|---|---|
| `-m, --model PATH` | 必填 | GGUF 模型文件 |
| `--port N` | `29200` | 监听端口 |
| `--layers A-B` | 全部 | 认领的层区间（与 master `GGML_REMOTE_EP_LAYERS` 一致） |
| `--experts A-B` | 全部 | 认领的专家区间，半开 `[A,B)`（专家级切分用） |
| `-t, --threads N` | 启动 autotune；关闭时 `8` | 计算线程。**实测 = 物理核数最优，超物理核严重劣化**（slave 带宽 32 线程即饱和） |
| `--no-autotune` | 关 | 关闭启动线程自动标定（同 `GGML_EPD_AUTOTUNE=0`） |
| `--no-mmap` | 关 | 认领层专家权重启动时一次性 pread 进匿名内存：RSS 全量常驻、零页入、免疫页缓存驱逐。冷缓存尖峰 12.3%→1.8%。**推荐**；需 slave 内存装得下认领层；**必须配 `GGML_EPD_NUMA=weighted`** |
| `--selftest` | 关 | 本地 vs loopback 数值一致性检查后退出 |
| `--selftest-layer N` | 首个认领 MoE 层 | selftest 用层 |
| `--selftest-tokens N` | `4` | selftest token 数 |

### 3.4 主线已有、勿混淆（vendor 基线已存在，**不是**本分支新增）

| 参数 | 核实结论 |
|---|---|
| `--repack` / `-nr, --no-repack`（common） | **主线已有**（`params.no_extra_bufts`，env `LLAMA_ARG_REPACK`）。HANDOVER 里写的「`--no-extra-bufts`」**并不存在这个 CLI 名**——字段叫 `no_extra_bufts`，开关叫 `--no-repack` |
| `--cpu-moe` / `-ncmoe, --n-cpu-moe` | **主线已有**（common 与 llama-bench 均在 vendor 存在） |
| `--numa distribute/isolate/numactl` | 主线已有；本分支只加了 `mirror` 取值 |
| `GGML_CPU_DISABLE_FUSION`、`LLAMA_KV_CACHE_DEBUG`、`GGML_TOTAL_THREADS`、`KMP_BLOCKTIME`、`OMP_WAIT_POLICY`、`LLAMA_GRAPH_REUSE_DISABLE`、`LLAMA_ATTN_ROT_DISABLE` 等 | 主线已有 |

---

## 4. 已废弃 / 实验性参数清单（别用或仅限调试）

| 参数 | 状态 | 原因 |
|---|---|---|
| `GGML_NUMA_THP=collapse` | 保留但别用 | 实测零收益（MADV_COLLAPSE 全部 EAGAIN） |
| `GGML_KV_THP=collapse` | 保留但别用 | 实测负收益（TG -6%/PP -55%） |
| `GGML_NUMA_EP_MMAP=1` | 避免 | 冷 page cache 下专家页错位钉死、PP 减半（已知坑 #4） |
| `GGML_REMOTE_EP_PIPELINE`(+`_CHUNK`) | 实验性 | 实测净收益 +0.7~1.4%，默认关 |
| `GGML_REMOTE_EP_SCHED_PP` | 实验性 | P4 未验收，默认仅 decode |
| `GGML_REMOTE_EP_SCHED_DEAL` | 名义参数 | P0 阶段 static/balance 同一 dealer，无实际差异 |
| `GGML_EP_RDMA_SPIN` | debug only | CQ busy-poll 调试开关 |
| `GGML_NUMA_FAKE_NODES` | 测试 only | 单节点机伪造 NUMA 拓扑 |
| `LLAMA_DSV4_2KV=old` | 调试 only | 旧 2kv 对比路径，CUDA 上已坏 |
| `LLAMA_FUSED_*`（7 个） | 调试 only | 生产保持默认全开，仅排查融合 op 问题时单个置 0 |
| 配置层面的废弃方案 | — | DSV4 slave 12 层（31-42）与 9 层（34-42）方案实测全面劣于 7/8 层，已弃用；GLM slave 25 层重分层实测变差，维持 15 层 |

---

## 5. 已知坑（血泪清单）

1. **纯 CPU 必须 `CUDA_VISIBLE_DEVICES=""`**：CUDA 设备可见 + `-ngl 0` 时 fused_lid 探针把 LIGHTNING_INDEXER 分派给 CUDA 而 layer 在 CPU → device mismatch → 整个 DSA 稀疏注意力退化为分解路径（FLASH_ATTN_EXT/LIGHTNING_INDEXER/TOP_K 消失，+65ms/token barrier 等待，慢 ~2x）。混合模式（ngl 99）layer 在 CUDA 上不受影响。注意 `CUDA_VISIBLE_DEVICES=`（空值语法）在某些路径会静默挂死，用 `=""`。
2. **CUDA 构建的 buft 顺序（已修复 2c53e164c）**：修复前 CUDA 构建下纯 CPU/混合模式 CPU 专家权重落 pinned host buffer、matmul 退化为 vec_dot（慢 ~2x）。旧构建（build-cuda-stale-*、修复前的 build-epdev*）不要用于测速；slave 的纯 CPU 构建（build-cpu）无此问题。
3. **`--numa mirror` 的内存取决于组件**：默认 `all` 会按节点复制权重/KV；大模型应显式选 `--numa-mirror weights`。同时开启 `GGML_NUMA_EP=1` 时 routed experts 被排除，不复制专家，只镜像非专家权重。DSV4 1M/Q8 KV 实测 RSS ~165GiB；仍须启动前核对余量，bench 连跑前清 page cache。
4. **mmap + `GGML_NUMA_EP_MMAP=1` 页错位钉死**：`mbind(MPOL_BIND, flags=0)` 不迁移已缓存页，重启后冷 cache 首触按 interleave 落两节点后永远钉死，PP 减半（TG 每 token 只读 8 个专家天然容忍，故只有 PP 发病）。**GLM master bench/生产一律 `--no-mmap`**（PP1020 34→76，代价 TG512 ~6%）。
5. **repack 双向效应**：PP 大批次 → repack gemm 内核 ~3 倍速（必开）；单机混合 TG batch=1 → repack 反慢 ~13%（用 `--no-repack`）；**双机场景 repack 双项全胜**（master 本地 CPU matmul 占比小，无 TG 惩罚），保持默认。单机纯 CPU 的 TG 不受此惩罚。
6. **`--no-mmap` worker 必须配 `GGML_EPD_NUMA=weighted`**：否则 ~80G 权重全落单 NUMA 节点，计算腰斩。
7. **基线对比统一 interleave 口径**：双路机上原版性能随页缓存放置运气波动可达 2 倍（88G 模型页缓存倾斜 node0 时 tg64 16.64→10.12）；bench 一律 `numactl --interleave=all` + drop_caches。
8. **A/B 测量 ~25% 运行顺序效应**（后跑的快）：关键对比必须同会话反序复测（ABBA）。
9. **SCHED 与 MIRROR 互斥**：同设时 SCHED 优先、MIRROR 自动禁用（打 warning）。
10. **worker 线程勿超物理核**：`-t 128/136`（超 36 物理核）TG512 崩到 4-7 t/s 且非单调；交给 autotune 或设物理核数。
11. **PP 口径警告**：本仓库 PP 数字默认 5-token 短 prompt，固定开销摊薄严重，不代表长 prompt 吞吐；比较请用 63/254/1020 档摊销曲线。
12. **任何模型进程全程 `flock -x /tmp/xllama-bench.lock`**：多 agent 并发加载模型曾 OOM 杀整个会话。
13. **llama-cli 脚本化必须 `--single-turn` + `</dev/null`**：新版 tools/cli 交互主循环没有 EOF 退出路径——`-no-cnv` 止不住，stdin=</dev/null 时 readline 立即返回 false → 无限打印 `\n> ` 空转（实测写出 3GB/410MB 垃圾并占锁 17 分钟）；唯一可靠退出 = `--single-turn`（cli-context.cpp:655 break）。配套坑：llama-cli 用 `--no-mmap`、llama-bench 用 `--mmap 0`（互不认，报 "error: invalid argument: 0"）；llama-cli `-fa` 只认 on/off/auto 不认 `1`；空输出 cmp 会假阳性，对拍前必须 `wc -c` 验非空。
14. **CUDA 不得直接执行 CPU_REPACK 权重**：CPU_REPACK 是 8x8 交错布局，CUDA MMQ 按原始 GGUF 布局读取会产生错误结果。当前 CUDA offload 已显式拒绝该 buffer；长 PP 要上 GPU，请开启 `GGML_CUDA_MOE_PP_MIN_TOKENS`，让专家权重从加载阶段保持 pinned 原始布局。

---

## 6. 参数总账

| 组 | 数量 | 参数 |
|---|---|---|
| NUMA 系 env | 16 | `GGML_NUMA_EP`、`GGML_NUMA_EP_MMAP`、`GGML_NUMA_EP_STEAL_MIN_TOKENS`、`GGML_NUMA_EP_STATIC`、`GGML_NUMA_EP_CHUNK`、`GGML_NUMA_EP_CLAIM`、`GGML_NUMA_EP_DEBUG`、`GGML_NUMA_HIER_BARRIER`、`GGML_NUMA_EP_GATE_UP_PARALLEL`、`GGML_NUMA_MIRROR_THREADS`、`GGML_NUMA_MIRROR_BUDGET_GB`、`GGML_NUMA_MIRROR_PARTIAL`、`GGML_NUMA_MIRROR_MOE`、`GGML_NUMA_FAKE_NODES`、`GGML_NUMA_THP`、`GGML_KV_THP` |
| 远程 EP 系 env | 15 | `GGML_REMOTE_EP`、`..._HOST`、`..._PORT`、`..._LAYERS`、`..._MIRROR`、`..._MIRROR_LAYERS`、`..._MIRROR_KREMOTE`、`..._SCHED`、`..._SCHED_ENDPOINTS`、`..._SCHED_KLOCAL`、`..._SCHED_PP`、`..._SCHED_DEAL`、`..._PIPELINE`、`..._PIPELINE_CHUNK`、`..._DEBUG` |
| RDMA 系 env | 2 | `GGML_REMOTE_EP_RDMA`、`GGML_EP_RDMA_SPIN` |
| EPD worker 系 env | 6 | `GGML_EPD_AUTOTUNE`、`GGML_EPD_NUMA`、`GGML_EPD_NUMA_WEIGHT`、`GGML_EPD_REPACK`、`GGML_EP_PREFAULT`、`GGML_EP_PREFAULT_THREADS` |
| 融合与链式系 env | 13 | `LLAMA_FUSED_GDN_AR/GDN_CH/LID/DSV4_HC_PRE/DSV4_HC_COMB/DSV4_HC_POST/DSV4_MOE_ROUTER`、`GGML_CHAIN_MAX_DST/MATH/COPY/GATHER/SRC/ROPE_ELEMS` |
| 调试观测系 env | 7 | `GGML_OP_TIMING`、`GGML_MM_PHASE`、`GGML_COPY_TRACE`、`LLAMA_DECODE_TIMING`、`LLAMA_NAN_DEBUG`、`LLAMA_DSV4_STATE_DEBUG`、`LLAMA_DSV4_2KV` |
| GPU CUDA 实验 env | 13 | `GGML_CUDA_MOE_PP_MIN_TOKENS`、`GGML_CUDA_MOE_PP_PREFETCH`、`GGML_CUDA_MOE_PP_DUAL`、`GGML_CUDA_MOE_PP_EP`、`GGML_CUDA_MOE_PP_EP_MIN_TOKENS`、`GGML_CUDA_MOE_PP_DEFER_PREFETCH`、`GGML_CUDA_MMQ_MOE_J`、`GGML_CUDA_P2P`、`GGML_CUDA_BATCHED_TOPK`、`GGML_CUDA_DSV4_KV_REUSE`、`LLAMA_DSV4_SPARSE_FA`、`GGML_CUDA_DSV4_SPARSE_RAW_COMPACT`、`LLAMA_LAYER_MAJOR_DEVICE_HC` |
| **env 合计** | **72** | |
| common CLI | 2 | `--numa mirror`（新取值）、`--numa-mirror` |
| llama-bench CLI | 2 | `--no-repack`、`--numa mirror`（新取值） |
| llama-epd CLI | 10 | `-m/--model`、`--port`、`--layers`、`--experts`、`-t/--threads`、`--no-autotune`、`--no-mmap`、`--selftest`、`--selftest-layer`、`--selftest-tokens` |
