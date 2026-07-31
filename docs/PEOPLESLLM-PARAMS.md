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
| 单机纯 CPU | `CUDA_VISIBLE_DEVICES=""` + `numactl --interleave=all`（或 `--numa mirror`）+ 默认 repack 开 | **CVD="" 是铁律**（fused_lid 探针误伤，DSA 退化慢 ~2x）；纯 CPU 的 TG 不受 repack 惩罚 |
| 单机混合 GPU+CPU（TG 为主） | `-ngl 99 -ncmoe 99 --no-repack`，线程 `-t 72` | 混合模式 TG 开 repack 反慢 ~13%；repack 的双向效应见「已知坑」 |
| 单机混合 GPU+CPU（PP 为主） | `-ngl 99 -ncmoe 99`（repack 默认开）`-t 72 --threads-batch 72` | PP 大批次 repack gemm 内核 ~3 倍速（实测 pp512 103→318） |
| 双机 DSV4（生产） | master：`GGML_REMOTE_EP=1 GGML_REMOTE_EP_HOST=10.0.0.2 GGML_REMOTE_EP_LAYERS=36-42`（有 RoCE 加 `GGML_REMOTE_EP_RDMA=1`）+ `-ngl 99 -ncmoe 99 -t 72 --numa mirror -fa 1 -b 4096 -ub 1024`，repack 保持默认开；slave：`llama-epd -m dsv4.gguf --port 29200 --layers 36-42 -t 72 --no-mmap` + `GGML_EPD_NUMA=weighted` | 分层 7 层（36-42）实测最优；**双机 repack 双项全胜不要关**；worker `--no-mmap` 必须配 `GGML_EPD_NUMA=weighted` |
| 双机 GLM-5.2 | master 同上但 `GGML_REMOTE_EP_LAYERS=3-17 -t 70 --no-mmap`；slave：`--layers 3-17` | slave 15 层（43.5G）是内存上限；**master 一律 `--no-mmap`**（mmap 冷缓存页错位钉死，PP 减半）；GLM 与 DSV4 不能同时跑 |
| 双机 decode / 长 PP 加速 | 在双机配置上加 `GGML_REMOTE_EP_MIRROR=1`（可选 `_LAYERS`/`_KREMOTE` 调层数与比例） | TG +9~11%、PP1020 +11~27%，代价 master 内存 +17~46G；**小档 PP（≤256 tok）回归，关掉即可** |
| 测速 bench | `llama-bench ... --no-mmap`，跑前 `sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'`，全程 `flock -x /tmp/xllama-bench.lock`，基线统一 `numactl --interleave=all` 口径 | A/B 必须同会话反序复测（~25% 运行顺序效应）；`--numa mirror` 双倍内存，连跑前必 drop_caches |

---

## 2. 环境变量详解

### 2.1 NUMA 系（11 个）

| 变量 | 默认 | 作用 | 何时用 / 注意 |
|---|---|---|---|
| `GGML_NUMA_EP` | 关 | 单机 NUMA 专家并行：把每个 routed-expert 张量（`*_exps`）的页按 `e * n_nodes / n_expert` 映射绑到对应节点，计算侧每个线程只算本节点的专家（`ggml/src/ggml-cpu/ggml-cpu.c` + `src/llama-model.cpp: numa_ep_place_experts`）。需要 `--no-mmap` 加载（匿名内存才可迁移页） | 多 NUMA 节点纯 CPU / CPU-MoE 场景；与 `--numa mirror` 是两条不同路线（EP=划分不复制，mirror=全复制） |
| `GGML_NUMA_EP_MMAP` | 关 | 允许在 mmap 加载的模型上做 **policy-only** 专家放置（`mbind(MPOL_BIND, flags=0)`，只设 VMA 策略、不迁移已缓存页） | **已知坑**：冷 page cache 下首次缺页按 interleave 落两节点后被钉死在错位节点，PP 减半。不要用，改用 `--no-mmap` |
| `GGML_NUMA_EP_STEAL_MIN_TOKENS` | `32` | NUMA EP 工作窃取的 token 阈值：批内 token 数 > 此值才跑「本地 + 窃取」两阶段协议，否则只做静态按节点划分 | 一般不用调；与分层 barrier 的 TG 判据一致 |
| `GGML_NUMA_HIER_BARRIER` | 关 | 纯自旋两级 NUMA 分级 barrier（先节点内、再跨节点），仅在 `--numa mirror` 且 ≥2 节点、多线程时生效 | 实测 OpenMP 构建下与 GOMP 树形 barrier 无差异，保持默认关 |
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
| `GGML_EPD_REPACK` | 开 | worker 侧专家权重启动时转 CPU_REPACK 交错布局，启用 repack gemv/gemm 内核；不匹配 traits 的张量保留原始布局 | **注意：该变量所在改动尚未提交（工作区 WIP）**。修复前 worker 权重是 vec_dot，双机 PP 慢 ~4.5 倍。`=0` 回退原始布局 |
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

### 2.6 调试观测系（7 个）

| 变量 | 默认 | 作用 |
|---|---|---|
| `GGML_OP_TIMING` | 关 | `=1` 开 per-op 墙钟 profiler（线程 0 计时，含 barrier 等待），进程退出时（destructor）按耗时排序打印汇总。定位「op 消失/退化」问题（如 fused_lid 误伤时 FLASH_ATTN_EXT 从图里消失） |
| `GGML_MM_PHASE` | 关 | `=1` 打 mul_mat 分阶段计时（src1 转换 vs gemm 本体），仅线程 0 |
| `GGML_COPY_TRACE` | 关 | 设任意值即开：跨 backend 拷贝 ≥256KB 时打 `[copy-trace]` 日志（`ggml-backend.cpp`，标注 temporary instrumentation） |
| `LLAMA_DECODE_TIMING` | 关 | 设任意值即开：每次 decode 打 ctx 类型 / token 数 / 耗时（ms） |
| `LLAMA_NAN_DEBUG` | 关 | `=1` 开 ubatch 级 NaN 检查（`src/llama-context.cpp`） |
| `LLAMA_DSV4_STATE_DEBUG` | 关 | `=1` 且 arch=DEEPSEEK4 时同步 sched 并 dump DSV4 状态 |
| `LLAMA_DSV4_2KV` | 关 | `=old` 强制走旧 `flash_attn_ext_2kv` 路径做对比（**CUDA 上已坏，仅 CPU 调试用**） |

---

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
| `GGML_EPD_REPACK` | **WIP（未提交）** | 改动尚在工作区（另一任务开发中），语义可能变动 |
| 配置层面的废弃方案 | — | DSV4 slave 12 层（31-42）与 9 层（34-42）方案实测全面劣于 7/8 层，已弃用；GLM slave 25 层重分层实测变差，维持 15 层 |

---

## 5. 已知坑（血泪清单）

1. **纯 CPU 必须 `CUDA_VISIBLE_DEVICES=""`**：CUDA 设备可见 + `-ngl 0` 时 fused_lid 探针把 LIGHTNING_INDEXER 分派给 CUDA 而 layer 在 CPU → device mismatch → 整个 DSA 稀疏注意力退化为分解路径（FLASH_ATTN_EXT/LIGHTNING_INDEXER/TOP_K 消失，+65ms/token barrier 等待，慢 ~2x）。混合模式（ngl 99）layer 在 CUDA 上不受影响。注意 `CUDA_VISIBLE_DEVICES=`（空值语法）在某些路径会静默挂死，用 `=""`。
2. **CUDA 构建的 buft 顺序（已修复 2c53e164c）**：修复前 CUDA 构建下纯 CPU/混合模式 CPU 专家权重落 pinned host buffer、matmul 退化为 vec_dot（慢 ~2x）。旧构建（build-cuda-stale-*、修复前的 build-epdev*）不要用于测速；slave 的纯 CPU 构建（build-cpu）无此问题。
3. **`--numa mirror` 双倍内存**：N 节点复制 N 份权重/KV；88G 模型 bench 连跑前必须 `drop_caches`，否则 page cache + 双倍 buffer 触发 OOM（systemd-oomd 会杀桌面进程）。
4. **mmap + `GGML_NUMA_EP_MMAP=1` 页错位钉死**：`mbind(MPOL_BIND, flags=0)` 不迁移已缓存页，重启后冷 cache 首触按 interleave 落两节点后永远钉死，PP 减半（TG 每 token 只读 8 个专家天然容忍，故只有 PP 发病）。**GLM master bench/生产一律 `--no-mmap`**（PP1020 34→76，代价 TG512 ~6%）。
5. **repack 双向效应**：PP 大批次 → repack gemm 内核 ~3 倍速（必开）；单机混合 TG batch=1 → repack 反慢 ~13%（用 `--no-repack`）；**双机场景 repack 双项全胜**（master 本地 CPU matmul 占比小，无 TG 惩罚），保持默认。单机纯 CPU 的 TG 不受此惩罚。
6. **`--no-mmap` worker 必须配 `GGML_EPD_NUMA=weighted`**：否则 ~80G 权重全落单 NUMA 节点，计算腰斩。
7. **基线对比统一 interleave 口径**：双路机上原版性能随页缓存放置运气波动可达 2 倍（88G 模型页缓存倾斜 node0 时 tg64 16.64→10.12）；bench 一律 `numactl --interleave=all` + drop_caches。
8. **A/B 测量 ~25% 运行顺序效应**（后跑的快）：关键对比必须同会话反序复测（ABBA）。
9. **SCHED 与 MIRROR 互斥**：同设时 SCHED 优先、MIRROR 自动禁用（打 warning）。
10. **worker 线程勿超物理核**：`-t 128/136`（超 36 物理核）TG512 崩到 4-7 t/s 且非单调；交给 autotune 或设物理核数。
11. **PP 口径警告**：本仓库 PP 数字默认 5-token 短 prompt，固定开销摊薄严重，不代表长 prompt 吞吐；比较请用 63/254/1020 档摊销曲线。
12. **任何模型进程全程 `flock -x /tmp/xllama-bench.lock`**：多 agent 并发加载模型曾 OOM 杀整个会话。

---

## 6. 参数总账

| 组 | 数量 | 参数 |
|---|---|---|
| NUMA 系 env | 11 | `GGML_NUMA_EP`、`GGML_NUMA_EP_MMAP`、`GGML_NUMA_EP_STEAL_MIN_TOKENS`、`GGML_NUMA_HIER_BARRIER`、`GGML_NUMA_MIRROR_THREADS`、`GGML_NUMA_MIRROR_BUDGET_GB`、`GGML_NUMA_MIRROR_PARTIAL`、`GGML_NUMA_MIRROR_MOE`、`GGML_NUMA_FAKE_NODES`、`GGML_NUMA_THP`、`GGML_KV_THP` |
| 远程 EP 系 env | 15 | `GGML_REMOTE_EP`、`..._HOST`、`..._PORT`、`..._LAYERS`、`..._MIRROR`、`..._MIRROR_LAYERS`、`..._MIRROR_KREMOTE`、`..._SCHED`、`..._SCHED_ENDPOINTS`、`..._SCHED_KLOCAL`、`..._SCHED_PP`、`..._SCHED_DEAL`、`..._PIPELINE`、`..._PIPELINE_CHUNK`、`..._DEBUG` |
| RDMA 系 env | 2 | `GGML_REMOTE_EP_RDMA`、`GGML_EP_RDMA_SPIN` |
| EPD worker 系 env | 6 | `GGML_EPD_AUTOTUNE`、`GGML_EPD_NUMA`、`GGML_EPD_NUMA_WEIGHT`、`GGML_EPD_REPACK`(WIP)、`GGML_EP_PREFAULT`、`GGML_EP_PREFAULT_THREADS` |
| 融合与链式系 env | 13 | `LLAMA_FUSED_GDN_AR/GDN_CH/LID/DSV4_HC_PRE/DSV4_HC_COMB/DSV4_HC_POST/DSV4_MOE_ROUTER`、`GGML_CHAIN_MAX_DST/MATH/COPY/GATHER/SRC/ROPE_ELEMS` |
| 调试观测系 env | 7 | `GGML_OP_TIMING`、`GGML_MM_PHASE`、`GGML_COPY_TRACE`、`LLAMA_DECODE_TIMING`、`LLAMA_NAN_DEBUG`、`LLAMA_DSV4_STATE_DEBUG`、`LLAMA_DSV4_2KV` |
| **env 合计** | **54** | |
| common CLI | 2 | `--numa mirror`（新取值）、`--numa-mirror` |
| llama-bench CLI | 2 | `--no-repack`、`--numa mirror`（新取值） |
| llama-epd CLI | 10 | `-m/--model`、`--port`、`--layers`、`--experts`、`-t/--threads`、`--no-autotune`、`--no-mmap`、`--selftest`、`--selftest-layer`、`--selftest-tokens` |
