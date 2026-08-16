> **本文档已由 [PARAMETERS.md](PARAMETERS.md) 取代**（按场景重组、修正过时默认值）。保留此文件仅作历史参考。

# PeoplesLLM 参数手册（local 分支相对 vendor 主线的新增可调参数）

> 适用范围：`llama-src` 仓库 `local` 分支（基线 `vendor`）。本文档由 `git diff vendor..local` 全量盘点生成，
> 只收录**本分支新增**的环境变量与 CLI 开关；主线（vendor）已有的参数在「主线已有、勿混淆」一节列出以便对照。
> 数据与实测结论出自 `/home/heiketu/x-llama.cpp/HANDOVER.md`（更新至 2026-08-13）。
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
| 双机 DSV4（生产） | master：`GGML_REMOTE_EP=1 GGML_REMOTE_EP_HOST=10.0.0.2 GGML_REMOTE_EP_LAYERS=36-42`（有 RoCE 时 master/worker **两侧**都加 `GGML_REMOTE_EP_RDMA=1`）+ `-ngl 99 -ncmoe 99 -t 72 --numa mirror -fa 1 -b 4096 -ub 1024`，repack 保持默认开；slave：`llama-epd -m dsv4.gguf --port 29200 --layers 36-42 -t 72 --no-mmap` + `GGML_EPD_NUMA=weighted` | 分层 7 层（36-42）实测最优；**双机 repack 双项全胜不要关**；worker `--no-mmap` 必须配 `GGML_EPD_NUMA=weighted` |
| 双机 GLM-5.2 | master 同上但 `GGML_REMOTE_EP_LAYERS=3-17 -t 70 --no-mmap`；slave：`--layers 3-17` | slave 15 层（43.5G）是内存上限；**master 一律 `--no-mmap`**（mmap 冷缓存页错位钉死，PP 减半）；GLM 与 DSV4 不能同时跑 |
| 四 NUMA worker 真 EP（单 slot） | master：`GGML_REMOTE_EP=1 GGML_REMOTE_EP_SCHED=1 GGML_REMOTE_EP_SCHED_KLOCAL=0 GGML_REMOTE_EP_SCHED_MAX_EFFORT=1 GGML_REMOTE_EP_SCHED_PP=1 GGML_REMOTE_EP_SCHED_PP_REPEAT_COST=250 GGML_REMOTE_EP_PARALLEL_IO=1 GGML_REMOTE_EP_WEIGHT_ON_MASTER=1 GGML_REMOTE_EP_RECONNECT_TIMEOUT_MS=90000 GGML_REMOTE_EP_SCHED_ENDPOINTS=10.0.0.1:29202,10.0.0.1:29203,10.0.0.2:29200,10.0.0.2:29201`；master/worker 两侧都设 `GGML_REMOTE_EP_RDMA=1`，仅 master 试 `GGML_EP_RDMA_SPIN=1`；先用真实 workload 画像生成 `--expert-list` | master 不分配这些层的 routed-expert 权重；strict 位图必须恰好覆盖一次，max-effort 可重叠但不得有缺口。DSV4 43 层、NMAX=3、e128 图热态 37.921 tok/s；GLM-5.2 PP512：2 NUMA 24.13→4 NUMA 40.59 tok/s（1.682×）。当前 decode 用同步 REQ4，`GGML_REMOTE_EP_PIPE=0`；PP repeat-cost 250 同组 A/B +0.94%；90000 ms 重连窗已用 66 秒真实 worker 重启通过 |
| DSV4 + DSpark，每槽完整 1M | 吞吐/质量默认：`-c 1048576 -np 1 --no-kv-unified -fit off -fa on -b 256 -ub 256`（F16 KV）；其余使用上一行四路 EP 配置 | `-c` 是总 context，不是每槽。16K PP 269.36 tok/s，比旧 UB64 +14.4%，TG 基本不变。容量档可改 Q8 KV `-c 3145728 -np 3 -ctk q8_0 -ctv q8_0`；三槽均为完整 1M，`1M×4` 在 PP compute buffer 预留阶段 OOM |
| 双机 decode / 长 PP 加速 | 在双机配置上加 `GGML_REMOTE_EP_MIRROR=1`（可选 `_LAYERS`/`_KREMOTE` 调层数与比例） | TG +9~11%、PP1020 +11~27%，代价 master 内存 +17~46G；**小档 PP（≤256 tok）回归，关掉即可** |
| 测速 bench | `llama-bench ... --no-mmap`，跑前 `sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'`，全程 `flock -x /tmp/xllama-bench.lock`，基线统一 `numactl --interleave=all` 口径 | A/B 必须同会话反序复测（~25% 运行顺序效应）；`--numa mirror` 双倍内存，连跑前必 drop_caches |

---

### 1.1 每槽 1M 的容量与验收口径

非 unified KV 下，server 总 context 与并发槽数必须一起扩大：

```bash
N=3
llama-server ... -c $((1048576*N)) -np "$N" --no-kv-unified -fit off
curl -sS http://127.0.0.1:18108/slots | jq '.[] | {id,n_ctx}'
```

验收必须看到每项 `n_ctx=1048576`。历史 F16 容量基线如下：

| 配置 | GPU0/GPU1 已用 | 单请求 TG | 结论 |
|---|---:|---:|---|
| `1M×1` 默认分布 | 17,816 / 8,590 MiB | 35.778 tok/s | 基准 |
| `1M×2` 默认分布 | 21,382 / 12,474 MiB | 35.526 tok/s | 历史 F16 生产档；额外 slot 本身约 -0.7% |
| `1M×3` target/draft 改分布 | 最佳候选 19,940 / 21,258 MiB | 36.167 tok/s（对短 context -4.6%） | 历史 F16 可装载候选 |

当前后台选择 F16 `1M×1`、`-b 256 -ub 256`：16K PP 三轮均值 269.36 tok/s，固定
128-token 请求最终热态四轮均值 36.84 tok/s。2026-08-12 的 Q8 KV 容量档把 target
43 层和 target KV 固定 CUDA0，把 DSpark、draft KV 及 target/draft 共用的
embedding/output 放 CUDA1：

| 配置 | GPU0/GPU1 已用 | 标准请求热态 TG | 结论 |
|---|---:|---:|---|
| `1M×2` Q8、默认双卡 split | 19,106 / 9,900 MiB | 34.168 tok/s | 对照 |
| `1M×2` Q8、连续布局 | 16,094 / 12,300 MiB | 34.436 tok/s | 比默认 split +0.79% |
| `1M×3` Q8、连续布局 | 20,456 / 12,304 MiB | 34.260 tok/s | 容量优先档；比同布局两槽 -0.51% |
| `1M×4` Q8、连续布局 | 未完成加载 | — | CUDA0 申请 2.73GiB PP compute buffer 时 OOM |

连续布局命令须增加
`-dev CUDA0,CUDA1 -sm layer -ts 1,0 -ot '^token_embd\.weight$=CUDA1,^output.*=CUDA1'`
`--spec-draft-device CUDA1 -ctk q8_0 -ctv q8_0`。DSpark sidecar 不带自己的 embedding/LM
head，不能把两个 scheduler 的 device list 完全隔离；否则 draft graph 会在复用 CUDA0
`output.weight` 时触发 backend scheduler assert。

三槽并发 128-token 总吞吐约 41.6 tok/s，低于历史 F16 两槽均值 42.998，因此第三槽用于
会话容量，不是聚合吞吐提速。固定单槽请求连续 7 轮输出一致；三个独立算术 chat 顺序/并发
均正确返回 2/4/6。raw completion 在近似并列 logits 上可能随合批归约路径改变，验收应比较
任务语义和独立 chat，不应把 raw 字节差异直接判成 KV/slot 污染。当前
`llama-fit-params` 在 DSV4 remote-EP 组合上会 segfault，所以容量测试必须使用 `-fit off`
真实加载，不能依赖自动 fit 后静默缩 context。Q8 会改变 KV 精度，不保证与 F16 bit-exact；
本轮仅完成短请求稳定性与隔离验收，真正接近 1M 的质量回归仍需单独执行。

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

### 2.2 远程 EP 系（master 侧，`src/llama-remote-ep.cpp`，28 个）

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
| `GGML_REMOTE_EP_SCHED_KLOCAL` | `2` | m*：每 token 本地保留的 slot 数。`0` 为严格 pure EP：master 不加载目标层 routed-expert 权重，worker 所有权必须完整覆盖 | `0` 模式不允许静默回退；CAP/拓扑失败直接中止，避免在无权重 master 上算错 |
| `GGML_REMOTE_EP_SCHED_MAX_EFFORT` | 关 | 配合 `KLOCAL=0` 允许 worker 所有权位图重叠，启用热点专家副本；仍要求每个专家至少有一个持有者 | dealer 在持有者集合内按当前 token 的 endpoint 槽数择最轻者，并用稳定轮转打破平票 |
| `GGML_REMOTE_EP_SCHED_PP` | 关 | 允许 n_tokens > 1（P4）；默认仅 decode（逐 token） | GLM-5.2 PP512 已完成 2/4 NUMA worker A/B 与位级对拍；建议配 `ub=256` 和真实 PP 画像的热点副本映射 |
| `GGML_REMOTE_EP_SCHED_TG_ACTIVATION_COST` | `0` | TG 首次启用某 endpoint 的 fanout 惩罚；`1000` 等于一个中位专家 assignment 的虚拟成本 | 仅影响有多个 holder 的专家；PP/多 token 不使用。用于在并行计算与每请求固定分发成本之间 A/B 调优 |
| `GGML_REMOTE_EP_SCHED_TG_REPEAT_COST` | `250` | 同一次 TG 请求再次命中同一专家时的虚拟成本，范围 0–1000；1000=重新流一次完整权重，250=按新 stream 的 1/4 计 | 对应 worker 的 nr2..4 shared-weight AVX512 路径；当前 DSV4 生产值 250 |
| `GGML_REMOTE_EP_SCHED_PP_REPEAT_COST` | `1000` | 同一次 PP batch 再次命中同一专家时的边际虚拟成本，范围 0–1000；1000 保持原先每行按完整权重流计费 | 当前 DSV4 生产显式设 250；同组 6 轮均值 263.555 vs 261.108 tok/s（+0.94%）。只改变 holder 选择，不改变计算或归并顺序 |
| `GGML_REMOTE_EP_SCHED_REPEAT_ACCOUNTING` | 关 | 把 endpoint 在飞队列和服务率样本统一成“新专家 + repeat 边际成本”工作单位 | 仅保留给真多 slot/多 stream 调度实验；DSV4 单 slot ABBA：默认旧口径 36.280，开启 35.754 tok/s（-1.45%），生产保持关闭 |
| `GGML_REMOTE_EP_RECONNECT_TIMEOUT_MS` | `0` | SCHED endpoint 断连后等待 worker 重新监听并重发暂存请求的最长毫秒数；范围 0–300000，0 保持一次立即重连 | pure EP 常驻服务建议 `90000`，可覆盖当前 EPD 约 70 秒的权重加载/repack；重连后会强制核对 expert map、kernel ID 和 CAP 位，任一变化都拒绝继续；超时或能力变化仍会终止进程，因为 master 没有本地专家结果可正确完成该层 |
| `GGML_REMOTE_EP_WEIGHT_ON_MASTER` | 关 | 同步 SCHED 使用 REQ4/RESP4：worker 返回未乘 router weight 的逐 slot 向量，master 按全局 slot 顺序加权合并 | 减少 worker 操作和响应路径；端点不支持时整路回退 REQ2。与异步 `_PIPE=1` 不兼容，后者会自动关闭它 |
| `GGML_REMOTE_EP_SCHED_DEAL` | — | `static` / `balance` | **当前两种模式用同一个确定性 dealer（P0），设置无实际差异** |
| `GGML_REMOTE_EP_PIPE` | 关 | REQ3/RESP3 异步请求号协议；每个 endpoint 有后台接收线程，可支撑跨 slot 流水 | 会隐式启用 SCHED；不同于旧的单层 token 分块 `_PIPELINE` |
| `GGML_REMOTE_EP_PIPE_MAX_MIB` | `512` | 异步 PIPE 所有在飞请求的总响应/隐藏 staging 字节 credit 上限 | 只配合 `_PIPE=1`；过小会回压，过大会提高峰值内存 |
| `GGML_REMOTE_EP_PIPE_MAX_REQUESTS` | `256` | 异步 PIPE 在飞 endpoint request 数上限 | 只配合 `_PIPE=1`；范围必须为正数 |
| `GGML_REMOTE_EP_PARALLEL_IO` | 关 | 对独立 endpoint 并行 send/recv；响应仍按全局 slot 次序合并，不改变浮点结合顺序 | 四 worker 同步 REQ4 生产开启；传输实现未验证时保留关闭即可 |
| `GGML_REMOTE_EP_MERGE_THREADS` | `8` | SCHED 响应按 token 并行、按 slot 原顺序合并；token 数 <64 自动单线程，设 `1` 恢复原路径 | 只影响 PP merge，decode 不变；本机 1/4/8/16 线程扫频以 8 最优，同热态配对约 +1.9%，输出逐字节 MATCH |
| `GGML_REMOTE_EP_PIPELINE` | 关 | 流水线分块投递：单层 token 维切块 + W=1 滑动窗口，worker 计算与 master 收发重叠；K<2（decode）自动走原路径 | 实测净收益仅 +0.7~1.4%（worker 修复后可重叠的不多），保持默认关 |
| `GGML_REMOTE_EP_PIPELINE_CHUNK` | `256` | 流水线切块大小（token 数）；自动封顶使单块 hidden 在飞行窗口内（TCP 3MiB / RDMA 12MiB，防死锁） | 仅配合 PIPELINE 使用；RDMA worker 使用 64×256KiB 预投递接收环 |
| `GGML_REMOTE_EP_DEBUG` | 关 | 每次 RPC 打 send/wait/compute 分段计时（master 与 worker 两侧） | 定位分层/延迟问题的第一手工具 |
| `GGML_REMOTE_EP_TRACE_ROUTER` | 关 | 在 `_DEBUG=1` 时输出小批次 router 的专家共激活 trace | 只用于 `ep-map-from-trace.py` 生成 decode/verify 热点副本；日志量大，生产关闭 |
| `GGML_REMOTE_EP_FREQ` | 关 | 统计每个 SCHED/PIPE 图中 router 对 `(layer, expert)` 的选择次数 | 收集真实 workload 画像后交给 `ep-map-from-freq.py`；只做计数，不改变派单 |
| `GGML_REMOTE_EP_FREQ_FILE` | 空 | 设置 CSV 路径，写出 `layer,expert,count` 完整表；未设置时只在退出打印逐层摘要 | 正常程序析构时落盘；`llama-ep-crossslot` 在 `_Exit` 前显式 flush，保证 0–42 全层数据不丢 |

### 2.3 RDMA 系（3 个）

| 变量 | 默认 | 作用 | 何时用 / 注意 |
|---|---|---|---|
| `GGML_REMOTE_EP_RDMA` | 关（=TCP） | RoCEv2 transport（rdma_cm 自动 GID，RC QP Send/Receive + 256KB 收发环）。三层 TCP 兜底：CMake 无 libibverbs 不编译 / 建连失败自动回退 TCP+warning / 默认零变化。master 与 worker 两侧都要设 | 有 RoCE 网卡（ConnectX-5 等）必开：跨机 64B RTT 42-74µs→10-13µs，尾延迟 ~1/4。DSV4 单-slot 四 worker 实测仅将 slave 两路 TCP→RDMA：目标 TG 14.8→15.5（+4.7%），DSpark n-max=2 17.15→17.6（+2.6%）。大帧 RNR 塌陷已修复（min_rnr_timer=0.01ms，16MB 帧 5.5GB/s 零停顿） |
| `GGML_EP_RDMA_SPIN` | 关 | busy-poll CQ 代替 completion channel | Ice Lake + ConnectX-5 四路 EP 实测只在 master 开：阻塞 RDMA 36.935→37.921 tok/s（约 +2.7%）；worker 同时开反降至 37.659。它不是 RDMA 总开关，并会持续占核，必须逐机 A/B |
| `GGML_EP_RDMA_COALESCE` | 开 | RDMA 在发帧时将 header 和分散 payload 收集到最少的已注册 SEND slot；`=0` 回到逐段发送 | 真 RDMA loopback 与协议回归均通过；生产 ABBA 中 on/off 的单-slot TG 相同（34.795/34.804），不宣称 TG 提速。主要价值是减少多 slot 时 send-ring/WR 压力，故默认保留开启 |

### 2.4 EPD worker 系（`tools/epd/llama-epd`，18 个）

| 变量 | 默认 | 作用 | 何时用 / 注意 |
|---|---|---|---|
| `GGML_EPD_AUTOTUNE` | 开 | 启动时未显式 `-t` 则对 {16,24,32,48,物理核} 阶梯实测专家 FFN 取 knee（边际增益 <3% 即停），<0.1s | 实测 knee=72 或 48；`--no-autotune` 或 `=0` 关闭（关闭后默认 8 线程） |
| `GGML_EPD_AUTOTUNE_ROWS` | 自动 | 覆盖启动标定所用的 compact assignment 行数 | 不设置时按 top-k × ownership 比例估算；显式 `-t` 时 autotune 本身不运行 |
| `GGML_EPD_NUMA` | `off` | worker 权重页 NUMA 放置策略：`interleave`（MPOL_INTERLEAVE）/ `weighted`（MPOL_WEIGHTED_INTERLEAVE，内核 ≥6.9），在任何权重分配/首触之前对进程生效 | 一个 worker 跨两路 CPU 且使用 `--no-mmap` 时设 `weighted`；四路真 EP 是每个 worker 单独 `numactl --cpunodebind=N --membind=N`，此时保持 `off`，不要再二次 interleave |
| `GGML_EPD_NUMA_WEIGHT` | 启动实测 | weighted 模式的节点权重比，`a:b` 或 `a,b`（每在线节点一个值）。不设则启动时做 ~150ms/节点 的带宽探针自动标定，再退回 sysfs 值 | 节点带宽不对称时手动指定 |
| `GGML_EPD_REPACK` | 开 | worker 侧专家权重启动时转 CPU_REPACK 交错布局，启用 repack gemv/gemm 内核；不匹配 traits 的张量保留原始布局 | 修复前 worker 权重是 vec_dot，双机 PP 慢 ~4.5 倍。`=0` 回退原始布局 |
| `GGML_EPD_CPP_GATHER` | 开 | 小 TG 请求用串行紧凑 memcpy 收集 ragged hidden，避免另一次 GET_ROWS 图调度/barrier | `=0` 仅用于 A/B；大 batch 自动保留图侧 GET_ROWS |
| `GGML_EPD_FUSE_GATE_UP` | 开 | 兼容的 separate gate/up 在加载期融合为同一 repacked 张量 | 总权重字节不变，减少一次 MMID 调度和一次激活量化；`=0` 仅诊断 |
| `GGML_EPD_FUSE_CLAMP_SWIGLU` | 开 | DSV4 clamped SWIGLU 在 CPU 单 op 内完成 clamp + 激活 | `=0` 回到三节点参考图，仅用于正确性/A-B |
| `GGML_EPD_SHARED_Q8_MIN_TOKENS` | `2` | gate/up 不能融合时，从 N token 起共享一次 Q8 激活量化 | `0` 关闭，`1` 强制所有 batch；TG=1 默认仍各投影内部量化 |
| `GGML_EPD_POLL` | `50` | persistent CPU threadpool 的混合轮询等级（0–100） | slave 独占 NUMA node 保持默认；与 master 同机、共享 CPU 的 worker 建议从 `0` 起测，避免空闲 worker 忙等争核 |
| `GGML_EPD_GRAPH_CACHE_MAX_ROWS` | `64` | 只缓存不超过该行数的 MoE 计算图；大 PP ragged 形状重建图但复用单块 grow-only allocator | 保留 decode 小图的 ~0.2ms 建图收益，同时避免不同 PP 路由形状永久累计几十 GiB；GLM 连扫 ub128/256/512 后 RSS 不再增长 |
| `GGML_EPD_GRAPH_CACHE_MIB` | `512` | worker 计算图缓存总预算，按 LRU 腾挪 | 只影响小图缓存；大 PP 使用 grow-only allocator，不永久缓存每个 ragged 形状 |
| `GGML_EPD_NO_GRAPH_CACHE` | 关 | 置 1 完全禁用 worker 计算图缓存 | 仅诊断；生产保留默认的“小图缓存、大图复用 allocator”分层策略 |
| `GGML_EPD_HUGEPAGES` | 关 | repack 完成、原始副本释放后对大匿名专家 buffer 做 `MADV_HUGEPAGE` | 当前两机实测无净收益且内存压力曾触发 systemd-oomd，生产保持 0 |
| `GGML_EPD_MAX_SESSIONS` | `64` | worker 同时持有的 client session 上限，合法范围 1–4096 | 多 server/多 slot 共享 worker 时按连接数调整；不是 llama-server slot 数 |
| `GGML_EPD_OP_TIMING_EVERY` | `0` | 每 N 个请求输出一次 worker op timing | 诊断专用；0 关闭，输出会与计算互斥并增加开销 |
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

### 2.8 CUDA / DSV4 执行实验（16 个）

| 变量 | 默认 | 作用 | 何时用 / 注意 |
|---|---|---|---|
| `GGML_CUDA_BATCHED_TOPK` | `0`（关） | `=1` 时对 NVIDIA CUDA 的 `k=512,nrows>=32` 使用每行一个 block 的 stable batched radix top-k；边界 ties 按较低索引稳定选择，其他形状和后端保持原实现 | 面向 DSV4 Lightning Indexer launch storm。`16384x4096,k=512` 单 op 21.59x，16K PP +8.24%，CUDA 456/456、非 CUDA 构建和重复 logits 已通过；补齐 8K/32K、生成文本和更多架构前仍不得设为全局默认 |
| `LLAMA_LAYER_MAJOR_DEVICE_HC` | `0`（关） | `=1` 时把 DSV4 layer-major 的完整 F32 HC layer-boundary state 保存在首个 GPU backend，层间用 D2D/P2P 传递；分配前保留至少 4 GiB 或 20% 显存，不满足时自动使用原 host HC | 16K/tile4096 需要 1 GiB device state；配合 stable top-k 和 FA KV lower bound，实测 209.00 -> 269.12 tok/s。当前使用每 tile 同步保证 split-copy 顺序，8K/32K/生成验收前保持显式实验开关 |
| `LLAMA_LAYER_MAJOR_SPECULATIVE` | `0`（关） | 允许 server 在 DSpark speculative full-batch prefill 上进入 layer-major executor | 仅实验。远程 CPU EP 实测普通 UB256 263.17 PP / 37.70 TG；host-HC 185.97 / 29.71，device-HC 259.09 / 30.15 tok/s，均为负收益；当前生产不得开启 |
| `LLAMA_LAYER_MAJOR_UBATCH` | `0`（跟随运行时 ubatch） | 覆盖 layer-major 自分块尺寸，最大 16384 | 仅用于受控实验；不能越过 KV/SWA 物理容量。曾尝试在 `-ub 256` 上强制 2048/512，被 512-cell raw-SWA 容量门正确拒绝并回落普通路径 |
| `GGML_CUDA_MOE_PP_EP` | `0`（关） | 将同层 routed experts 沿 expert 轴拆到 CUDA0/CUDA1，两支各自在所属 backend 计算后归并；每 rank 用融合的 `MOE_WREDUCE` 按升序 slot 归并本地专家（跳过越界 slot 的 zero-fill 与读取），只跨卡传 `[n_embd,n_tokens]` partial | 双 3090/NVLink true-EP；必须配 `_EP_MIN_TOKENS`，单卡/P2P/OOM fallback 产品验收前保持显式开启 |
| `GGML_CUDA_MOE_PP_EP_MIN_TOKENS` | `2048` | true-EP 的最小 query batch | q1 decode 不进入 GPU EP；长 prefill 建议从 2048 起测 |
| `GGML_CUDA_MOE_PP_DEFER_PREFETCH` | `0`（关） | 将 expert slot 预取延后到 scheduler 已解析真实 view 权重后启动 | 当前 3-slot true-EP 基准使用；必须保留 slot 生命周期和失败回退 |
| `GGML_CUDA_MMQ_MOE_J` | `0`（自动） | 强制 MoE `MUL_MAT_ID` MMQ 的 tile 宽度 J（8..128 步进 8），仅用于 A/B 测量；自动模式按每专家典型行数（+25% 方差余量，向上取 16 倍）选择，n<=2048 的 MoE 调用单 op 快 1.1--2.3 倍 | 实验开关，不保证跨版本保留；n=4096 以上自动选择与原 J=128 一致 |
| `GGML_CUDA_P2P` | `0`（关） | 允许双卡 backend 使用 peer/NVLink copy | 本机 RTX3090 间为 NVLink；没有 P2P 能力时不得假定可用 |
| `GGML_CUDA_DSV4_KV_REUSE` | `0`（关） | 对 DSV4 `K=V` alias、512 维、64 列 FA specialization 保留完整 K shared tile供 V 阶段复用 | 16K true-EP 581.47 -> 604.23 tok/s，保持原 KQ 运算顺序与精确 logits；约需 100480 bytes dynamic shared memory |
| `LLAMA_DSV4_SPARSE_FA` | `0`（关） | 用 8-query union 组装 raw/compressed sparse physical rows 和逐 query mask | 只在 query batch `>=256` 建图，q1 自动 dense；会改变浮点归约分组，不能当作 bit-exact 优化。量化 KV 下该 op 由 CUDA 侧全宽 backing 物化兜底（仍是 GPU MMA，不落 CPU）；q1 decode 请用下面的 fused indexed / Q8 compact 默认路径 |
| `LLAMA_DSV4_FUSED_INDEXED_FA` | `0`（关） | raw/compressed 两段 KV 不经全宽 concat 直接进 sparse kernel（`flash_attn_ext_sparse_2kv`），q1 decode 免去每步全宽 KV 拷贝 | 仅 F16 KV 生效；`=1` decode+prefill，`=2` 仅 q1 decode。16K 实测 q1 sparse（ncols2=8 tile）输给 dense q1 16/32-head MMA：fixed TG64 -12%、TG512 -2%（2026-08-07），故暂不默认开；长上下文或 sparse 宽 head 组落地后再评估。`=3` q1 decode + 多流 decode（每流 1 token，opt-in），`=4` `=1` 语义 + 多流 decode；多流条件 `n_seqs_unq == n_tokens` 且非 coupled batch（coupled 保持 dense 回退），混合轮次等 token 切分的头 ubatch 同样命中量化 KV 请用 `LLAMA_DSV4_Q8_SPARSE_FA` 默认路径 |
| `LLAMA_DSV4_FUSED_INDEXED_FA` | `2`（q1 decode 开） | raw/compressed 两段 KV 不经全宽 concat 直接进 sparse kernel（`flash_attn_ext_sparse_2kv`），q1 decode 免去每步全宽 KV 拷贝 | 仅 F16 KV 生效；`=0` 关闭，`=1` 恢复旧的 decode+prefill 全开，`=2`（默认）仅 q1 decode，`=3` q1 decode + 多流 decode（每流 1 token，opt-in），`=4` `=1` 语义 + 多流 decode。多流条件：`n_seqs_unq == n_tokens` 且非 coupled batch（coupled 流数塌为 1，保持 dense 回退）；混合轮次等 token 切分的头 ubatch 同样命中。改变浮点归约分组，验收口径为指纹对拍 |
| `LLAMA_DSV4_Q8_SPARSE_FA` | `1`（开） | 量化 KV（如 q8_0）q1 decode 走 compact gather 路径：`get_rows` 块拷贝仅物化选中的 compressed 行，与 raw 窗口拼接后由 dense FA 仅对 compact concat 物化 F16 scratch | 仅 KV 非 F16 时生效；`=0` 回落 dense top-k mask 全扫，`=1`（默认）仅单流 q1，`=2` 追加多流 decode（每流 1 token，top_k 先 reshape 为 `[topk,1,ns,1]` 再 batched gather，fattn kernel 不变，opt-in）。任何不支持组合都不会落 CPU backend |
| `LLAMA_DSV4_COMPACT_KV` | `0`（关） | 对任意 KV 类型（含 F16）强制走 compact gather 路径 | `=1` 仅单流 q1，`=2` 追加多流 decode（与 `LLAMA_DSV4_Q8_SPARSE_FA=2` 同语义，opt-in）；quantized KV 默认已由 `LLAMA_DSV4_Q8_SPARSE_FA` 覆盖，一般不需要此开关 |
| `GGML_CUDA_DSV4_SPARSE_RAW_COMPACT` | `0`（关） | 在 sparse FA 内仅保留每组有效 raw span（最多 512 行），再追加 compressed union | `=1` 要求同时开启 sparse FA 且 query batch `>=256`；16K sparse 596.08 -> 754.60 tok/s，TG 不命中。`=2` 只用于强制小形状 CUDA 单测，生产勿用 |
| `LLAMA_DSV4_COMPACT_DECODE_SWA` | `1`（开） | layer-major 大 prefill 完成后，把 raw SWA 的最后 128 行搬到 256-cell 物理环，使 q1 decode 的 raw KV 图宽与 prompt 长度解耦；`=0` 恢复全宽行为 | 16K fixed TG64 `9.666 -> 11.801 tok/s`（+22.08%）；TG512 `11.298 -> 12.184 tok/s`（+7.84%）且完成环绕写回。prefill logits bit-exact，decode 会因 FA/stream-K 归约分组变化产生小数值差（语义等价、非 bit-exact）。多 slot 上下文（`-np>1`，raw 每序列独立 stream）与 cache 被其他序列占用时自动回退全宽语义 |

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
llama-epd -m model.gguf --port 29200 --layers 3-42 [--experts 0-255 | --expert-mod R/N | --expert-list SPEC] [-t N] [--no-autotune] [--no-mmap]
llama-epd -m model.gguf --selftest [--selftest-layer N] [--selftest-tokens N]
```

| 开关 | 默认 | 说明 |
|---|---|---|
| `-m, --model PATH` | 必填 | GGUF 模型文件 |
| `--port N` | `29200` | 监听端口 |
| `--layers A-B` | 全部 | 认领的层区间（与 master `GGML_REMOTE_EP_LAYERS` 一致） |
| `--experts A-B` | 全部 | 认领的连续专家区间，CLI 两端均包含（如 `0-63` 共 64 个） |
| `--expert-mod R/N` | — | 稀疏认领满足 `expert_id % N == R` 的专家；`0/4`…`3/4` 是无统计信息时的四路基线 |
| `--expert-list SPEC` | — | 任意稀疏专家集合，支持逗号和闭区间（如 `0,4,8-11,19`）。worker 紧凑加载这些全局专家平面，CAP 附带所有权位图 |
| `-t, --threads N` | 启动 autotune；关闭时 `8` | 计算线程。**实测 = 物理核数最优，超物理核严重劣化**（slave 带宽 32 线程即饱和） |
| `--no-autotune` | 关 | 关闭启动线程自动标定（同 `GGML_EPD_AUTOTUNE=0`） |
| `--no-mmap` | 关 | 认领层专家权重启动时一次性 pread 进匿名内存：RSS 全量常驻、零页入、免疫页缓存驱逐。冷缓存尖峰 12.3%→1.8%。**推荐**；需内存装得下认领层。跨 NUMA 单 worker 配 `GGML_EPD_NUMA=weighted`；每 NUMA 一个 worker 则用外部 `numactl --membind=N` |
| `--selftest` | 关 | 本地 vs loopback 数值一致性检查后退出 |
| `--selftest-layer N` | 首个认领 MoE 层 | selftest 用层 |
| `--selftest-tokens N` | `4` | selftest token 数 |

从 router 统计生成四路等容量稀疏表（输出可直接作为每个 worker 的
`--expert-list`）；增加热点副本时 master 同时开启 `SCHED_MAX_EFFORT`：

```bash
tools/epd/ep-map-from-freq.py ep-freq.csv --experts 256 --workers 4
tools/epd/ep-map-from-freq.py ep-freq.csv --experts 256 --workers 4 \
  --extra-per-worker 16 --json
```

工具按层做多维负载均衡并保持 primary shard 数量相同；`extra-per-worker` 是每路
副本上限，只加入能改善离线估算的热点副本。副本只改变持有者集合，在线 dealer
仍按当前 token 决定实际执行端点。DSV4 0–42 层实测画像中，模 4 最差层负载为
理想值的 1.928 倍，平衡 primary 为 1.072 倍，热点副本估算为 1.057 倍。

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
| `GGML_REMOTE_EP_SCHED_REPEAT_ACCOUNTING` | 多槽实验 | 单 slot 稳态 ABBA 为 36.280→35.754 tok/s（-1.45%），默认关；只有真多 stream 队列压力下重测胜出才考虑升级 |
| `GGML_REMOTE_EP_SCHED_PP` | 已验收、仍 opt-in | GLM PP512 2/4 worker A/B 与 DSV4 service 正确性已通过；默认仅 decode 是兼容策略，不再是“未验收” |
| `GGML_REMOTE_EP_SCHED_DEAL` | 名义参数 | P0 阶段 static/balance 同一 dealer，无实际差异 |
| `GGML_EP_RDMA_SPIN` | 硬件相关性能档 | 当前只在 master 开有 +2.7%，worker 不开；占核且必须 A/B，不能全局照抄 |
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
6. **`--no-mmap` 的 NUMA 策略取决于 worker 拓扑**：一个 worker 跨两个 NUMA node 时必须配 `GGML_EPD_NUMA=weighted`，否则 ~80G 权重会倾斜到单节点；四路真 EP 每 node 一个 worker 时应使用 `numactl --cpunodebind=N --membind=N`，不要再跨节点 weighted/interleave。
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
| 远程 EP 系 env | 30 | `GGML_REMOTE_EP`、`..._HOST`、`..._PORT`、`..._LAYERS`、`..._MIRROR`、`..._MIRROR_LAYERS`、`..._MIRROR_KREMOTE`、`..._SCHED`、`..._SCHED_ENDPOINTS`、`..._SCHED_KLOCAL`、`..._SCHED_MAX_EFFORT`、`..._SCHED_PP`、`..._WEIGHT_ON_MASTER`、`..._SCHED_DEAL`、`..._SCHED_TG_ACTIVATION_COST`、`..._SCHED_TG_REPEAT_COST`、`..._SCHED_PP_REPEAT_COST`、`..._SCHED_REPEAT_ACCOUNTING`、`..._RECONNECT_TIMEOUT_MS`、`..._PIPE`、`..._PIPE_MAX_MIB`、`..._PIPE_MAX_REQUESTS`、`..._PARALLEL_IO`、`..._MERGE_THREADS`、`..._PIPELINE`、`..._PIPELINE_CHUNK`、`..._DEBUG`、`..._TRACE_ROUTER`、`..._FREQ`、`..._FREQ_FILE` |
| RDMA 系 env | 3 | `GGML_REMOTE_EP_RDMA`、`GGML_EP_RDMA_SPIN`、`GGML_EP_RDMA_COALESCE` |
| EPD worker 系 env | 18 | `GGML_EPD_AUTOTUNE`、`GGML_EPD_AUTOTUNE_ROWS`、`GGML_EPD_NUMA`、`GGML_EPD_NUMA_WEIGHT`、`GGML_EPD_REPACK`、`GGML_EPD_CPP_GATHER`、`GGML_EPD_FUSE_GATE_UP`、`GGML_EPD_FUSE_CLAMP_SWIGLU`、`GGML_EPD_SHARED_Q8_MIN_TOKENS`、`GGML_EPD_POLL`、`GGML_EPD_GRAPH_CACHE_MAX_ROWS`、`GGML_EPD_GRAPH_CACHE_MIB`、`GGML_EPD_NO_GRAPH_CACHE`、`GGML_EPD_HUGEPAGES`、`GGML_EPD_MAX_SESSIONS`、`GGML_EPD_OP_TIMING_EVERY`、`GGML_EP_PREFAULT`、`GGML_EP_PREFAULT_THREADS` |
| 融合与链式系 env | 13 | `LLAMA_FUSED_GDN_AR/GDN_CH/LID/DSV4_HC_PRE/DSV4_HC_COMB/DSV4_HC_POST/DSV4_MOE_ROUTER`、`GGML_CHAIN_MAX_DST/MATH/COPY/GATHER/SRC/ROPE_ELEMS` |
| 调试观测系 env | 7 | `GGML_OP_TIMING`、`GGML_MM_PHASE`、`GGML_COPY_TRACE`、`LLAMA_DECODE_TIMING`、`LLAMA_NAN_DEBUG`、`LLAMA_DSV4_STATE_DEBUG`、`LLAMA_DSV4_2KV` |
| GPU CUDA 实验 env | 15 | `GGML_CUDA_MOE_PP_MIN_TOKENS`、`GGML_CUDA_MOE_PP_PREFETCH`、`GGML_CUDA_MOE_PP_DUAL`、`GGML_CUDA_MOE_PP_EP`、`GGML_CUDA_MOE_PP_EP_MIN_TOKENS`、`GGML_CUDA_MOE_PP_DEFER_PREFETCH`、`GGML_CUDA_MMQ_MOE_J`、`GGML_CUDA_P2P`、`GGML_CUDA_BATCHED_TOPK`、`GGML_CUDA_DSV4_KV_REUSE`、`LLAMA_DSV4_SPARSE_FA`、`GGML_CUDA_DSV4_SPARSE_RAW_COMPACT`、`LLAMA_LAYER_MAJOR_DEVICE_HC`、`LLAMA_LAYER_MAJOR_SPECULATIVE`、`LLAMA_LAYER_MAJOR_UBATCH` |
| **env 合计** | **98** | 2026-08-13 按上述分组更新；同名共享 debug 变量只计一次 |
| common CLI | 2 | `--numa mirror`（新取值）、`--numa-mirror` |
| llama-bench CLI | 2 | `--no-repack`、`--numa mirror`（新取值） |
| llama-epd CLI | 12 | `-m/--model`、`--port`、`--layers`、`--experts`、`--expert-mod`、`--expert-list`、`-t/--threads`、`--no-autotune`、`--no-mmap`、`--selftest`、`--selftest-layer`、`--selftest-tokens` |
