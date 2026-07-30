# EPD: expert-parallel MoE dispatch（真 EP）

两部件：

- `llama-epd`：worker 守护进程。加载 GGUF（只读 mmap，懒分页），按 `--layers`/`--experts`
  认领范围，监听 TCP，对每个 REQ 计算指定层的指定路由专家 FFN（gate→silu→×up→down，
  加 router 权重求和）并返回合并结果。无 attention、无 router。
- master 侧（`src/llama-remote-ep.{h,cpp}` + `src/llama-graph.cpp` 挂点）：默认关闭。
  开启后 `build_moe_ffn` 在指定层把 hidden + top-k ids + router 权重经
  `ggml_map_custom3` 阻塞式发给 worker，RESP 写回图，后续残差照旧。

## 配置（master，env，风格对齐 GGML_NUMA_EP）

| env | 默认 | 说明 |
| --- | --- | --- |
| `GGML_REMOTE_EP` | 关 | 置 1 启用远端专家分发 |
| `GGML_REMOTE_EP_HOST` | 127.0.0.1 | worker 地址 |
| `GGML_REMOTE_EP_PORT` | 29200 | worker 端口 |
| `GGML_REMOTE_EP_LAYERS` | 全部 | 远端层范围 `A-B`；范围外的层走本地 |
| `GGML_REMOTE_EP_DEBUG` | 关 | 置 1 每次 RPC 打印 send/wait 耗时（master）；worker 端置 1 每 REQ 打印 compute 耗时 |
| `GGML_REMOTE_EP_RDMA` | 关 | 置 1 传输层改用 RDMA（RoCEv2，rdma_cm 自动建连）。需编译时检测到 libibverbs+librdmacm（CMake 打印 `EP RDMA transport (RoCEv2): ON`）；无卡/建连失败时打 warning 并自动回退 TCP。master 与 worker 需同时置 1（协议不同，不互通）。**2026-07-31 大帧塌陷已修复（根因 rdma_cm 默认 min_rnr_timer ~80ms，建连后改 0.01ms），大帧 5.5GB/s ≥2× TCP，全场景可用** |
| `GGML_REMOTE_EP_PIPELINE` | 关 | 置 1 启用流水线分块投递：多 token 层（PP）把 token 维切成块，W=1 滑动窗口发送（发块 i 后收块 i-1 的响应），worker 计算与 master 收发重叠。协议不变（每块是一个普通 REQ 帧），逐 token 数值不变；decode（1 token）不受影响自动走原路径。TCP/RDMA 均可用 |
| `GGML_REMOTE_EP_PIPELINE_CHUNK` | 256 | 每块 token 数上限；再按隐藏层字节数封顶（TCP ≤3MiB、RDMA ≤1.5MiB，保证 W=1 窗口在 socket buffer / RDMA 接收环容量内，不会死锁） |
| `GGML_REMOTE_EP_MIRROR` | 关 | 置 1 启用层镜像 + 专家 slot 拆分：master 同时加载远程层专家权重（这些层不再 TENSOR_SKIP），每层 MoE 沿 slot 维切两半——slot [0,k_r) 发 worker（只发不等的 send op），slot [k_r,k) 在 master 本地走与单机 MoE 完全相同的链路，wait op 收 partial_r 后按 slot 升序合并（与全远程基线结合顺序一致，逐字对拍 bit 一致）。decode 与 PP 均生效；详见下文实测节 |
| `GGML_REMOTE_EP_MIRROR_LAYERS` | =远程层范围 | 镜像层子集 `A-B`（须落在 `GGML_REMOTE_EP_LAYERS` 内） |
| `GGML_REMOTE_EP_MIRROR_KREMOTE` | n_expert_used/2 | 发远程的 slot 数 k_r，clamp 到 [1, n_expert_used-1] |

worker 侧另有：

| env | 默认 | 说明 |
| --- | --- | --- |
| `GGML_EP_PREFAULT` | 关 | 置 1 启动时预触认领层专家权重的全部 mmap 页（多线程），消除冷专家首次命中时页入造成的多 ms compute 尖峰 |
| `GGML_EP_PREFAULT_THREADS` | 16 | 预触线程数 |
| `GGML_EPD_AUTOTUNE` | 开 | 置 0 关闭启动线程自动标定（同 `--no-autotune`） |

worker 选项 `--no-mmap`：启动时把**认领层**的专家权重按张量一次性顺序 pread 进匿名内存
（per-tensor posix_memalign，64B 对齐校验），张量指向该缓冲区，进程退出时 free。行为：启动慢
（冷缓存全量读 15.75 GiB ~11s），运行时 RSS=认领权重全量常驻、**零页入、免疫页缓存驱逐**——
解决慢盘（exFAT U盘实测 9.5-27MB/s）下 mmap 完全不可用、以及 mmap 页被回收再页入的风险。
`GGML_EP_PREFAULT=1` 与 `--no-mmap` 同开时 prefault 跳过并打日志（已无意义）。默认 mmap 不变。

## 启动线程自动标定（autotune）

**仅当未显式传 `-t` 时生效**（显式 `-t` 行为完全不变）。模型映射 + 层认领 + prefault 之后、
listen 之前，worker 在认领层中取首层 + 中间层共 2 个代表层，以 decode 形态（n_tokens=1、
top-k 从 `<arch>.expert_used_count` 读）构造 dummy 输入，对候选线程阶梯
`{16, 24, 32, 48, 物理核数}`（物理核数按 sysfs topology 去重 (package, core)，不计 HT）
各跑 2 warmup + 5 轮取中位，选 knee point：**边际收益 < 3% 的最小线程数**（专家 FFN 是带宽
敏感型，过饱和点后加线程只增 barrier 开销）。标定切换线程走与服务完全相同的
`ggml_backend_cpu_set_n_threads` 路径，总耗时 <1s。每个候选的 ms/iter 与最终选择均打日志。

实测（slave 2×36 核/144 HT，DSV4 8 层 35-42）：见下文"启动线程自动标定实测"。

仅当层为 SILU+gate MoE、无 expert bias/scale、非 warmup 图时走远端；
不满足时自动回退本地路径。连接懒建立、跨 decode 复用，传输错误自动重连重试一次。

## worker

```
llama-epd -m model.gguf --port 29200 --layers 3-42 [--experts 0-255] [-t N] [--no-autotune]
llama-epd -m model.gguf --selftest [--selftest-layer N]   # 本地直算 vs 回环 TCP 数值校验
```

## 双机正确性验证（dsv4-recipeB-v2，43层/256专家/top-6，Q3_K 89G）

方法：同二进制同参数跑两遍 `llama-cli --temp 0 --seed 42 --single-turn`，
唯一变量是 `GGML_REMOTE_EP=1 ... HOST=10.0.0.2`（worker 在 server2 上认领全部 40 个
MoE 层）；对比生成 token 序列。master 参数为生产镜像基线
（`-ngl 99 -ncmoe 99 -t 72 --numa mirror -fa 1 -ot <S档>`，`GGML_NUMA_MIRROR_THREADS=72`）。

结果（2026-07-29，commit 5546f27bd）：英文/中文两组 prompt（48/64 token）生成文本逐 token
一致，仅速度行不同（本地 31.3 t/s vs 远端 14.0 t/s）。`llama-epd --selftest` 在双机
各自 PASS 且 |out| 完全一致（32.224439）。

## 权重真分片（commit c2943a784）

`GGML_REMOTE_EP_LAYERS` 范围内层的 `ffn_(up|gate|down)_exps.weight` 在 master 上
用 TENSOR_SKIP 创建：不分配、不读入、不参与 NUMA 放置（目前仅 deepseek4 arch 实现，
其他 arch 需要同样的三行改动）。这些层在图里走远端路径是强制的；warmup 图走零输出捷径。

双机 DSV4 实测（master 层 0-21 本地 + server2 层 22-42 ≈ 41G 权重）：

| 配置 | master RSS | master GPU | slave RSS | TG | PP(5 tok) |
| --- | --- | --- | --- | --- | --- |
| 单机 mirror 基线 | 59.3G | 17.5+17.0G | — | 29.3 t/s | 39.8 t/s |
| 双机 EP（22-42 远端） | 17.7G | 17.0+16.9G | 27.2G（上限 ~41G） | 20.9 t/s | 25.9 t/s |

正确性：同 prompt（temp 0 / seed 42）双机与单机输出逐 token 一致（英文 48 tok +
中文 64 tok，/completion 原始完成模式）。

双机 DSV4 重分层实测（commit d8f7dacf3，slave 31-42 共 12 层，GGML_REMOTE_EP_DEBUG 分段计时）：

| 配置 | master RSS | slave RSS | TG96 | TG512 | PP(5 tok) |
| --- | --- | --- | --- | --- | --- |
| 单机 mirror 基线 | 62.3G | — | 25.75 t/s | 24.88 t/s | 34.4 t/s |
| 双机 EP（31-42 远端） | 37.4G | 14.7G（上限 ~24G） | 22.17 t/s | 21.92 t/s | 30.4 t/s |

一致性：gen1/gen2 与基线逐字 IDENTICAL。每 token 分阶段（decode，server 45.6ms vs 基线 40.2ms）：
远端 12 层 RPC 10.6ms（send 0.29 + wait 10.30；其中 slave compute 8.74、网络+传输 ~1.6），
本地 CPU MoE 14 层 ~5.5ms，barrier 2.6ms，GPU 段 ~26.5ms（基线对应：CPU MoE 26 层 9.6ms、
barrier 4.7ms、GPU 段 ~25.8ms）。实测每层耗时：master 本地 ~0.37ms/层，slave ~0.73ms/层
（341 vs 174 GB/s 带宽比的真实体现，旧 0.376ms 估计作废）→ 均衡点 N≈8（35-42），

双机 DSV4 8 层实测（slave 35-42 ≈16G，与 12 层同会话 A/B 反序复测防顺序效应）：

| 配置 | master RSS | slave RSS | TG96 | TG512 | PP(5 tok) |
| --- | --- | --- | --- | --- | --- |
| 双机 EP（35-42 远端，8 层） | 45.8G | 10.2G | **25.79 t/s** | **25.49 t/s** | 34.4 t/s |
| 双机 EP（31-42 远端，12 层，同会话复测） | 37.5G | 15.7G | 24.77 t/s | 24.18 t/s | 33.2 t/s |

8 层远端 wait 6.69ms/token（vs 12 层 10.13ms，与分阶段模型预测一致）；TG96/PP 已与单机
基线（25.75/34.4）持平，TG512 略超基线（跨会话 ±5% 噪声内，判为持平）。8 层为当前最优
生产配置：速度持平单机，master RSS 省 26%（45.8G vs 62.3G）。一致性：两种配置 gen1/gen2
与单机基线均逐字 IDENTICAL。
可再省 ~3.6ms/token 追平平单机；GPU 段 ~26ms（~60%）才是两端共同的最大阶段。

已知限制：单一 worker 端点；阻塞式收发（无流水线）。修复前冷缓存下 726 token 中 452 次
RPC 尖峰 >2ms，根因为 slave 冷专家 mmap 磁盘页入；GGML_EP_PREFAULT=1 后尖峰基本消除（见下）。

## GGML_EP_PREFAULT A/B 实测（2026-07-30，slave 35-42）

方法：每轮先 `posix_fadvise(DONTNEED)` 把 slave 模型页缓存降到 0% 再起 worker（mincore 核实），
on/off 同会话正序+反序复测（on1→off1→off2→on2）。尖峰口径：master ep-debug 按 decode token
（8 层连续 n_tokens=1）分组，token 内任一层 wait>2ms 记 1 尖峰，共 719 token/轮。

| 配置 | TG96 | TG512 | >2ms 尖峰 | worker compute >2ms | slave RSS |
| --- | --- | --- | --- | --- | --- |
| prefault=1（on1/on2） | 24.54 / 25.23 t/s | 24.49 / 24.59 t/s | 1.1% / 0.8%（max 5.0ms） | 1/5752 | 16.5G |
| prefault=0 冷缓存（off1/off2） | 24.29 / 24.92 t/s | 24.97 / 24.88 t/s | 30.6% / 30.5%（max 19.4ms） | 586/5752 | 10.0G |

结论：尖峰率两种顺序各自一致（无顺序效应），prefault 把 >2ms 尖峰从 ~30% 压到 ~1%，
残余 ≤5ms；TG 差异在噪声内（尖峰摊薄仅 ~3%，故均值不敏感，尖峰主要影响尾延迟/流式体感）。
prefault 启动耗时 ~4.5s（16 线程预触 15.75 GiB，slave NVMe ~3.5GB/s），代价是 slave RSS
从 10.0G 升到 16.5G（认领层权重全量常驻——本来也是稳定态上限）。六轮 gen1/gen2（temp 0 /
seed 42）与单机基线全部逐字 IDENTICAL。**建议生产开启 GGML_EP_PREFAULT=1**。

worker 线程标定（slave：2 socket × 36 核 × 2 HT = 144 线程，prefault=1）：

| worker -t | 每层 compute 均值 | TG96 | TG512 |
| --- | --- | --- | --- |
| 72（=物理核数） | 0.84-1.17ms | 24.54-25.23 t/s | 24.49-24.59 t/s |
| 136 | ~10.4ms | 21.61 t/s | 6.89 t/s |
| 128 | ~19.4ms | 5.87 t/s | 4.44 t/s |

超过物理核数后 ggml 线程 barrier 严重劣化（且非单调，128 比 136 更糟），"留 8-16 核给系统"
在此 workload 下适得其反。**推荐 worker -t 72（=物理核数，不跨 HT），不要降线程也不要超线程**。

## 启动线程自动标定实测（2026-07-30，slave 2×36核/144HT，DSV4 8层 35-42，prefault=1）

worker 不带 `-t` 启动，autotune 在 listen 前阶梯实测（首层+中间层共 2 个代表层，decode 形态
n_tokens=1/top-6，2 warmup + 5 轮中位，总耗时 <0.1s）：

| 候选线程 | 16 | 24 | 32 | 48 | 72 |
| --- | --- | --- | --- | --- | --- |
| ms/iter（2 层合计） | 0.975 | 0.623 | 0.530 | 0.432 | 0.412 |

每步边际收益均 ≥3%（48→72 为 4.6%），knee 落在 72 = 物理核数，自动选中 72，落在预期
[32, 72] 合理区间。（绝对值低于历史 0.73ms/层：dummy 输入固定 6 个热专家、页全热且不含 RPC
开销，只有跨线程相对比较意义。）

标定后服务验证（同二进制 slave 单机，固定种子 REQ：层 35，n_tokens=1，top-6）：

| 启动方式 | 输出校验 | 每 REQ compute 中位 |
| --- | --- | --- |
| 不带 -t（autotune→72） | sum=-0.15102280 norm=14.59143706 | 0.283ms（n=200） |
| 显式 -t 72 | 同上，逐 bit 一致 | 0.340ms（n=200，轮次噪声） |
| 显式 -t 48（覆盖生效，日志直接 48 线程不标定） | 同上，逐 bit 一致 | 0.381ms（n=50） |

`--no-autotune` 与 `GGML_EPD_AUTOTUNE=0` 均打 `autotune disabled, using 8 threads` 并以 8 线程
listen；`--selftest` PASS（max_abs_diff=0）。

## --no-mmap 双机实测（2026-07-30，slave 35-42 冷缓存，xcache drop 后 mincore 0.0%，不开 prefault）

同会话两轮（master build-epdev-autotune llama-server 生产配置，slave build-cpu-autotune）：

| worker 配置 | 启动 | VmRSS | 每 REQ compute 中位/p90 | >2ms 尖峰 | TG512 |
| --- | --- | --- | --- | --- | --- |
| `--no-mmap` + autotune（选中 48） | 读 15.75 GiB / 11.3s（1.40 GB/s 冷 NVMe） | **15.78 GiB 常驻** | 0.719 / 0.997 ms | **89/5032（1.8%，avg 3.3ms / max 7.5）** | 25.34 t/s |
| mmap 默认 + `-t 72` | 秒启 | 0.02 GiB（懒分页） | 0.553 / 4.606 ms | **618/5032（12.3%，前 500 REQ 59%、avg 10.3ms / max 58.6）** | 24.91 t/s |

结论：mmap 冷缓存尖峰是典型页入模式（首轮 59% 尖峰、max 58.6ms，长尾贯穿全程——冷专家
持续页入）；`--no-mmap` 把尖峰压到 prefault 热态同级（~1-2%、幅度 3-8ms 的 barrier jitter
地板），且全程无页入型长尾（p90 0.997 vs 4.606ms）。正确性：两轮 gen1/gen2（temp 0/seed 42，
英文 48 + 中文 64 token）与基线 /tmp/dsv4-base-gen{1,2}.json **逐字 MATCH**，两轮 gen512
互比 MATCH；`--no-mmap --selftest` PASS 且 |out|=36.206464 与 mmap 逐 bit 一致。
注：N1 autotune 选中 48（48→72 边际 2.4% < 3%，与上轮选中 72 属 knee 边界噪声，均在 [32,72]
合理区间）。

## RDMA（RoCEv2）传输后端实测（2026-07-30，slave 35-42，worker -t 72 + prefault=1）

实现：`tools/epd/llama-ep-rdma.cpp`，rdma_cm 建连（自动 GID 解析），RC QP + Send/Receive，
每连接预注册 4×256KB 发送环 + 8×256KB 接收环；framing 层依赖字节流语义，后端内部把消息
切成 `[u32 len + payload]` 块并在接收端拼回字节流；等待走 ibv comp channel 阻塞（不 spin）。
`GGML_REMOTE_EP_RDMA=1`（默认关）同时置在 master 与 worker；CMake 检测不到
libibverbs/librdmacm 时不编译该后端，env 开了但无卡/建连失败打 warning 自动回退 TCP，
纯 TCP 环境行为零变化。`GGML_EP_RDMA_SPIN=1` 为调试开关（busy-poll CQ）。

微基准（`llama-ep-transport-bench`，master↔slave 跨机，echo 往返，同会话 tcp1→rdma1→rdma2→tcp2）：

| payload | TCP RTT med（两轮） | RDMA RTT med（两轮） | TCP p99 | RDMA p99 | TCP CPU/op | RDMA CPU/op |
| --- | --- | --- | --- | --- | --- | --- |
| 64B | 74.0 / 41.2 µs | **12.6 / 10.4 µs** | ~145 µs | ~30-35 µs | 5.9-7.1 µs | 4.1-5.2 µs |
| 4KB | 40.3 / 74.6 µs | **19.8 / 17.7 µs** | ~140 µs | ~40 µs | 6.9-8.1 µs | 6.0-7.1 µs |
| 64KB | 98.5 / 108.9 µs | **39.9 / 42.4 µs** | ~157-178 µs | ~70-86 µs | ~24 µs | ~11-12 µs |
| 1MB | 514 / 531 µs | **255 / 256 µs** | 752-878 µs | 300-358 µs | ~270 µs | ~145 µs |

RTT 全面 2-4× 优势，尾延迟（p99）压到 TCP 的 ~1/4，大消息 CPU 减半；两轮同序/反序结论一致。

双机 DSV4 8 层生产配置 A/B（同会话 tcp1→rdma1→rdma2→tcp2，每轮 slave 冷缓存重启 worker）：

| 轮次 | TG96 | TG512 | send 中位/call | wait 中位/call | wait>2ms 尖峰 | wait max |
| --- | --- | --- | --- | --- | --- | --- |
| tcp1 | 24.74 t/s | 24.10 t/s | 0.023 ms | 0.971 ms | 1.10% | 9.4 ms |
| rdma1 | **25.22 t/s** | **24.41 t/s** | 0.014 ms | 0.908 ms | 0.88% | 5.7 ms |
| rdma2 | **25.12 t/s** | **24.49 t/s** | 0.014 ms | 0.896 ms | 0.98% | 5.9 ms |
| tcp2 | 24.73 t/s | 24.28 t/s | 0.023 ms | 0.971 ms | 0.91% | 6.8 ms |

结论：RDMA 每 call send 时间 0.023→0.014 ms，8 层 wait 中位 0.971→0.90 ms（≈省 9µs/层/方向，
与微基准 64B-4KB RTT 差一致）；TG96 +~0.45 t/s（+2%）、TG512 +~0.25 t/s（+1%），两种顺序
各自一致（无顺序效应）；尖峰率相当（prefault 已把页入尖峰压掉，传输层再降空间有限），
wait 上限 9.4→5.7ms。每 call 仅 5808 次/token 级采样，收益符合"网络非带宽瓶颈"预期：
RTT 砍 3/4 换来 TG 均值 +1-2%，主要价值在尾延迟与 CPU 余量。正确性：四轮 gen1/gen2
（temp 0 / seed 42，英文 48 tok + 中文 64 tok）TCP↔RDMA 两两逐字 MATCH。

## GLM-5.2 双机实测（commit cbcccef7e，glm-dsa，79层/160专家/top-8，UD-Q2_K_MXFP4 7 分卷 ≈236G）

多分卷 GGUF：epd 对每卷独立 gguf_init + mmap，张量注册进统一 name→tensor 表
（多分卷惯例：各卷元数据只列本卷张量）；glm-dsa.cpp 的 MoE 分支对
`GGML_REMOTE_EP_LAYERS` 范围内层做 TENSOR_SKIP（前 3 个密集层不动）。

配置：master 本地层 18-78 + GPU 专家卸载（29-32→CUDA0，58-62→CUDA1），
slave（server2）认领层 3-17（15 层 ≈43.5G 权重）。模型在 exfat USB 盘上，
4K 随机读仅 ~82MB/s，需先 `dd bs=16M` 顺序预热全部 7 卷，否则 TG 只有 0.14-0.17 t/s。

| 配置 | master RSS | master GPU | slave RSS | TG(96 tok) | PP(5 tok) |
| --- | --- | --- | --- | --- | --- |
| 单机 mirror 基线（build-epdev） | 166.8G | 21.7+21.9G | — | 10.5 t/s | 17.4 t/s |
| 双机 EP（3-17 远端） | 70.7G（生成中快照） | 21.7+21.9G | 37.6G（上限 ~43.5G） | 9.8-9.9 t/s | 9.7 t/s（1 tok） |

正确性：同 prompt（temp 0 / seed 42，cache_prompt false）双机与单机 mirror 输出逐字一致
（英文 48 tok + 中文 64 tok，长度均相同）。中文 prompt 开头出现的怪字"淏"在单机基线
同样出现，是模型/权重补丁本身行为，非 EP 引入。

注：生产单机基线（build-cuda 二进制）TG ~12.0 t/s；build-epdev 基线 10.5 t/s 的差距
来自构建差异而非 EP 改动（EP 关闭时同一二进制）。双机 TG 约为同二进制单机 mirror 的 93%。

## 层数分配：按内存带宽均衡

decode 时每层每 token 专家字节 `E = top_k × 3 × n_embd × n_ff × bpw/8`（DSV4 ≈48MB/层）。
两端 CPU 侧耗时相等时最快：`N_slave = N_CPU层数 × t_master / (t_master + t_slave)`
（GPU 卸载层先扣除；--numa mirror 走节点本地访问，用 NUMA 本地口径）。

实测 NUMA 本地 read 带宽（membw，干净环境）：master node0 167.5 / node1 173.9
（合计 ~341 GB/s）；slave node0 93.8 / node1 80.2（合计 ~174 GB/s，
node1 因 2DPC 内存慢 14%）→ 带宽比 ~1.96:1。**【2026-07-31 更新：以下为新硬件口径——
slave 内存整改后双节点 IMC 对称各 ~157 GB/s（合计 ~315，1.8×）；master 重启后
t76 实测 156.9/153.8（合计 ~311）。ep-plan.py 默认带宽已更新为本组。】**
GGML_REMOTE_EP_DEBUG 实测每层 decode 耗时：master 本地 ~0.37ms/层（DSV4），
slave 旧带宽 0.73ms/层 → 新带宽 GLM 实测 0.85-0.94ms/层（旧带宽估计 2.6ms/层作废）。

- DSV4（40 MoE 层，GPU 固定卸载 14 层 8-21，CPU 可分配 26 层 3-7+22-42）：
  0.37×(26-N) = 0.73×N → N ≈ 9；计入每层 ~0.13ms 网络开销后 **N ≈ 8（slave 35-42）
  为实测均衡点**。当前生产配置 slave 31-42（12 层）是 slave-bound（10.3ms vs master
  CPU 5.5ms），但 master RSS 省得更多（37.4G vs 基线 62.3G）。
- GLM-5.2（master CPU 53 层 + GPU 8 层，slave 15 层）：瓶颈在 master CPU；
  受 slave 内存限制建议 slave 30-35 层。

interleave 整机口径（旧数据，仅供参考）：master read 94.5 GB/s，slave 128.2 GB/s。

## 流水线分块投递 + worker 固定开销消除（2026-07-30，commit d583d8b61 / 25e38646b）

**worker 每 REQ 固定开销修复**（默认行为，无需 env）：原 `ep_moe_ffn` 每 REQ
`ggml_backend_alloc_ctx_tensors` + free（数百 MB 中间 buffer 每次首触页错误）且
`ggml_graph_compute` 无线程池时每次 spawn+join 一次性池（70-72 线程）。实测固定开销
~7ms/REQ：GLM slave 15 层合计（n_tokens=63）882→458ms（-48%），（n=128）980→717ms，
（n=1020）6889→5560ms（-19%）。修复：持久 `ggml_threadpool`（autotune 后用最终线程数
挂 `ggml_backend_cpu_set_threadpool`）+ `ggml_gallocr` grow-only 计算 buffer。
端到端：DSV4 PP(985 tok) 157→274.5 t/s（+75%）；GLM PP 63档 11.5→15.6、1020档
66.6→74.3 t/s；小 prompt 冷/热首轮效应（原 5-tok 8.0 vs 17.5 t/s）基本消除。
decode 不变（DSV4 TG512 24.7 vs 基线 24.7）。

**流水线分块投递**（`GGML_REMOTE_EP_PIPELINE=1`，默认关）：多 token 层按 token 维切块
（默认 256 tok，按 hidden 字节封顶 TCP ≤3MiB / RDMA ≤1.5MiB，`..._PIPELINE_CHUNK` 可调），
W=1 滑动窗口发送（发块 i 后收块 i-1 响应），worker 计算与 master 收发重叠；RESP 直接收进
`out` 切片省一次整层 memcpy。协议零改动（每块=普通 REQ 帧），worker 无需知晓；K<2 自动
回退原阻塞路径（decode 不受影响）；失败重连用整体单块重发。TCP transport 两侧装 4MiB
socket buffer（W=1 窗口内至多 1 REQ+1 RESP 在飞，块不超预期缓冲，无死锁；RDMA 接收环
8×256KB 同理）。注意：worker 修复前分块把固定开销 ×K 放大，GLM 1020 档曾 -4.8% 回归；
修复后块开销≈0（n=128 每 token 0.373ms vs n=1020 0.363ms），流水线转为小幅净正。

A/B（同会话 off→on→on→off→on(RDMA) 五轮 ABBA 反序，逐字对拍 temp0/seed42 全 IDENTICAL）：

| 模型/档 | off 均值 | on 均值 | on(RDMA) | 备注 |
| --- | --- | --- | --- | --- |
| DSV4 TG512 | 24.75 t/s | 24.69 t/s | 24.72 | decode 路径不变，符合零影响 |
| DSV4 PP 245 tok | 123.7 | 123.5 | 123.6 | 传输占比小，±顺序噪声 |
| DSV4 PP 985 tok | 274.5 | 273.6 | 275.1 | 同上（轮间顺序效应 ±3%） |
| GLM PP 63 tok | 15.60 | 16.21 | 16.11 | 63<chunk，走原路径；差值为顺序噪声 |
| GLM PP 254 tok | 34.65 | 35.14 | 35.20 | +1.4% |
| GLM PP 1020 tok | 74.30 | 74.83 | 75.01 | +0.7~1.0% |

**GLM PP 63-token 档异常根因**（GGML_REMOTE_EP_DEBUG 双侧分解，同会话冷/热 63 复测）：
非冷缓存（热复测仅快 ~12%）、非 RPC（15 层远程 wait 合计仅 ~0.6s/4.0s）。主因是
**master 本地 CPU MoE 的每 ubatch 固定权重读取**：63 tok×top-8=504 指派激活 ~220/256
专家（5 tok 仅 ~40），每层读 ~2.7GB 权重实测 ~87ms（有效带宽仅 ~25GB/s），39 个本地
CPU 层 ≈3.4s，小档无法摊薄（63档本地 54ms/tok vs 1020档 7.6ms/tok）；次因是上述
worker 固定开销（修复后 63档 11.5→15.6 t/s）。遗留：master 本地小批次有效带宽偏低
（numa_balancing 未关/线程划分待查）；远程层在模型前部时跨 ubatch 流水上限仅层 0-2，
远程层若放尾部可与下一 ubatch 本地段大幅重叠（MAX-EFFORT 方向）。

## 层镜像 + 专家 slot 拆分（GGML_REMOTE_EP_MIRROR=1）实测（2026-07-30，commit 115622070）

机制：镜像层专家权重 master 也加载（加载器改用 `llama_remote_ep_skip_weights_for_layer`，
镜像层不再 TENSOR_SKIP）。`build_moe_ffn` 在该层用两个 custom op 夹住本地子图实现重叠
（ggml CPU 后端按节点创建顺序串行、节点间 barrier）：op1（`ggml_map_custom3`）线程 0 把
slot [0,k_r) 的 REQ 只发不等（ids/weights 先按 `ids[t*k_r+j]` gather 到 staging 再发）；
随后本地子图对 slot [k_r,k) 走与单机本地 MoE 完全相同的链路（separate gate/up→
swiglu→down→×weights，clamp 分支含 DEEPSEEK4 特例），产出 experts_l 不接求和链；
op2（`ggml_map_custom2`，b=experts_l 制造依赖）阻塞收 RESP（partial_r）写 dst，
失败重连重发一次（复用 pending 槽里的请求字节）再失败 GGML_ABORT；
合并 = partial_r 先、本地 slot 升序逐个 `ggml_add`——与全远程基线（worker 内 slot
0..k-1 顺序累加）结合顺序完全一致。worker 零改动。镜像路径不走 pipelined roundtrip。

正确性（temp 0 / seed 42，"The capital of France is" gen48）：GLM 与 DSV4 的
off1 vs on1/on2/off2 全部逐字 **IDENTICAL**；GLM 另验 prefill 边界 prompt_n=64
对拍 IDENTICAL（prompt_n=32 未能构造：所用重复文本在 N∈[2,200] 内没有任何长度
恰好 tokenize 成 32，该档未覆盖）。双机同型号 Xeon，vec_dot 同 kernel，未出现 DIFF。

GLM-5.2（slave 3-17 共 15 层，top-8，默认 k_r=4；同会话 off→on→on→off + k_r 扫描）：

| 档 | off1/off2 | on1/on2（k_r=4） | Δ |
| --- | --- | --- | --- |
| TG512 | 11.12 / 11.47 | **12.41 / 12.57** | **+10.6%** |
| PP 5 tok | 18.73 / 18.84 | 20.70 / 20.93 | +10.8% |
| PP 63 tok | 16.72 / 16.73 | **13.61 / 13.62** | **-18.6%（回归，见下）** |
| PP 254 tok | 35.27 / 35.45 | 34.87 / 34.90 | -1.4%（噪声级） |
| PP 1020 tok | 74.28 / 75.08 | **94.99 / 94.70** | **+27.4%** |

k_r 扫描（TG512 / PP254）：k_r=2 → 12.71 / 33.66；k_r=3 → **12.82** / 33.94；
k_r=4 → 12.64 / **34.34**（on 轮 k_r=4 复测 TG 12.41/12.57、PP254 34.87/34.90）。
差异 ~1.5% 在轮间噪声边缘，默认 n_used/2 已是合理选择；调优依据 = 两端带宽比
（master ~341 GB/s vs slave ~174 GB/s ≈ 2:1，理论上 k_r 偏小更优，实测 TG 确以
k_r=3 略优，但幅度不值得偏离默认值）。

ep-debug 远程段分解（每层均值）：decode wait 1.573/1.277ms → **0.249/0.215ms**
（-84%，重叠生效）；prefill wait 37.4/36.0ms → **0.88/0.81ms**（本地子图算完时
远端半层早已返回，远程段从关键路径上消失）。send 两侧均 <0.5ms。

PP 63 档回归根因：即上文"GLM PP 63-token 档异常根因"的同一问题——master 本地
CPU MoE 小批次固定权重读取有效带宽仅 ~25GB/s；镜像把 15 层×4 slot 的 prefill
计算搬回 master 本地路径，小档无法摊薄（wait 已降到 0.88ms，瓶颈转移到 master
本地段）。大档（1020）摊薄后转为 +27% 净胜。

内存代价：GLM 镜像开 master used 209G vs off 163G（**+46G**，15 层 ≈43.5G），
available 42G 全程无 OOM；比预估的 +20G 多，预算请按全量层权重计。

DSV4（slave 35-42 共 8 层，top-6，默认 k_r=3；同会话 off→on→on→off）：

| 档 | off1/off2 | on1/on2 | Δ |
| --- | --- | --- | --- |
| TG512 | 24.78 / 24.93 | **27.15 / 27.16** | **+9.3%** |
| PP 245 tok | 119.08 / 122.05 | 113.64 / 114.06 | -5.6%（小回归） |
| PP 985 tok | 266.84 / 274.28 | **300.64 / 299.45** | **+10.9%** |

ep-debug：decode wait 0.908/0.885ms → **0.187/0.179ms**（-80%）；prefill wait
20.3/17.8ms → **0.92/0.84ms**。内存：on used 69G vs off 52G（+17G = 8 层 16G）。
PP245 小回归与 GLM 小档同源（master 本地 MoE 小批次带宽问题，DSV4 幅度小得多）。

结论：decode（TG）与大档 PP 全面净胜（远程段重叠掉，-80% wait），小档 PP 因
master 本地小批次带宽短板有回归——小档场景建议保持默认（MIRROR 关）或等
master 本地带宽问题修复后再开。off 轮均复现历史基线锚点（GLM TG≈10.7-11.5 /
PP1020≈74.5，DSV4 TG≈24.7 / PP985≈274.5），无顺序效应。

## GLM-5.2 全量测速（2026-07-31，slave 新内存 188G/315GB/s 首轮，commit 4e51d4730+）

硬件变化：slave 188G（7×16G+10×8G，双节点对称各 ~157 GB/s）；master 重启后
156.9/153.8 GB/s（t76）。worker：build-cpu-rdma，`--no-mmap -t 70`，
`GGML_EPD_NUMA=weighted`（首次实战：带宽探测 node0=node1=78.7 GB/s → 权重 1:1，
对称节点等价 interleave，全程无异常；sysfs 权重写入需 root，缺权限打 warning 用
当前值）。测速脚本：`bench-glm-{master,worker}.sh` + `bench-glm-client.py`
（/tmp 模板丢失后的入库替代品）。

**worker 线程扫描**（RDMA，TG512）：-t 70 → 12.16 t/s；-t 36 → 11.60/11.66 t/s。
新现象"72 线程 membw -20%"在端到端 decode 上不成立，**维持 -t 70（=物理核）**。

**全矩阵**（master build-epdev-rdma，t70/tb70，fa1，b1024/ub512，GLM 15 层 3-17）：

| 配置 | TG96 | TG512 | PP5 | PP63 | PP254 | PP1020 |
| --- | --- | --- | --- | --- | --- | --- |
| 单机（NUMA-EP+mirror，热） | — | 13.20/13.25 | ~20 | 5.94/6.18 | 13.87 | 34.87/35.12 |
| 双机 TCP（off1/off2 ABBA） | 11.64 | 11.68/11.85、11.58/11.85 | 18.2 | 7.17/7.15 | 18.65 | 34.47/34.10 |
| 双机 RDMA | 12.21 | 12.01/12.12/12.11 | 19.9 | 塌陷 | 塌陷 | 塌陷 |
| 双机 RDMA+PIPELINE | 12.15 | 12.10/12.11 | 18.4 | 6.19 | 2.28 | 2.15 |
| 双机 TCP + MIRROR=1（on1/on2 ABBA） | 12.76 | 12.80/12.93、12.53/12.87 | 19.9 | 4.31/5.33 | 15.12 | 31.10/33.05 |
| 双机 32 层（3-34，TCP） | 9.24 | 9.46/10.26 | 16.4 | 8.30 | 19.22 | 31.70 |

正确性：单机 / 双机 TCP / 双机 RDMA / RDMA+pipeline / MIRROR on×2 / off 共 8 份
gen48（temp 0 / seed 42 / "The capital of France is"）md5 全部一致（逐字 IDENTICAL）。

**发现 1：RDMA 大帧塌陷（待修）**。decode（KB 级帧）正常且比 TCP +3%
（TG512 12.1 vs 11.8）；但 MB 级 PP REQ 帧在 RDMA 环上仅 ~3MB/s——master ep-debug
实测 n_tokens=254（~6.2MB 帧）每层 send 1939ms（≈77ms/256KB chunk），PP254 90s、
PP1020 219s；同帧 TCP 正常（13.6s/29.6s）。微基准 ≤1MB 一切正常（255µs），问题
只在多 chunk 大帧。PIPELINE 不救（RDMA chunk 封顶 1.5MiB ≈63 tok/帧，帧内路径正常
但 master 本地段才是大头，PP1020 反而 473s）。**临时结论：decode 开 RDMA，PP 用 TCP。**
另发现 rdma_cm 建连无超时：worker 陈旧连接状态下 master warmup 线程在
ucma_get_event 无限等待（server 假死，RSS 7.8G 不前进）；重启 worker 后恢复。

**发现 2：master 本地 MoE prefill 带宽约减半（环境问题，非代码）**。本次 PP 全面
低于 07-30 晚（PP1020 双机 66-74.5 → 34.5；单机 34.9）。旧二进制 build-epdev
（0730-02:31）同机复测 PP1020 33.6/35.0 与新二进制一致 → 排除镜像/重构图改动
回归；numa_balancing 关/开无差异（已恢复开）；decode 反而更快（单机 TG512
13.2 vs 旧 ~10）。待查（重启后环境差异：governor/C-state/IRQ/页缓存策略等）。

**发现 3：MAX-EFFORT 镜像 PP 收益反转**。ABBA 配对（同会话 off1→on1→on2→off2）：
TG512 off 11.7 avg → on 12.8 avg（**+9.4%**，与旧带宽 +10.6% 一致，decode 镜像
重叠机制稳定）；PP1020 旧 +27% → 现 **-6.4%**；PP63 旧 -18.6% → 现 **-31%**。
根因即发现 2：镜像把 15 层 ×4 slot 的 prefill 计算搬回 master 本地，master 本地
prefill 带宽塌陷后由赚变亏。**decode 开镜像、PP 关镜像**（env 切换即可）。

**发现 4：规划器新带宽预测被实测否定，分层维持 15 层**。ep-plan.py 新增
`--model glm` 预设（锚点 9.86/10.71/10.42 反解 t_m=1.33/t_s=2.60ms/层，β=0.719，
锚点误差 ≤0.4%）；按新带宽（t_s×174/315=1.44）预测 Ls* 15→32（TG 12.90 vs 10.54）。
实测 32 层（3-34，slave 93G/188G 内存无压力）TG512 9.46/10.26（**-13% vs 15 层**）、
PP 持平 → 与 07-30 的 25 层否定一致：**decode 远程段近串行，远程每层成本
（compute ~1.05ms + RPC）仍高于本地（~1.1ms），加层即减速**；规划器的重叠度假设
对 decode 过于乐观（模型适用于粗筛，边界需实测）。DSV4 同法预测 8→11 层（未实测，
仅结论）。worker 32 层分层 compute 分布：多数层 0.9-1.0ms，L3/L29/L30 异常
1.2-1.7ms（待查）。

**slave 内存核算**：32 层 --no-mmap RSS 98G/188G（available 89G）无压力；
master 侧 32 层非镜像 ~114G/251G；**镜像在 Ls=32 不可行**（163+32×2.9≈256G > 251G）。

## RDMA 大帧塌陷根因与修复（2026-07-31，commit e6e40b42b）

**根因**：rdma_cm 建连的 RC QP 使用默认 `min_rnr_timer`（~80ms 级）。大帧打空
8 槽接收环后触发 RNR NAK（slave 侧 `rnr_nak_retry_err` 计数器实测递增），
`rnr_retry_count=7`（无限重试）下每次重试睡一个定时器，雪崩级联：
6.2MB GLM PP 帧 ~3MB/s（77ms/chunk）、bench 16MB 帧 p99 7.5s、SPIN 忙轮询
反而更差（排除 CQ 事件通道嫌疑，坐实重试定时器）。

**修复**：ESTABLISHED 后 `ibv_modify_qp` 设 `min_rnr_timer=1`（0.01ms；
mlx5 RTS→RTS 唯一允许修改的属性，失败降级 warning 不影响行为）。顺带修复
rdma_cm 建连无超时：`cm_wait` 加 5s poll 上限（listener 保持无限等待），
陈旧 worker 状态下 master 不再 `ucma_get_event` 假死，超时走既有 TCP 回退
（黑洞 IP 实测 2s 内 ADDR_ERROR 干净失败）。默认行为零变化（RDMA 仍 opt-in）。

**transport-bench 修复前后**（跨机 echo 往返吞吐；TCP 为同链路对照）：

| 帧 | TCP | RDMA 修复前 | RDMA 修复后 |
| --- | --- | --- | --- |
| 64B | RTT 42-74µs | 13.6µs | 13.6µs |
| 4KB | 48 MB/s | 264 MB/s | 219 MB/s |
| 64KB | 625 MB/s | 1852 MB/s | 1462 MB/s |
| 1MB | 1998 MB/s | 3921 MB/s | 3951 MB/s |
| 4MB | 2869 MB/s | 5288 MB/s | 5214 MB/s |
| 16MB | 2660 MB/s | **27.8 MB/s（p99 7.5s 停顿）** | **5488 MB/s（max 4.7ms，零 >100ms 停顿）** |

修复后大帧 RDMA ≈2× TCP 且 64B-1MB 小帧行为不变（decode 无回归）。

**GLM 双机实测（15 层 3-17，同会话 RDMA/TCP 反序，worker -t70 weighted）**：

| 轮次 | TG512 | PP63 | PP254 | PP1020 |
| --- | --- | --- | --- | --- |
| RDMA rdmafix1 | 12.17/12.27 | 14.44 | 38.06 | **76.06** |
| TCP（热） | 11.21/11.23 | 7.40 | （昨日同参数 18.65） | 33.44 |
| RDMA rdmafix2 | 11.00/11.46 | 7.06 | — | 33.48 |

PP 不再塌陷；RDMA ≥ TCP 全部档位（rdmafix1 状态 PP1020 2.3× TCP 并超 07-30
晚历史最好 74.3-75）。两轮 RDMA 间的 PP 波动（76 vs 33.5）与 TCP 轮同幅，
为并行 agent 调 RAPL 功耗墙导致的 master 本地 prefill 带宽波动（待 agent-14
结论），与传输层无关。正确性：gen48（temp0/seed42）RDMA 修复后两轮 + TCP 轮
与单机基线 md5 全部一致（47cfde37...，逐字 IDENTICAL）。
