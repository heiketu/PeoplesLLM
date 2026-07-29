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
| `GGML_REMOTE_EP_RDMA` | 关 | 置 1 传输层改用 RDMA（RoCEv2，rdma_cm 自动建连）。需编译时检测到 libibverbs+librdmacm（CMake 打印 `EP RDMA transport (RoCEv2): ON`）；无卡/建连失败时打 warning 并自动回退 TCP。master 与 worker 需同时置 1（协议不同，不互通） |

worker 侧另有：

| env | 默认 | 说明 |
| --- | --- | --- |
| `GGML_EP_PREFAULT` | 关 | 置 1 启动时预触认领层专家权重的全部 mmap 页（多线程），消除冷专家首次命中时页入造成的多 ms compute 尖峰 |
| `GGML_EP_PREFAULT_THREADS` | 16 | 预触线程数 |
| `GGML_EPD_AUTOTUNE` | 开 | 置 0 关闭启动线程自动标定（同 `--no-autotune`） |

## 启动线程自动标定（autotune）

**仅当未显式传 `-t` 时生效**（显式 `-t` 行为完全不变）。模型映射 + 层认领 + prefault 之后、
listen 之前，worker 在认领层中取首层 + 中间层共 2 个代表层，以 decode 形态（n_tokens=1、
top-k 从 `<arch>.expert_used_count` 读）构造 dummy 输入，对候选线程阶梯
`{16, 24, 32, 48, 物理核数}`（物理核数按 sysfs topology 去重 (package, core)，不计 HT）
各跑 2 warmup + 5 轮取中位，选 knee point：**边际收益 < 3% 的最小线程数**（专家 FFN 是带宽
敏感型，过饱和点后加线程只增 barrier 开销）。标定切换线程走与服务完全相同的
`ggml_backend_cpu_set_n_threads` 路径，总耗时 <1s。每个候选的 ms/iter 与最终选择均打日志。

实测（slave 2×36 核/144 HT，DSV4 8 层 35-42）：见下文标定实测节。

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
node1 因 2DPC 内存慢 14%，硬件现状）→ 带宽比 ~1.96:1。GGML_REMOTE_EP_DEBUG 实测
每层 decode 耗时：master 本地 ~0.37ms/层，slave ~0.73ms/层（旧 0.31/0.376 估计作废）。

- DSV4（40 MoE 层，GPU 固定卸载 14 层 8-21，CPU 可分配 26 层 3-7+22-42）：
  0.37×(26-N) = 0.73×N → N ≈ 9；计入每层 ~0.13ms 网络开销后 **N ≈ 8（slave 35-42）
  为实测均衡点**。当前生产配置 slave 31-42（12 层）是 slave-bound（10.3ms vs master
  CPU 5.5ms），但 master RSS 省得更多（37.4G vs 基线 62.3G）。
- GLM-5.2（master CPU 53 层 + GPU 8 层，slave 15 层）：瓶颈在 master CPU；
  受 slave 内存限制建议 slave 30-35 层。

interleave 整机口径（旧数据，仅供参考）：master read 94.5 GB/s，slave 128.2 GB/s。
