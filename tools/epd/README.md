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

仅当层为 SILU+gate MoE、无 expert bias/scale、非 warmup 图时走远端；
不满足时自动回退本地路径。连接懒建立、跨 decode 复用，传输错误自动重连重试一次。

## worker

```
llama-epd -m model.gguf --port 29200 --layers 3-42 [--experts 0-255] [-t 72]
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

已知限制：单一 worker 端点；阻塞式收发（无流水线）。

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
两端 CPU 侧耗时相等时最快：`N_slave = N_CPU层数 × B_slave / (B_master + B_slave)`
（GPU 卸载层先扣除）。

实测带宽（membw，read，整机 interleave）：master 94.5 GB/s，slave 128.2 GB/s
（node0 本地 93.2 / node1 本地 80.6，不对称）→ slave 应承担约 57.6% 的 CPU 层。

- DSV4（master GPU 已卸 14 层）：当前 slave 21 层 ≈7.9ms/token、master CPU 5 层 ≈2.5ms，
  slave 是瓶颈；平衡点 slave ~15 层（28-42）+ master CPU 11 层 + GPU 14 层 ≈5.6ms。
- GLM-5.2（master CPU 53 层 + GPU 8 层，slave 15 层）：瓶颈在 master CPU（耗时比 ≈1:4.8），
  理论平衡 slave ~38 层；受 slave 125G 内存限制（≈2.5G/层）建议 30-35 层。
