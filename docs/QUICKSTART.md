# x-llama.cpp 上手指南（Quickstart）

> 面向想在类似平台上复现本项目性能的用户：双路 Xeon/EPYC（2 NUMA 节点）+ 2 × GPU，
> 可选第二台双路机器。所有参数语义见 [PARAMETERS.md](PARAMETERS.md)，逐项实测数据见
> [PEOPLESLLM-PARAMS.md](PEOPLESLLM-PARAMS.md) 与 [tools/epd/README.md](../tools/epd/README.md)。
>
> 目录：1. 构建 → 2. 配置一（单机混合推理）→ 3. 配置二（双机 EP）→ 4. 纯 CPU 推理 → 5. 常见问题。

---

## 1. 构建

依赖（Ubuntu 24.04 为例）：

```bash
sudo apt install -y git cmake ninja-build build-essential pkg-config \
    numactl libnuma-dev rdma-core libibverbs-dev librdmacm-dev jq
```

### 1.1 CUDA 构建（master / 单机混合推理）

```bash
cmake -S . -B build-cuda -G Ninja -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build-cuda -j --target llama-server llama-bench llama-cli
```

本仓库参考构建 `build-cuda/CMakeCache.txt` 的实际配置为 `GGML_CUDA=ON`、`GGML_NATIVE=ON`
、`CMAKE_BUILD_TYPE=Release`，并已检测到 RDMA 依赖（`LLAMA_EP_IBVERBS_LIBRARY` 等）。
配置阶段应看到 `EP RDMA transport (RoCEv2): ON`；没有它，运行时设 `GGML_REMOTE_EP_RDMA=1`
也只能回退 TCP。构建前用 `nvidia-smi` / `nvcc --version` 验证 CUDA 环境。

### 1.2 纯 CPU 构建（EP worker / 纯 CPU 推理）

```bash
cmake -S . -B build-noncuda -G Ninja -DGGML_CUDA=OFF -DGGML_NATIVE=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build-noncuda -j --target llama-epd llama-server llama-bench
```

`GGML_NATIVE=ON` 按本机 CPU 指令集编译（Ice Lake/EPYC 上得到 AVX-512 + VNNI 内核）。
跨机 EP 的 worker 用这个构建；把 `build-noncuda/bin/llama-epd` 及同目录动态库同步到 slave，
两台机器必须使用**同一版本**的 `llama-epd` 与协议代码。

### 1.3 AVX2-only 机器（无 AVX-512，如老 Xeon / 消费级 CPU）

```bash
cmake -S . -B build-avx2 -G Ninja \
    -DGGML_NATIVE=OFF -DGGML_AVX512=OFF -DGGML_VNNI=OFF \
    -DGGML_CUDA=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build build-avx2 -j --target llama-epd llama-server
```

本仓库参考构建 `build-avx2/CMakeCache.txt` 即此口径（`GGML_AVX2=ON`、`GGML_NATIVE=OFF`、
`GGML_AVX512=OFF`、`GGML_VNNI=OFF`）。注意 `GGML_REPACK_GEMV_PREFETCH` 等 AVX-512 GEMV
内核旋钮在 AVX2 构建上无效；slave 若是 AVX2-only 机器，预期带宽与 EP 收益都要打折。

### 1.4 AMX 机器（Sapphire Rapids / Emerald Rapids / Granite Rapids）

```bash
cmake -S . -B build-amx \
    -DGGML_NATIVE=OFF -DGGML_CUDA=OFF \
    -DGGML_AVX512=ON -DGGML_AVX512_VNNI=ON -DGGML_AVX512_VBMI=ON -DGGML_AVX512_BF16=ON \
    -DGGML_AMX_TILE=ON -DGGML_AMX_INT8=ON -DGGML_AMX_BF16=ON \
    -DCMAKE_CXX_FLAGS="-mavx512vpopcntdq -mgfni" -DCMAKE_C_FLAGS="-mavx512vpopcntdq -mgfni"
cmake --build build-amx -j
```

AMX buft 只在运行时 `arch_prctl` 授权成功（真 AMX 硬件或 SDE 模拟）时注册，非 AMX 机器自动
回退 CPU_REPACK，无需额外开关。权重布局与 gemv/gemm 分流细节见 [PARAMETERS.md](PARAMETERS.md) §8。
无 AMX 硬件时用 Intel SDE 验证正确性：`sde64 -spr -- <prog>`。

---

## 2. 推荐配置一：单机混合推理（2 × GPU + 双路 CPU）

当前生产实测最优（DeepSeek-V4-Flash mxfp4，attention/router 在双 GPU、全部 MoE 专家在 CPU）：

```bash
env \
  GGML_NUMA_EP=1 \
  GGML_NUMA_HIER_BARRIER=1 \
  GGML_REPACK_GEMV_PREFETCH=1 \
  build-cuda/bin/llama-server \
    -m /path/to/DeepSeek-V4-Flash-mxfp4.gguf \
    -md /path/to/dspark-DeepSeek-V4-Flash-MXFP4.gguf \
    --spec-type draft-dspark --spec-draft-n-max 2 --spec-draft-p-min 0 \
    -ngl 99 -ncmoe 99 \
    -t 72 -tb 72 \
    --numa distribute \
    -fa 1 -b 4096 -ub 1024 \
    -c 131072 -np 1 \
    --host 0.0.0.0 --port 8080
```

实测性能：**tg512 ≈ 24–25 tok/s，pp2048 ≈ 299 tok/s**。

要点解读：

- `-ngl 99 -ncmoe 99`：全部 dense/attention 层 offload 到 GPU，MoE 专家权重留在 CPU（`-ncmoe` 为上游参数）。
- `--numa distribute` + `GGML_NUMA_EP=1`：CPU 专家按 NUMA 行窗放置，两路 CPU 各自从本地内存流权重；
  `GGML_NUMA_HIER_BARRIER=1` 换分级 barrier，减少跨 UPI 同步开销。
- `GGML_REPACK_GEMV_PREFETCH=1`：AVX-512 repack GEMV 的软件预取（AVX2 机器无效，可省略）。
- dspark 调度档 `--spec-draft-n-max 2 --spec-draft-p-min 0`：每步起草 2 token、贪婪接受，
  是当前实测最优档；不接受更大 n-max。
- `-t 72` 对应 2 × 38 核平台减少量余量，见 §5 线程数建议；`-fa 1` 开 flash attention，
  `-b 4096 -ub 1024` 为 logical/physical batch。
- 显存富余时还可以把个别层的专家权重卸载到 GPU（`-ot "blk.N.ffn_*_exps.weight=CUDA0"`），每卸载一层 TG 约 +1%。

---

## 3. 推荐配置二：双机 EP（master + slave）

适合单机内存装不下模型，或想把 routed experts 的权重流量分摊到第二台机器的场景。
机制：master（CUDA build 的 `llama-server`）通过 `GGML_REMOTE_EP_*` 环境变量把认领层的 MoE FFN
经 RPC 发给 slave 上的 `llama-epd` worker（纯 CPU build），**传输的是 KB 级激活，不是权重**。
注意：这与上游 `llama-server --rpc`（RPC backend 整机 offload）是两套不同机制，EP worker 的
endpoint 由 `GGML_REMOTE_EP_SCHED_ENDPOINTS` / `GGML_REMOTE_EP_HOST:PORT` 指定。

### 3.1 基础双机（单 worker，classic 模式）

slave（build-noncuda / build-avx2）：

```bash
cd build-noncuda/bin
env LD_LIBRARY_PATH=$PWD GGML_EPD_NUMA=weighted \
  ./llama-epd -m /path/to/model.gguf --port 29200 --layers 36-42 -t 72 --no-mmap
# 有 RoCE 网卡时追加 GGML_REMOTE_EP_RDMA=1（master 同设）
```

master（build-cuda）：

```bash
env \
  GGML_REMOTE_EP=1 GGML_REMOTE_EP_HOST=10.0.0.2 GGML_REMOTE_EP_PORT=29200 \
  GGML_REMOTE_EP_LAYERS=36-42 \
  GGML_NUMA_EP=1 GGML_NUMA_HIER_BARRIER=1 GGML_REPACK_GEMV_PREFETCH=1 \
  build-cuda/bin/llama-server \
    -m /path/to/model.gguf -ngl 99 -ncmoe 99 -t 72 \
    --numa distribute -fa 1 -b 4096 -ub 1024 --no-mmap
```

分层点由两端带宽/算力比决定而非内存：DSV4（43 层 MoE）实测 slave 认领 7–8 层（如 36-42）最优。
slave `--no-mmap` 必须配 `GGML_EPD_NUMA=weighted`（单个 worker 跨两路 CPU 时），否则 ~80 G 权重
倾斜到单节点。现成脚本范例见 `tools/epd/bench-glm-master.sh` / `bench-glm-worker.sh`。

### 3.2 生产形态：四 NUMA 真 EP（每台机器每 NUMA 节点一个 worker）

完整配置（`SCHED_KLOCAL=0` 严格 EP、热点专家副本、`GGML_REMOTE_EP_RDMA=1`、画像派单等）见
[README.md](../README.md)「四 NUMA 真 EP 快速部署」一节与
[tools/epd/README.md](../tools/epd/README.md)，此处不重复。画像工具：

```bash
# master 侧采集 router 频率画像后，生成各 worker 的 --expert-list
GGML_REMOTE_EP_FREQ=1 GGML_REMOTE_EP_FREQ_FILE=/tmp/router-freq.csv ... llama-server ...
python3 tools/epd/ep-map-from-freq.py /tmp/router-freq.csv -w 4 -e 256 --extra-per-worker 64 --json
```

### 3.3 关键坑（血泪清单，务必读）

**a) worker 必须等旧进程死透再拉起。** `llama-epd` 带几十 GB RSS 时 SIGTERM 拆除可能远超 5 秒；
端口刚空出来不等于旧进程已退出，新 worker 与旧进程共存会双倍占内存甚至触发 OOM。
正确姿势（`/tmp/start-master-only.sh` 的模式）：

```bash
pkill -x llama-epd
for i in $(seq 1 60); do pgrep -x llama-epd >/dev/null || break; sleep 2; done
pgrep -x llama-epd >/dev/null && { echo "old workers still alive"; exit 1; }
# 确认死透后再启动新 worker，并等日志出现 "llama-epd: listening"（不是只看端口）
```

**b) slave 关机降级到 master-only 时，endpoint 要用 `127.0.0.1`，不要用 `10.0.0.1`。**
slave 关机后 master 上点对点链路 `10.0.0.1` 掉载波（no carrier），连自己的 `10.0.0.1:29202`
都不通；server 的 endpoint 必须写 `GGML_REMOTE_EP_SCHED_ENDPOINTS=127.0.0.1:29202`。

**c) master-only 降级模式参考 `/tmp/start-master-only.sh`**（该文件在本机存在时以它为准）。
其模式：杀干净旧 server/worker → 起 1 个全专家本地 worker（build-noncuda 的
`llama-epd --port 29202 --layers 0-42 --experts 0-255 -t 72 --no-autotune --no-mmap`，
env `GGML_NUMA_EP=1 GGML_NUMA_HIER_BARRIER=1`）→ 轮询等端口就绪 → server 挂单 endpoint
`127.0.0.1:29202`（`GGML_REMOTE_EP=1 GGML_REMOTE_EP_LAYERS=0-42 GGML_REMOTE_EP_SCHED=1 ...`）
→ 健康检查 + 一条 32-token completion 冒烟。生产 server 建议用 `systemd-run --user --scope`
包起来（脚本中的 `xllama-dsv4-production.scope`），便于 `systemctl --user stop` 干净拆除。

其他高频坑：

- RDMA 变量叫 `GGML_REMOTE_EP_RDMA`，**不存在** `GGML_EPD_RDMA`；master 与所有 worker 两侧必须同设。
- worker 线程 `-t` 勿超物理核数；交给 autotune（不传 `-t`）或设物理核数。
- master 侧跨机 EP 生产一律 `--no-mmap`（mmap 冷缓存页错位钉死会 PP 减半，见 PARAMETERS §8）。
- `GGML_REMOTE_EP_SCHED` 与 `GGML_REMOTE_EP_MIRROR` 互斥（同设 SCHED 赢）。

---

## 4. 纯 CPU 推理配置

双路机器、无 GPU 或不用 GPU 时（DeepSeek-V4-Flash mxfp4，2 × 38C/251 GiB 实测 RSS ~165 GiB、
TG512 ≈ 12 tok/s 量级）：

```bash
env \
  CUDA_VISIBLE_DEVICES="" \
  GGML_NUMA_EP=1 GGML_NUMA_HIER_BARRIER=1 GGML_NUMA_EP_GATE_UP_PARALLEL=1 \
  build-noncuda/bin/llama-server \
    -m /path/to/model.gguf \
    -ngl 0 -t 72 --numa distribute --no-mmap \
    -fa 1 -b 2048 -ub 512
```

要点：

- **`CUDA_VISIBLE_DEVICES=""` 是铁律**（注意是 `=""` 不是 `=`）：CUDA 设备可见 + `-ngl 0` 时
  DSA 稀疏注意力会被误分派到 CUDA 而 layer 在 CPU，整段退化慢 ~2×。
- `GGML_NUMA_EP_GATE_UP_PARALLEL=1` 只在双节点、repack 权重、小批 decode 下生效（不满足自动回退），
  纯 CPU TG 再 +5% 左右。
- 想进一步提速且内存充足，可试 `--numa mirror --numa-mirror weights`（非专家权重每节点复制一份；
  EP 下专家仍是单份行窗）。内存吃紧保持 `--numa distribute`。
- `GGML_NUMA_EP` 要求 `--no-mmap`（或显式 `GGML_NUMA_EP_MMAP=1`，但有已知坑，见 PARAMETERS §8）。

---

## 5. 常见问题

### mmap 还是 no-mmap？

- **EP / mirror 生产配置一律 `--no-mmap`**：权重一次性 pread 进匿名内存，RSS 全量常驻、零页入、
  免疫页缓存驱逐；NUMA 放置/mbind 迁移也只对匿名内存可靠生效。
- mmap 的代价：冷缓存首次命中按 interleave 落页，配合 `GGML_NUMA_EP_MMAP=1` 会被 `mbind` 钉死在
  错位节点（PP 减半）；慢盘（U 盘/外置盘）下 mmap 基本不可用。
- 例外：只想快速试一下、机器内存富余且不在乎尖峰时，mmap 启动更快（免全量读盘）。EP worker 若坚持
  mmap，至少开 `GGML_EP_PREFAULT=1` 预触权重页消除冷专家尖峰（尖峰率 30%→~1%）。
- 注意 `llama-cli` 用 `--no-mmap`、`llama-bench` 用 `--mmap 0`，两者互不通用。

### 线程数怎么设？

经验法则：**物理核心数 − 2**（给 OS/中断/IO 留余量）。例：2 × 38 核平台用 `-t 72` 左右，
`-tb`（threads-batch）同值。不要超过物理核数——超线程对带宽敏感的 MoE GEMV 是负收益
（实测 `-t` 超物理核后 TG 崩到个位数）。`llama-epd` 不传 `-t` 时会自动阶梯标定（autotune），
直接信任它即可。

### Q8 KV 的上下文占用？

`-ctk q8_0 -ctv q8_0` 把 KV cache 从 F16 压到 Q8_0，**KV 字节数减半**，长上下文容量直接翻倍量级。
双 3090 实测（DSV4，1M context/槽）：F16 `1M×2` 占 GPU0/1 约 21.4/12.5 GiB，Q8 连续布局 `1M×2`
约 16.1/12.3 GiB、`1M×3` 约 20.5/12.3 GiB（`1M×4` 在 PP compute buffer 预留阶段 OOM）。
代价：Q8 KV 不保证与 F16 bit-exact，多轮长上下文质量回归需自行验收。多槽完整 1M 的通式是
`-c $((1048576*n)) -np n --no-kv-unified`（`-c` 是总 context，不是每槽），起服后必须
`curl /slots` 核对每槽 `n_ctx` 达标。

### 更多

- 纯 CPU 退化慢 2×？检查 `CUDA_VISIBLE_DEVICES=""`（§4）。
- 双机 A/B 测速有 ~25% 运行顺序效应：必须同会话反序复测（ABBA），且跑前
  `sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'`、`numactl --interleave=all` 统一基线口径。
- 多个模型进程全程 `flock -x /tmp/xllama-bench.lock`，防止并发加载 OOM。
- 完整已知坑清单见 [PEOPLESLLM-PARAMS.md](PEOPLESLLM-PARAMS.md) 第 5 节。
