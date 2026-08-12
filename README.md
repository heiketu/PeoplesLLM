# PeoplesLLM

> **[English README →](README.en.md)**

**让 200B~3T 级超大 MoE 模型在本地廉价硬件上跑起来、跑得快。**
llama.cpp 特化分支，面向双路/多路 CPU 服务器 + 消费级 GPU：CPU 计算内核、NUMA 体系、GPU 超长上下文 prefill、跨机专家并行四条主线全部自研优化。

![头条加速比](docs/benchmarks/headline_speedups.png)

## 简介

作者硬件方向出身，本项目全部设计、编码、测试、调参由 AI（Kimi K3 + Kimi Code CLI）完成，作者只指定技术方向（EP、GEMM 内核、AVX512/VNNI/VBMI、NUMA 等）。代码可读性可能不佳、可能有潜在 BUG，目前只对个人平台（双路 Xeon 8360Y + 2× RTX 3090）上的运行负责。主要调优对象：**DeepSeek-V4 / DSV4-Flash** 与 **GLM-5.2**。欢迎 issue。

测试平台：双路 Xeon 8360Y（Ice Lake，152 线程，251G RAM）+ 2× RTX 3090 24G（NVLink）+ ConnectX-5 100G 直连从机（双路 Ice Lake ES，188G）。基线 = 上游 llama.cpp 同机同口径 A/B。

## 特性一览

### CPU 推理

| 特性 | 使用场景 | 性能影响 |
|---|---|---|
| **AVX512/VNNI/VBMI 8×8 repack 内核**（Q2_K~Q6_K、Q8_0、MXFP4、IQ1_S/IQ1_M/IQ2_XXS 全格式） | 所有 CPU 计算；长 prompt 批量收益最大 | prefill gemm 微基准最高 **4.9×**；GLM-5.2 端到端 **PP 4.1×** |
| **IQ2_XXS AVX512 repack gemv/gemm 内核**（x8 布局） | IQ2_XXS 权重模型 CPU 推理 | gemv 微基准 **17×**；全模型 A/B 暂持平（瓶颈在专家 mul_mat_id 未覆盖），模型级收益待扩展 |
| **NUMA 镜像 mirror**（`--numa mirror`） | 双路服务器、内存充裕 | TG 最优（vs 主线 **+9%**），跨插槽 UPI 流量归零 |
| **NUMA 行窗 EP**（`GGML_NUMA_EP=1`） | 内存受限、超大模型装载 | 专家权重**内存减半**，TG 追平 mirror，dst 行单写者零归并 |
| **融合算子**（超连接 HC、MoE router、RMS_NORM 吸收、DSA 闪电索引器） | DSV4 / GLM 全场景 | 消除分解路径与中间激活读写，PP 固定开销显著降低 |
| **MTP 投机解码** | DSV4 decode | TG 进一步提升（本文数据均未开 MTP） |

### GPU 推理（超长上下文）

| 特性 | 使用场景 | 性能影响 |
|---|---|---|
| **layer-major 超长 prefill**（`llama_decode_layer_major()`，层权重驻留 CUDA slot） | GPU-local MoE 的 4K~1M 超长 prompt | 16K PP 161→**604 tok/s**（稀疏 compact opt-in **752**），4.7×；远程 CPU EP + DSpark 不适用，见最新进展 |
| **MoE 流式 prefill + 3-slot 双卡预取** | 专家权重在 host 内存的混合推理 | 每层权重只上传一次，跨 tile 复用 |
| **真双卡同层 expert-axis EP**（`GGML_CUDA_MOE_PP_EP`，opt-in） | 双 GPU prefill | 2K PP **+63%**（277→452 tok/s），输出逐位一致 |
| **batched top-k**（`GGML_CUDA_BATCHED_TOPK`） | DSA 稀疏注意力模型长上下文 | top-k 内核 **21.6×**，e2e PP +8.2% |
| **q1 32-head FA + raw-SWA decode ring**（ring 默认开，`LLAMA_DSV4_COMPACT_DECODE_SWA=0` 可关） | 长上下文 decode | q1 decode 图宽与 prompt 长度解耦，fixed TG64 8.89→**12.09**（+33~36%）、TG512 +4~8%；多 slot 自动回退 |
| **q8 compact 稀疏 FA**（默认开，q8_0 KV） | q8 KV 长上下文 decode | q1 decode 只物化 top-k 选中行，TG64 9.75→**10.69**（+10%），最快 q1 decode 路径（超 f16 dense）；f16 fused sparse 保持 opt-in（16K 实测 -12%） |
| **多流（小 q）稀疏 FA**（`LLAMA_DSV4_FUSED_INDEXED_FA=3/4`、`LLAMA_DSV4_Q8_SPARSE_FA=2`，opt-in） | 多 slot 并发 decode | 正确性三方 byte-identical；≤16K 持平（权重带宽主导），收益主场 256K+ 长上下文 |
| **GPU 流式 prefill**（方案A，server 集成） | server 长 prompt | 整 tile 路径 PP 63→**127.6 tok/s**（2×）；紧凑环形 SWA 缓存下按资格门回落 chunked，`--swa-full` 启用 |
| **GPU 专家卸载**（`-ot blk.N.ffn_*_exps=CUDA0/1`） | 显存富余 | 每卸载一层 TG 约 +1% |

### 跨机分布式 EP（`tools/epd/`）

| 特性 | 使用场景 | 性能影响 |
|---|---|---|
| **跨机专家并行**（激活派发，KB 级流量，非权重传输） | 单机内存装不下模型 | DSV4 双机追平单机速度 + master 内存**省 26%** |
| **RoCEv2 RDMA 传输**（`GGML_REMOTE_EP_RDMA=1`，TCP 自动兜底） | 有 IB/RoCE 网卡；纯千兆环境自动回退 | RTT 42-74µs→**10-13µs**；大帧修复后 PP1020 33.4→**76.1 tok/s（2.3×）** |
| **MAX-EFFORT 层镜像**（`GGML_REMOTE_EP_MIRROR=1`） | master 内存有空闲、decode 为主 | 远程段从关键路径消失，TG **+9~11%** |
| **单 slot 四 NUMA 真 EP**（`SCHED_KLOCAL=0` + 热点专家副本） | 2 台双路机器共同计算同一请求 | GLM-5.2 PP512：2 NUMA **24.13**→4 NUMA **40.59 tok/s（1.682×）**；75 层输出逐字节一致 |
| **EP 规划器**（`tools/epd/ep-plan.py`） | 部署前分层决策 | 按实测带宽/延迟给分层点，校准误差 ≤1.5% |

### 模型支持

DeepSeek-V4 / DSV4-Flash（含原生 MXFP4）、GLM-5.2（DSA）、MiniMax-M3；GGUF 字节级修复工具链（量化块腐化扫描+补丁，修复过 GLM-5.2 官方文件 138 个腐化块导致的乱码）。

## 安装与构建

以下以 Ubuntu 24.04、NVIDIA CUDA 和可选 RoCEv2 为例。master 需要 CUDA build，纯 CPU
EP worker 使用 non-CUDA build；两台机器必须使用同一版本的 `llama-epd` 和协议代码。

```bash
sudo apt update
sudo apt install -y git cmake ninja-build build-essential pkg-config \
    libcurl4-openssl-dev libssl-dev numactl libnuma-dev \
    rdma-core libibverbs-dev librdmacm-dev jq

git clone https://github.com/heiketu/PeoplesLLM.git
cd PeoplesLLM

# master：CUDA、llama-server
cmake -S . -B build-cuda -G Ninja \
    -DGGML_CUDA=ON -DLLAMA_CURL=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build-cuda -j --target llama-server

# master/slave：Ice Lake AVX512 CPU worker
cmake -S . -B build-cpu -G Ninja \
    -DGGML_CUDA=OFF -DGGML_NATIVE=ON -DLLAMA_CURL=OFF \
    -DCMAKE_BUILD_TYPE=Release
cmake --build build-cpu -j --target llama-epd llama-ep-transport-bench
```

CMake 应打印 `EP RDMA transport (RoCEv2): ON`；否则运行时即使设置
`GGML_REMOTE_EP_RDMA=1` 也只能回退 TCP。CUDA toolkit/driver 的安装随发行版而异，构建前先用
`nvidia-smi` 和 `nvcc --version` 验证。把 `build-cpu/bin/llama-epd` 及同目录动态库同步到
slave，并确保两机都能读取同一个 GGUF（路径可以不同）。

## 四 NUMA 真 EP 快速部署

本模式是一个请求的 routed experts 分散到四个 NUMA worker，**不是每台机器复制一份模型跑
四个 slot**。推荐先用 `--expert-mod R/4` 建立无副本基线，再按真实 router 画像生成热点副本图：

```bash
# 采集后生成；输出 JSON 的 workers[i].expert_list 可直接传给 llama-epd
python3 tools/epd/ep-map-from-freq.py router-freq.csv \
    -w 4 -e 256 --extra-per-worker 64 --json > ep-map.json

# 每个 NUMA node 启一个 worker。IDX/NODE/PORT 分别为：
# master (0,0,29202)、master (1,1,29203)、slave (2,0,29200)、slave (3,1,29201)
IDX=0 NODE=0 PORT=29202 MODEL=/path/to/target.gguf MAP=/path/to/ep-map.json
EXPERTS=$(jq -r ".workers[$IDX].expert_list" "$MAP")
env GGML_REMOTE_EP_RDMA=1 GGML_EPD_POLL=50 \
  numactl --cpunodebind="$NODE" --membind="$NODE" \
  build-cpu/bin/llama-epd -m "$MODEL" --port "$PORT" --layers 0-42 \
  --expert-list "$EXPERTS" -t 36 --no-autotune --no-mmap
```

四个 worker 都必须设置准确的 `GGML_REMOTE_EP_RDMA=1`；不存在
`GGML_EPD_RDMA` 这个开关。端口开始监听不等于模型已就绪，应等待日志中的
`llama-epd: listening`，并确认日志带 `[rdma]`、`rdma resource show qp` 能看到连接。
systemd transient unit 重用同名时还要给 `journalctl --since` 指定本次启动时间，避免匹配旧日志。

### DSV4-Flash + DSpark，F16 KV、1M × 1 生产 server

`-c` 是非 unified 模式下所有 slot 的总 context。每槽 1M 必须用
`-c $((1048576*n)) -np n --no-kv-unified`；`-c 1048576 -np 2` 得到的是每槽 512K。
当前吞吐/质量优先配置使用 `n=1` 和默认 F16 K/V。43 层 target
dense/attention/router 与 target KV 连续放在 CUDA0；DSpark 主体和 draft KV 放在
CUDA1。DSpark GGUF 不自带 token embedding/LM head，因此共用的 embedding/output
边界张量也放 CUDA1，避免每层跨卡，只保留输入与最终 logits 两个边界传输：

```bash
TARGET=/path/to/DeepSeek-V4-Flash-Q4-mxfp4.gguf
DRAFT=/path/to/dspark-DeepSeek-V4-Flash-MXFP4.gguf

env \
  GGML_REMOTE_EP=1 \
  GGML_REMOTE_EP_LAYERS=0-42 \
  GGML_REMOTE_EP_SCHED=1 \
  GGML_REMOTE_EP_SCHED_KLOCAL=0 \
  GGML_REMOTE_EP_SCHED_MAX_EFFORT=1 \
  GGML_REMOTE_EP_SCHED_PP=1 \
  GGML_REMOTE_EP_SCHED_ENDPOINTS=10.0.0.1:29202,10.0.0.1:29203,10.0.0.2:29200,10.0.0.2:29201 \
  GGML_REMOTE_EP_RDMA=1 \
  GGML_EP_RDMA_SPIN=1 \
  GGML_REMOTE_EP_PIPE=0 \
  GGML_REMOTE_EP_PARALLEL_IO=1 \
  GGML_REMOTE_EP_WEIGHT_ON_MASTER=1 \
  GGML_REMOTE_EP_SCHED_TG_REPEAT_COST=250 \
  GGML_REMOTE_EP_SCHED_PP_REPEAT_COST=250 \
  GGML_REMOTE_EP_RECONNECT_TIMEOUT_MS=90000 \
  GGML_CUDA_BATCHED_TOPK=1 \
  GGML_CUDA_DSV4_KV_REUSE=1 \
  build-cuda/bin/llama-server \
    -m "$TARGET" -md "$DRAFT" \
    --spec-type draft-dspark --spec-draft-n-max 3 --spec-draft-p-min 0 \
    -dev CUDA0,CUDA1 -sm layer -ts 1,0 \
    -ot '^token_embd\.weight$=CUDA1,^output.*=CUDA1' \
    --spec-draft-device CUDA1 \
    -ngl all -ncmoe 99 -ngld all -t 72 -tb 72 -b 256 -ub 256 \
    -c 1048576 -np 1 \
    --no-kv-unified -fa on -fit off \
    --numa distribute --no-warmup --no-ui --host 0.0.0.0 --port 18108
```

需要后台常驻时，用 `systemd-run --user --unit=peoplesllm-dsv4-1m1-f16 --property=Restart=on-failure`
包住上面的 `env ... llama-server ...` 命令；用
`systemctl --user status/stop peoplesllm-dsv4-1m1-f16`
查看或停止。`0.0.0.0` 会监听所有网卡；本机实测 LAN 地址是
`http://192.168.0.48:18108`。跨不可信网络开放前应加 API key、TLS、可信反向代理或防火墙白名单。

使用 OpenAI 兼容聊天接口：

```bash
curl -sS http://192.168.0.48:18108/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"dsv4","messages":[{"role":"user","content":"1+1 等于多少？"}],"temperature":0,"max_tokens":128}'

# 核对每个 slot 的实际 context；每项都应显示 n_ctx=1048576
curl -sS http://127.0.0.1:18108/slots | jq '.[] | {id,n_ctx,speculative,is_processing}'
```

容量优先时可改为 `-c 3145728 -np 3 -ctk q8_0 -ctv q8_0`；双 3090 已验证
`1M×3`，但 Q8 KV 不与 F16 bit-exact，且单请求更慢。`1M×4` 已在 CUDA0 的 PP compute
buffer 预留阶段真实 OOM。无论槽数是多少，`-c` 都是总 context，必须满足
`-c $((1048576*n)) -np n --no-kv-unified` 才是每槽完整 1M。

完整配置语义、内存约束和故障排查见
[参数手册](docs/PEOPLESLLM-PARAMS.md)、[EP 专项文档](tools/epd/README.md) 与
[1M 上下文手册](docs/LONG-CONTEXT-1M.md)。

## 最新进展（2026-08-06~13）

- **PP repeat-affinity 小幅提速并进入生产配置**：PP dealer 现在可让同一批次里重复命中的热点专家尽量留在同一 holder，减少重复权重流；`GGML_REMOTE_EP_SCHED_PP_REPEAT_COST=250` 对默认 1000 的同组 6 轮 A/B 为 263.555 vs 261.108 tok/s，**+0.94%**。默认值仍是 1000 以保持兼容，当前 DSV4 生产显式设 250。
- **MXFP4 5–8 行 shared-weight 内核只保留能力，不进入生产热路**：AVX512 内核和 NR2..8 bit-exact 回归均通过；但真实 expert-first 派发从 NR2..4 放宽到 NR2..8 后，排除首个冷请求的热态均值 267.128，对配对旧路径 268.443 tok/s 为 **-0.49%**。因此 expert-first 已恢复只合并 2–4 行，不能把早先约 +0.9% 的非配对波动当成收益。
- **DSpark + 远程 CPU EP 的 layer-major 已验证为负收益并默认关闭**：普通 UB256 为 263.17 PP / 37.70 TG；layer-major host-HC 为 185.97 / 29.71，device-HC 为 259.09 / 30.15 tok/s；整套 `-b/-ub 2048` device-HC 也只有 228.51 / 21.77。原因是 GPU layer-major 权重驻留无法减少远端 CPU 专家权重流，反而增加 HC 边界搬运并受 raw-SWA decode 状态约束。框架入口保留给 GPU-local/未来 HBM 实验，`LLAMA_LAYER_MAJOR_SPECULATIVE` 默认关，生产未设置。
- **1M×1 的 PP 默认从 UB64 升到 UB256**：同一 F16 KV、NMAX=3、四路 KLOCAL=0 EP 下，16K 独立长提示三轮为 270.01/267.12/270.95 tok/s，均值 **269.36 tok/s**；旧 UB64 三轮均值 235.37，提升 **14.4%**。UB2048 虽达到 298.20 tok/s，但 TG 热态下降约 2.8%，故生产选择 UB256。UB4096/8192 在 1M 配置下因 GPU0 compute buffer OOM 被正确拒绝，而不是再被 RDMA 12MiB 接收环误拦。
- **EP 拓扑热路缓存与重连一致性**：pure EP 在建图时一次生成 dealer holder 表，不再每个 MoE 调用重扫四份 CAP bitmap；固定请求 2386 次调用的 `deal+send` 累计 177.6→172.1ms（-3.1%），复测后四轮均值 36.84 tok/s，输出 SHA256 与 DSpark 145/78 接受数不变。CAP/稀疏 bitmap 已拆到独立可测试模块，worker 重连若 expert map、kernel ID 或能力位变化会拒绝继续执行。
- **DSV4 单 slot 四 NUMA 真 EP 达到 37.921 tok/s**：43 层 routed MoE 由 master/slave 各两个 NUMA worker 共同计算，target dense/attention/router 与 DSpark draft 在双 RTX 3090；NMAX=3、四 worker `-t 36`、e128 热点副本图、真 RDMA、仅 master CQ spin 的热态六轮为 38.016/37.429/37.873/37.695/38.423/38.092 tok/s，输出 SHA256 全部一致。worker 也 spin 的均值 37.659，已否决。
- **Q8 KV 容量档达到每槽完整 1M × 3**：target 43 层与 target KV 连续放 CUDA0，DSpark 与 draft KV 放 CUDA1，共用 embedding/output 也放 CUDA1；GPU0/1 实占 20,456/12,304 MiB，三个 slot 均为 1,048,576。标准 128-token 请求热态六轮均值 34.260 tok/s，只比同布局 `1M×2` 的 34.436 低 0.51%；`1M×4` 在 CUDA0 申请 2.73GiB PP compute buffer 时真实 OOM。三槽并发合计约 41.6 tok/s，并不高于旧 F16 双槽的 42.998，因此第三槽是容量档而非总吞吐优化，不是当前 F16 默认。单槽固定请求 7/7 输出一致，三个独立算术 chat 顺序/并发均返回 2/4/6；raw continuation 的近似并列 logits 会随合批路径分叉，不能拿 raw 字节哈希误判 slot 污染。
- **AVX512 MXFP4 重复专家路径**：同一次请求 2/3/4 个 assignment 命中同一专家时复用权重流，真实 worker 微门 gate/up 约 +24%、down 约 +22%；请求紧凑 gather 与 repeat-aware dealer 继续降低小 batch 固定开销。
- **EP 紧凑激活量化并行修正**：REQ2/REQ4 把 assignment 放在张量 `ne12` 维，而旧 CPU_REPACK 只按 `ne11` 分线程；生产形状 `ne11=1` 时 F32→Q8 激活量化因此长期只有线程 0 工作。现在按完整 `(ne12,ne11)` 行空间分摊，单行 TG 行为不变。六行微基准 gate/up（K=7168）由 31~33 µs 降到 19~20 µs、down（K=2048）由 17~20 µs 降到约 15 µs；两次部署复测六轮均值为 36.573/36.902 tok/s，对旧 worker 36.280 为约 +0.8~1.7%，输出与 draft 接受数不变。
- **e144 副本图没有升级为生产默认**：每路 144 个专家的两组六轮均值为 37.075/36.843 tok/s，12 轮总均值 36.959；相邻 e128 六轮 36.902，仅 +0.16%，低于抖动且四路内存峰值升到约 97.6/99.9/96.5/96.8G，故恢复 e128，也不再浪费时间测试 e160。
- **pure EP worker 重启不再立即打崩 master**：`GGML_REMOTE_EP_RECONNECT_TIMEOUT_MS` 允许 SCHED 在断连后等待 worker 重新监听并重发暂存请求，健康热路径无额外工作。生产设为 90000 ms；真实重启 slave w2 后，请求等待约 66 秒并正确完成，正文 hash 与 draft 接受数不变，master PID 未变、`NRestarts=0`。systemd unit 另设 `Restart=on-failure` 作为最终兜底。

- **AVX512 EPD 路径完整化**：修复零字节 CPU_REPACK trait 探测导致真实模型静默退回 raw kernel 的问题；兼容的 separate gate/up 在加载期融合，mmap 原始页在每个张量转换完成后立即回收，线程标定改测每路实际紧凑 assignment。GLM-5.2 layer 3、64 专家受控 A/B 中，warm REQ2 PP128 从 33.14~36.73 ms 降至 31.53~31.60 ms，TG 从 0.505~0.595 ms 降至 0.460~0.477 ms；输出一致。该数据仅代表 CPU MoE worker 层内计算，不混入 GPU dense/attention。
- **异构 worker 动态负载反馈**：热点副本 dealer 不再假定四个 NUMA 等速；它分别学习每路 TG/PP 的空队列服务时间，并用在飞虚拟工作量派单。纯策略、溢出与 dealer 回归已通过；该改动尚未计入下述 40.59 tok/s，需在四路安静窗口复测。
- **单 slot 四 NUMA 真 EP（GLM-5.2）**：master 与 slave 的每个 NUMA node 各一个 worker，`KLOCAL=0` 下 master 不复制专家权重；ragged 分配紧凑执行、PP 图缓存分层、真实 PP 频率画像 + 76 份热点副本后，PP512 从 2 NUMA 24.13 提升到 4 NUMA 40.59 tok/s（+68.2%，延迟 -40.6%），128-token 输出逐字节 MATCH。
- **raw-SWA decode ring 默认启用**（分支 fa-decode-fix）：q1 decode 图宽与 prompt 长度解耦，fixed TG64 8.894→11.837/12.085 tok/s（+33~36%）、TG512 +4~8%；多 slot 场景自动回退全宽语义。
- **q8 compact 稀疏 FA 默认启用**：q8_0 KV 下 q1 decode 只物化 top-k 选中行，TG64 9.75→10.69（+10%）、TG512 +1.3%，成为最快 q1 decode 路径（超 f16 dense）；f16 fused sparse 保持 opt-in（16K 实测 -12%，长上下文再评估）。
- **多流（小 q）稀疏 FA**（opt-in，`LLAMA_DSV4_FUSED_INDEXED_FA=3/4`、`LLAMA_DSV4_Q8_SPARSE_FA=2`）：多 slot 并发 decode 的稀疏扩展，正确性三方 byte-identical 验证；≤16K 性能持平（权重带宽主导），收益主场在 256K+ 长上下文。
- **GPU 流式 prefill（方案A，server 集成）**：长 prompt 整 tile 路径 PP 63→127.6 tok/s（2×）；紧凑环形 SWA 缓存下按资格门自动回落 chunked，`--swa-full` 可启用；指纹四组逐位命中。
- **IQ2_XXS AVX512 kernel**：gemv 微基准 **17×**；全模型 A/B 持平（3.78→3.71，瓶颈在专家 mul_mat_id 未覆盖）——微基准提升、模型级收益待扩展。

## 性能实测

### 对标主线 llama.cpp（同机同口径 A/B）

DeepSeek-V4 284B（Q3_K），GPU 卸载 14 层专家，72 线程：

![DSV4 vs upstream](docs/benchmarks/dsv4_vs_upstream.png)

行窗 EP 与 mirror 双双反超主线：**PP 2.1-2.2×、TG +9%**；行窗 EP 额外省一半专家内存。PP 优先或单路场景 isolate 配置 pp512 可达 370 tok/s。

### vs 主线全档复测（2026-08-07，DSV4-Flash 284B mxfp4 生产形态）

![vs mainline 2026-08-07](docs/benchmarks/vs_mainline_20260807.png)

主线 `e9fa0781f` vs 本分支（含 Q2_K/mxfp4 VNNI dpbusd 化），同机同口径 llama-bench A/B。**GPU 卸载（-ngl 99 -ncmoe 99 EP，生产形态）：PP 全档 +40~59%（pp2048 265.1 vs 166.5、pp8192 257.7 vs 164.5、pp16384 227.5 vs 163.1）、TG +20%（23.3 vs 19.5）**；纯 CPU（-ngl 0）：PP +48%（135.1 vs 91.3），TG 已知回归（3.7 vs 8.1，−54%，EP 无关系，隔离定位中——GPU 卸载为生产场景，不受影响）。Q2_K repack 经 dpbusd 化后 PP 164→218（+33% 端到端），仍略落后 no-repack（244），Q2_K 建议维持 `--no-repack`。

### GLM-5.2 745B：IQ traits + gemm 分流

![GLM traits](docs/benchmarks/glm_traits.png)

### CPU 内核：8×8 repack vs 主线 legacy vec_dot

![CPU kernel speedup](docs/benchmarks/cpu_kernel_speedup.png)

gemv（decode）≈1× 是内存带宽物理上限；prefill gemm 提升来自 8×8 重排摊销权重读带宽。完整 300 格数据可用 `tests/test-repack-kernels --perf` 复现。

### NUMA 局部性：为什么 EP/mirror 必然赢过主线 distribute

![NUMA locality](docs/benchmarks/numa_locality.png)

主线 `--numa distribute` 把权重页 interleave 摊到两路，每次权重读约 50% 跨 UPI；实测跨路读带宽只有本地的 ~38%（53.7-54.6 vs 136-143 GB/s，membw2 76 线程/路）。行窗 EP 与 mirror 结构性做到 ~100% 本地读，双路合计有效带宽 279.2 GB/s，比主线 interleave 模式（177.6 GB/s）高 **57%**。带宽矩阵全表见 `docs/CHANGES.md`。

### 纯 CPU 推理（-ngl 0，同机 A/B）

![Pure CPU vs upstream](docs/benchmarks/pure_cpu_vs_upstream.png)

DSV4 284B Q3_K 纯 CPU（72 线程 interleave）：**PP +38%、TG +17-18%**。2026-08-05 负载环境复测，安静窗口定稿后更新。

### 多 slot 并发与混合配置复测（2026-08-05）

![Multi-slot concurrency](docs/benchmarks/multislot_concurrency.png)

llama-server 8 槽并发压测（每槽 512 tok）：**EP（行窗）配置全并发档领先主线——1 槽 +22%、8 槽 +18%（74.5 vs 63.2 tok/s）**；mirror 作为兼容性选项在 8 槽被主线反超 7%（高并发下其每令牌权重流量摊薄、结构性带宽优势贬值），生产多并发请用 `GGML_NUMA_EP=1` 配置。混合配置（14 层专家上双卡）同口径 llama-bench：EP pp512 **227.8 vs 主线 158.4（+44%）**、tg512 31.3 vs 29.1（+7.5%）；mirror pp512 200.2（+26%）。

### 超长上下文：layer-major prefill（DSV4-Flash，16K）

![16K PP 演进](docs/benchmarks/longctx_pp_progression.png)

![MXFP4 Hybrid CPU 审计与双卡 EP](docs/benchmarks/mxfp4_hybrid_cpu_audit.png)

原生 MXFP4 版（155GB，137GiB 专家单份 CPU_REPACK + 双路 NUMA EP）经 CPU 审计三连修后 4K PP +144%。

### 长上下文 decode（16K，固定负载 A/B）

![16K TG 改进](docs/benchmarks/longctx_tg_improvements.png)

长上下文 TG 衰减根因已定位为 GPU attention/KV 的物理 dense 扫描，逐项修复（raw-SWA ring 已于 08-06 默认化，见下节）。16K GPU 侧时间分解（Nsight）：

![16K 热点分解](docs/benchmarks/longctx_hotspots.png)

### FA decode 结构性修复：TG64 累积演进（2026-08-06/07）

![FA decode TG64 累积演进 / progression](docs/benchmarks/fa_decode_tg64_progression.png)

左：f16 KV，raw-SWA decode ring 默认化把 fixed TG64 从 8.894 推到 12.085 tok/s（+36%），多 slot 自动回退。右：q8_0 KV，compact 稀疏 FA 只物化 top-k 选中行，9.75→10.69（+9.6%），超越 f16 dense（9.66）成为最快 q1 decode 路径。多流稀疏 FA（opt-in）≤16K 持平，收益主场在 256K+ 长上下文。

### 双机 expert-parallel（GLM-5.2，100G RoCEv2 直连）

![双机 EP](docs/benchmarks/dual_machine_ep.png)

最新单 slot 真 EP 口径（2026-08-10，GLM-5.2 UD-Q2_K_MXFP4，MoE 层 3–77，dense/attention 在双 3090，CPU 只跑 routed experts）：

| 对比 | 单机 2 NUMA worker | 双机 4 NUMA worker | 4 / 2 |
|---|---:|---:|---:|
| MoE 阶段 decode 时间（75 层合计，扣除 prompt） | 86.71 ms/token | 69.58 ms/token | **1.246×** |
| PP512（b512/ub256，同代码三轮均值） | 24.13 tok/s（21.22s） | **40.59 tok/s（12.62s）** | **1.682×** |

PP 映射来自该 workload 的 `layer,expert,count` 画像：四路各保留 64 个 primary + 16–23 个热点副本（80–87/256 专家），在线 dealer 在 holder 集合内逐 token 选最轻端点。逐层关键路径负载/理想平均从 1.374× 降到 1.091×；worker RSS 约 71–78GiB/NUMA，未使用 swap。MoE decode 与 PP 均完成 128-token 逐字节参考对拍。

### 生产 server：DSV4-Flash 每槽 1M 上下文

![Flash PP/TG 曲线](docs/benchmarks/flash_pp_tg_curve.png)

当前后台是 **F16 KV、单槽完整 1M、UB256**。连续布局让 target 43 层与 target KV 留在
CUDA0，DSpark、draft KV 及共用 embedding/output 位于 CUDA1。16K PP 三轮均值
269.36 tok/s；本轮恢复生产 2–4 行派发前的配对三轮为 267.114/269.287/268.928，均值
268.443 tok/s。最终四 worker 回退后，固定 128-token 请求三轮为
35.642/36.898/36.386，均值 36.309 tok/s，三轮 draft/accepted 均为 145/78 且输出一致。
历史最佳短上下文配置
37.921 tok/s 保留为性能上限，不与 1M context 口径混算。

PP 极限档把 `-b/-ub` 提到 2048 时，热态约 318.08 tok/s，相对同组 UB256 约 +20.7%，
但 TG 36.092 对 36.707 tok/s 下降 1.68%，所以没有替换当前 TG/质量优先的 UB256。
现阶段 `TG 40+ / PP 1000+` 目标尚未达到；README 只记录已复测结果，不把实验峰值或失败路径
写成生产性能。

容量档另行验证过 Q8 KV `1M×3`：GPU0/1 实占 20,456/12,304MiB，三个 slot 均由
`/slots` 确认为 `n_ctx=1048576`，单请求热态 34.260 tok/s；`1M×4` 在 CUDA0 申请
2.73GiB PP compute buffer 时 OOM。因此三槽是双 3090 的 Q8 容量上限，不是当前质量优先
默认。历史 F16 `1M×2` 的 35.526 tok/s、并发总计 42.998 tok/s继续作为对照。

> 所有数字均为实测，口径与复现方式见 `docs/CHANGES.md` 与 `docs/benchmarks/`（绘图脚本同目录）。已实测否决的路线（full tensor split -44%、meta-backend TP -20%、跨 tile 双 scheduler）也记录在案。

## 文档

- **[docs/CHANGES.md](docs/CHANGES.md)** — 完整改动清单（NUMA 体系、CPU 内核格式矩阵、融合算子、分布式 EP）
- **[docs/PEOPLESLLM-PARAMS.md](docs/PEOPLESLLM-PARAMS.md)** — 全部 env/CLI 参数手册与生产配方
- **[docs/DEVELOPMENT.md](docs/DEVELOPMENT.md)** — 构建、测试与分支约定
- **[docs/LONG-CONTEXT-1M.md](docs/LONG-CONTEXT-1M.md)** — 1M 上下文内存/显存预算与验收门槛
- **[tools/epd/README.md](tools/epd/README.md)** — 双机 EP 快速上手

## 当前状态

早期开发阶段，生产使用前请按本文的输出哈希、slot context 和显存余量门槛自行验收。持续推进：
GPU prefill 向 vLLM/KTransformers 级吞吐演进（16K 目标 1000+ tok/s）、CPU/GPU 联合流水、
统一 GPU-prefill/CPU-decode 权重表示。当前主线同步目标为 llama.cpp
`030ebb558`（2026-08-11），`vendor`/专题分支保留可复测基线。

## 许可证与版权

本项目是 [llama.cpp](https://github.com/ggml-org/llama.cpp) 的分支。**原项目版权归 ggml-org 及 llama.cpp 全体贡献者所有**，采用 MIT 许可证发布。本分支的全部改动同样以 MIT 许可证发布，并保留原项目的版权声明与许可证全文（见 [LICENSE](LICENSE)）。
