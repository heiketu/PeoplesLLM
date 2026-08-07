# PeoplesLLM 项目更改与特性说明

本文档完整列出本分支相对主线 llama.cpp 的全部改动。基线：upstream `e8f19cc0a`（2026-07-16，约 b10130），vendor 分支保留该锚点，本地改动以独立 commit 层维护，定期 merge 主线。

---

## 一、NUMA 内存体系（核心护城河）

针对双路（多路）CPU 服务器的内存子系统重构，解决超大 MoE 模型"跨 socket 访问惩罚"问题。

### 1. NUMA mirror（权重镜像）
- 非专家权重 + KV cache 在每个 NUMA 节点各复制一份，线程按节点绑核，全部本地读
- 启动：`--numa mirror --numa-mirror kv`（kv = 仅镜像 KV，省内存）
- 效果：dsv4 284B TG 5.85 → ~33 t/s（5.6×）
- 环境变量：`GGML_NUMA_MIRROR_THREADS`（镜像构建并行度）

### 2. NUMA-EP（机箱内专家并行）
- MoE 专家**单副本**按 socket 切分放置（`owner(e) = e × n_nodes / n_expert`），专家内存减半——这是 745B 级模型能装下的关键
- 计算侧：`mul_mat_id` 双路径（legacy ggml-cpu.c + repack.cpp）phase 0 每节点只算本节点专家（权重 100% 本地读），phase 1 大 batch 时跨节点偷取（阈值 `GGML_NUMA_EP_STEAL_MIN_TOKENS`，默认 32）
- **mbind 策略级放置补丁**（本项目原创）：`ggml_numa_bind_policy` 对 mmap 文件映射做 MPOL_BIND 策略绑定，零页面迁移、无 OOM 风险，`GGML_NUMA_EP_MMAP=1` 启用
- 实测（GLM-5.2 745B）：TG +3~4%，PP +12.6%；dsv4 上 EP 比 mirror 慢 14% 但内存减半——EP 是容量技术，mirror 是速度技术
- 环境变量：`GGML_NUMA_EP=1`、`GGML_NUMA_EP_MMAP=1`

### 3. NUMA 层级 barrier
- 两级 barrier（节点内 arrive → 节点间 release）替代所有线程竞争单 cache line 的 flat barrier，降低 72+ 线程同步开销

## 二、CPU 计算内核（x86 AVX512/VNNI）

### 1. 8×8 interleave repack 内核（`ggml/src/ggml-cpu/arch/x86/repack.cpp` 等）
- 运行时将量化权重重排为 8×8 interleave 布局，配 AVX512F/BW/VNNI（`_mm512_dpbusd_epi32`）gemv/gemm 内核
- **支持格式**：Q2_K、Q3_K、Q4_K、Q5_K、Q6_K、Q8_0、Q4_0、MXFP4、IQ1_S、IQ1_M、IQ2_XXS。其中 x86 AVX512 原生内核 **Q3_K、Q5_K、Q6_K、Q8_0 为本分支新增**（主线仅 NEON/generic），**IQ1_S/IQ1_M 为全套新增**（新块布局 + repack + generic + x86 内核，主线完全没有），**IQ2_XXS repack x8 布局 + AVX512 gemv/gemm 内核为本分支新增**（单行 vec_dot 主线已有 AVX512/AVX2）；Q2_K、Q4_0、Q4_K、IQ4_NL、MXFP4 内核为主线已有。`gemm_min_nrows=4` 小批路由（Q2_K/Q3_K/Q5_K/IQ1_S/IQ1_M/IQ2_XXS）为本分支新增
- 效果：gemm（PP 批量）单线程大形状 nr≥16 实测 **2.6~4.9×**（Q4_K 4.87、Q6_K 4.85、Q5_K 4.70、Q8_0 3.76、Q4_0 3.67、IQ1_M 3.81、IQ1_S 3.67、MXFP4 3.53、Q3_K 3.02、Q2_K 1.13）；72 线程满核 1.4~3.2×；端到端纯 CPU PP +17.7%/TG -1.8%，GPU 卸载下 <1%（完整 300 格基准表与端到端 A/B 见 [README](../README.md)）；gemv（TG 单 token）受内存带宽封顶 ≈1×（物理上限，非内核问题）
- 路由：Q2_K/Q3_K/Q5_K/IQ1_S/IQ1_M/IQ2_XXS 的 4 行尾块全程向量化，`gemm_min_nrows=4`；其余格式 min_nrows=16（Q4_K/Q6_K 尾块为标量实现，小批刻意走 gemv）
- 工具：`llama-bench` 补 `--no-repack` 开关（映射模型加载 `use_extra_bufts=false`，主线 llama-bench 无此参数），用于 repack 开/关 A/B 对比
- **repack 全模型净收益格式依赖（2026-08-07 实测，dsv4 Flash 284B、-ngl 99 -ncmoe 99 EP、q8 KV、72t）**：IQ2_XXS 专家模型（生产 mxfp4 版）repack 开 pp2048=212.4/tg256=20.5，关 168.6/17.5（**关 repack PP −21%/TG −15%，生产必须开**）；Q2_K 模型相反，开 164.1/23.2、关 243.8/23.5（**开 repack PP −33%，Q2_K 应加 --no-repack**）。根因：IQ2_XXS 有本分支 AVX512 repack 内核（快），Q2_K repack gemm 仍是 maddubs 老路径（慢，512 条待 dpbusd 化）
- generic 回退路径全格式可用（非 AVX512 机器）
- **MXFP4/NVFP4 单行 vec_dot VNNI 化（本分支新增）**：`ggml_vec_dot_mxfp4_q8_0`/`ggml_vec_dot_nvfp4_q8_0` 的 AVX2 点积链 maddubs+madd 改为 `_mm256_dpbusd_epi32`（VNNI 编译期分支，非 VNNI 目标保留原链）；与旧实现逐位一致（整数点积数学等价），n=7168 单核 vec_dot 各 −7%

### 2. 图级融合
- **RMS_NORM(+权重乘) 吸收进 mul_mat**：norm 结果直接物化进矩阵乘输入，省一次全量激活读写
- fused QKV 路径
- 调试基建：`GGML_MM_PHASE=1` 逐阶段计时

## 三、CUDA 侧

- `dsv4-hc.cu`：DeepSeek-V4 hyper-connection 融合 kernel（+425 行）
- `mmvq.cu` / `quantize.cu` / `fattn.cu` / `ggml-cuda.cu` 若干优化
- 融合算子框架：`llm_graph_result::add_fused_node`，8 种 fused op（FLASH_ATTN、GDN_AR/CH、LIGHTNING_INDEXER、DSV4_HC_PRE/COMB/POST、DSV4_MOE_ROUTER）
- GPU 专家卸载配方：`-ot "blk\.(层号)\.ffn_(up|gate|down)_exps\.weight=CUDA0/1"` 把指定层专家放上显卡，实测每加一层 TG +1% 左右
- 长 Prompt MoE 流式 Prefill：专家权重保留 pinned 原布局，达到阈值后以完整张量 H2D；可选 1-4 个私有设备槽和独立 H2D/commit stream 跨 split 预取。每槽分配前保留 2GiB 显存，申请失败无损退化到更浅窗口或原串行路径；默认关闭，不改变 Decode 与非 CUDA 后端
- **超长上下文 layer-major prefill（2026-08-04）**：`llama_decode_layer_major()` 层外 token-tile 内执行器，层权重驻留 CUDA slot；16K PP 161→604 tok/s 精确基线（稀疏 raw-KV compact opt-in 752）；真双卡同层 expert-axis EP（`GGML_CUDA_MOE_PP_EP`，2K +63%）；完整有序 K/V tile reuse（+3.9%，logits 逐位一致）；batched top-k（`GGML_CUDA_BATCHED_TOPK`，k=512 21.6×）；q1 FA 16/32-head MMA（fixed TG64 +17%）；raw-SWA 256-cell decode ring（fixed TG512 +7.8%，opt-in 验收中）；MXFP4/Q8 精确 RNE activation 边界；scheduler 分 backend 传输/时间 profile（`GGML_SCHED_PROFILE_INPUTS`）
- **FA decode 结构性修复（2026-08-06）**：① raw-SWA 256-cell decode ring **默认启用**（`LLAMA_DSV4_COMPACT_DECODE_SWA=0` 可关）——layer-major 大 prefill 后把 raw SWA 窗口打包进 `GGML_PAD(2*n_swa,256)` 物理环，q1 decode 图宽与 prompt 长度解耦；多 slot 上下文（`-np>1`，dsv4 raw 强制每序列独立 stream）与 cache 被其他序列占用时自动回退全宽语义，序列清空后 ring 界限自动复位。② q1 单序列 decode 放行 mask-bounds tile 裁剪：`n_kv_masked>=2048` 且整 256 对齐时扫描 mask 上下界，raw SWA 全 -inf tile 整体跳过（16K 场景约 99% 行）；两者均只改变 FA/stream-K 归约分组，语义等价、非 bit-exact
- **DSV4 q1 decode indexed sparse FA + q8 KV 兼容（2026-08-06/07）**：量化 KV（q8_0 等）q1 decode 默认走 compact gather 路径（`LLAMA_DSV4_Q8_SPARSE_FA=1`）：`get_rows` 仅物化选中的 compressed 行（CUDA 新增同类型量化 dst 块拷贝 gather；CPU 原生消费量化行），dense FA 只对 compact concat 物化 F16 scratch；16K 实测 fixed TG64 `9.75→10.69`（+9.6%）、TG512 +1.3%，greedy 512 生成与 dense 无异常分叉。`=0` 回落 dense top-k mask 全扫。CUDA `supported()` 兜底：非 2kv sparse op + 量化 KV 时对**全宽物理 backing** 物化 F16（`K->view_src`），任何组合都留在 GPU MMA，绝不回落 CPU backend（实测 `LLAMA_DSV4_SPARSE_FA=2`+q8 9.48 tok/s ≈ dense 9.66，sched splits 无 FA 落 CPU）。fused indexed sparse_2kv（F16 KV，`LLAMA_DSV4_FUSED_INDEXED_FA`）保持 **opt-in**：16K q1 实测回归（TG64 -12%、TG512 -2%，sparse 仅 ncols2=8 tile vs dense 16/32-head MMA），长上下文或 sparse 宽 head 组后再评估默认化。数值口径：sparse 本非 bit-exact（top1 hash 逐 run 可变，f16 greedy 512 与 dense 完全一致），gather-dequant 顺序对拍 bit-exact（`test-dsv4-gather-dequant`）
- **DSV4 多流（小 q）稀疏 FA（2026-08-06，opt-in）**：多 slot 纯 decode 与混合轮次的等 token 切分头 ubatch（每流 q=1）此前被 `build_csa_lid_attention` 的 `n_tokens==1` batch 级门误伤、静默回退 dense 全扫。新增"每流 q==1"分支：`ubatch.n_seqs_unq == ubatch.n_tokens && !dsv4_ubatch_has_coupled`（coupled 判定从 kv-cache-dsv4.cpp 提升为公开函数；coupled batch 流数塌为 1，保持回退）。P0：fused indexed 门 `LLAMA_DSV4_FUSED_INDEXED_FA` 新增 `=3`（q1 decode + 多流）/`=4`（decode+prefill + 多流），kernel 不变（sparse_idx 按 (流, query) 寻址早已就绪）。P1：q8 compact gather 多流化——`LLAMA_DSV4_Q8_SPARSE_FA=2` / `LLAMA_DSV4_COMPACT_KV=2` 时 top_k/mask 从 `[topk,1,1,ns]` reshape 为 `[topk,1,ns,1]` 走 batched `get_rows`，gather 后每流 concat raw+topk 仍走 dense MMA，fattn kernel 不变。默认值保持旧行为（多流一律 opt-in）；test-backend-ops 新增 q=1 × 2/4/8 流用例，未选中 compressed 行 NaN 毒化验证 gather 不越界、不读错流。GPU A/B（2026-08-07，dsv4-recipeB，双 3090，生产 `-np 8 -c 131072` 配置）：FIXED_TG hash 开/关门完全一致（f16 `4d4adeb84c829521`、q8 `4f86a00d5f62869c`），q8 LOGITS_TRACE 513 行逐位一致；greedy 256 token dense/单流/多流三方 byte-identical。吞吐：短 ctx 4/8 槽 bench-slots 三方均 ±1% 内（噪声级）；~11K ctx/槽 4 槽（q8 dense 202.8s/单流 200.8s/多流 202.8s）与 8 槽（dense 400.3s/多流 397.9s）同样持平——该 ctx 档 decode 为权重带宽主导，CSA KV 扫描占比可忽略，稀疏多流收益按设计预期在 256K+ 长 ctx（`-c` 上限内无法到达），默认 ON 前需安静窗口 1M ctx sweep
- **CUDA 正确性修复**：`offload_op` 拒绝 CPU_REPACK buffer（此前 batch≥32 的 mul_mat_id 把 repack 8×8 布局喂给 MMQ，pp>32 输出垃圾）
- 已实测否决路线：full tensor split（PP -44%）、跨 tile 全图双 scheduler（CPU backend threadpool 跨 graph arena 语义冲突，基线 208→25-123 tok/s）

## 四、模型支持

- **DeepSeek-V4（deepseek4.cpp）**：284B MoE，含 **MTP 投机解码接线**（`t_h_nextn`/`embd_nextn`/投机验收）、fused hyper-connection、fused MoE router
- **GLM-5.2 DSA（glm-dsa.cpp）**：含 Lightning Indexer 稀疏注意力；MTP 张量已加载待接图
- **MiniMax-M3（minimax-m3.cpp）**：手抄精简版（text-only，M2 GQA + DSv3 式专家）；主线 07-26 已合入官方版，后续 merge 将切换主线实现
- **GGUF 字节级修复工具链**：量化块 f16 scale 腐化（Inf/NaN）扫描与补丁（`nan_fix_backup.json` 可回滚），修复了 GLM-5.2 UD-Q2_K 官方文件 138 个腐化块导致的输出乱码

## 五、跨机分布式 EP（可用，`tools/epd/`）

- **架构**：master 发送激活、router ids 和路由权重，worker 读取本地专家权重计算；网络传激活而不是传模型权重
- `llama-ep-transport`：LEP1/协议 v2，支持 TCP 和 RoCEv2 RDMA；RDMA 建连失败自动回退 TCP
- `llama-epd`：支持多分卷 GGUF、mmap/`--no-mmap`、持久线程池、运行时 repack、NUMA 加权交织和启动 autotune
- classic 模式按层分片；mirror 模式把远端专家槽与本地槽并行计算；SCHED 模式通过 REQ2/RESP2 做专家槽级派单
- DSV4 与 GLM-5.2 已完成逐字输出对拍和双机实测；生产参数与当前限制见 `docs/PEOPLESLLM-PARAMS.md`

## 六、版本控制策略

- `local`：本地主集成分支；验证后的快照再发布到 `main`
- `vendor`：主线基线锚点（e8f19cc0a）
- `upstream-master`：主线跟踪引用
- 功能实验通过 Git worktree 隔离；`local` 保持可运行集成状态，成熟实验再按主题合入
- 跟进上游时逐文件解决冲突并重新跑正确性/性能验收，不默认整文件保留任一侧实现

## 七、实测数据汇总

> 注：dsv4 各项速度均未启用 MTP 投机解码。

| 场景 | 配置 | 结果 |
|---|---|---|
| dsv4 284B TG | NUMA mirror | ~33 t/s（主线 5.85） |
| dsv4 284B TG | NUMA-EP（内存减半） | 28.5 t/s |
| dsv4 284B PP | NUMA-EP | 310 t/s（主线 23.6） |
| GLM-5.2 745B TG | EP + GPU 4/5 层专家卸载 | 12.0 t/s（基线 11.06） |
| GLM-5.2 PP | EP | 33.7 t/s（+12.6% vs 无 EP） |
| Q5_K repack gemm nr=16 | 微基准 | ~4× vs legacy |

### NUMA 带宽矩阵（membw2，2026-08-05，双路 Xeon 8360Y，DDR4-3200 8ch×2）

| 访问模式 | 线程 | read GB/s | write GB/s | triad GB/s |
|---|---|---|---|---|
| local0（本路） | 76 | 136.4 | 92.4 | 120.9 |
| local1（本路） | 76 | 142.8 | 96.8 | 125.8 |
| cross01（跨 UPI） | 76 | 54.6 | 34.3 | 71.2 |
| cross10（跨 UPI） | 76 | 53.7 | 33.0 | 70.1 |
| interleave（主线 distribute 模式） | 152 | 177.6 | 85.4 | 164.6 |
| EP/mirror 模式（两路本地合计） | 152 | 279.2 | 189.2 | 246.7 |
| 单线程 local / cross | 1 | 9.2 / 6.9 | 10.3 / 7.6 | 14.6 / 11.8 |

跨路读 = 本地 ~38%；EP/mirror 本地读模式比主线 interleave 模式有效读带宽 +57%。perf 在 Ice Lake 无 UPI uncore PMU，跨路数字即 UPI 有效带宽口径。

### 纯 CPU vs 主线（DSV4 284B Q3_K，-ngl 0，72 线程 interleave，2026-08-05 负载环境）

| 实现 | pp512 | tg512 | tg64 |
|---|---|---|---|
| upstream b10173 | 73.83 | 10.29 | 11.03 |
| PeoplesLLM | 101.90 | 12.19 | 12.95 |
| 提升 | **+38%** | **+18%** | **+17%** |

注：本轮 upstream 因 `--mmap 0` deprecated 实际 mmap 运行（对主线略不利），下一轮用 `--load-mode` 对齐后在安静窗口复测定稿。

## 八、关键环境变量速查

| 变量 | 作用 | 默认 |
|---|---|---|
| `GGML_NUMA_EP=1` | 启用机箱内专家并行 | 关 |
| `GGML_NUMA_EP_MMAP=1` | mmap 文件映射的策略级专家放置 | 关 |
| `GGML_NUMA_EP_STEAL_MIN_TOKENS` | phase1 偷取阈值 | 32 |
| `GGML_NUMA_MIRROR_THREADS` | mirror 构建线程数 | — |
| `GGML_MM_PHASE=1` | mul_mat 分阶段计时（调试用） | 关 |
| `LLAMA_DSV4_COMPACT_DECODE_SWA=0` | 关闭 raw-SWA 256-cell decode ring，恢复全宽 raw KV | 开 |
