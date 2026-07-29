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
- **支持格式**：Q2_K、Q3_K、Q4_K、Q5_K、Q6_K、Q8_0、Q4_0、MXFP4、IQ1_S、IQ1_M。其中 x86 AVX512 原生内核 **Q3_K、Q5_K、Q6_K、Q8_0 为本分支新增**（主线仅 NEON/generic），**IQ1_S/IQ1_M 为全套新增**（新块布局 + repack + generic + x86 内核，主线完全没有）；Q2_K、Q4_0、Q4_K、IQ4_NL、MXFP4 内核为主线已有。`gemm_min_nrows=4` 小批路由（Q2_K/Q3_K/Q5_K/IQ1_S/IQ1_M）为本分支新增
- 效果：gemm（PP 批量）单线程大形状 nr≥16 实测 **2.6~4.9×**（Q4_K 4.87、Q6_K 4.85、Q5_K 4.70、Q8_0 3.76、Q4_0 3.67、IQ1_M 3.81、IQ1_S 3.67、MXFP4 3.53、Q3_K 3.02、Q2_K 1.13）；72 线程满核 1.4~3.2×；端到端纯 CPU PP +17.7%/TG -1.8%，GPU 卸载下 <1%（完整 300 格基准表与端到端 A/B 见 [README](../README.md)）；gemv（TG 单 token）受内存带宽封顶 ≈1×（物理上限，非内核问题）
- 路由：Q2_K/Q3_K/Q5_K/IQ1_S/IQ1_M 的 4 行尾块全程向量化，`gemm_min_nrows=4`；其余格式 min_nrows=16（Q4_K/Q6_K 尾块为标量实现，小批刻意走 gemv）
- 工具：`llama-bench` 补 `--no-repack` 开关（映射模型加载 `use_extra_bufts=false`，主线 llama-bench 无此参数），用于 repack 开/关 A/B 对比
- generic 回退路径全格式可用（非 AVX512 机器）

### 2. 图级融合
- **RMS_NORM(+权重乘) 吸收进 mul_mat**：norm 结果直接物化进矩阵乘输入，省一次全量激活读写
- fused QKV 路径
- 调试基建：`GGML_MM_PHASE=1` 逐阶段计时

## 三、CUDA 侧

- `dsv4-hc.cu`：DeepSeek-V4 hyper-connection 融合 kernel（+425 行）
- `mmvq.cu` / `quantize.cu` / `fattn.cu` / `ggml-cuda.cu` 若干优化
- 融合算子框架：`llm_graph_result::add_fused_node`，8 种 fused op（FLASH_ATTN、GDN_AR/CH、LIGHTNING_INDEXER、DSV4_HC_PRE/COMB/POST、DSV4_MOE_ROUTER）
- GPU 专家卸载配方：`-ot "blk\.(层号)\.ffn_(up|gate|down)_exps\.weight=CUDA0/1"` 把指定层专家放上显卡，实测每加一层 TG +1% 左右

## 四、模型支持

- **DeepSeek-V4（deepseek4.cpp）**：284B MoE，含 **MTP 投机解码接线**（`t_h_nextn`/`embd_nextn`/投机验收）、fused hyper-connection、fused MoE router
- **GLM-5.2 DSA（glm-dsa.cpp）**：含 Lightning Indexer 稀疏注意力；MTP 张量已加载待接图
- **MiniMax-M3（minimax-m3.cpp）**：手抄精简版（text-only，M2 GQA + DSv3 式专家）；主线 07-26 已合入官方版，后续 merge 将切换主线实现
- **GGUF 字节级修复工具链**：量化块 f16 scale 腐化（Inf/NaN）扫描与补丁（`nan_fix_backup.json` 可回滚），修复了 GLM-5.2 UD-Q2_K 官方文件 138 个腐化块导致的输出乱码

## 五、跨机分布式 EP（开发中，`tools/epd/`）

- **架构**：激活 dispatch（每 token 每层 KB 级流量），非权重传输——这是跨机场景唯一可行的数据通路（权重流 ~700MB/token vs 激活流 ~1.8MB/token）
- `llama-ep-transport`：帧协议 LEP1 + 传输函数表隔离（TCP 实现已就绪，RDMA verbs 后端预留接口）
- `llama-epd`：专家计算守护进程，gguf 元数据 + 只读 mmap 直接加载（284B 模型 0.3s 启动），单层数值验证 diff=0（逐位一致）
- 目标：双路 8360Y 节点经 InfiniBand EDR 互联，可横向扩展 CPU MoE 节点，跑 2.8T 级模型

## 六、版本控制策略

- `main`（= local）：生产分支 = vendor base + 本地改动层
- `vendor`：主线基线锚点（e8f19cc0a）
- `upstream-master`：主线跟踪引用
- merge 实测：345 个主线提交 trial merge 仅 9 文件冲突，集中在 chat.cpp 等外围文件；性能栈文件（ggml-cpu/repack/llama-graph）merge 时永远保留本分支实现

## 七、实测数据汇总

| 场景 | 配置 | 结果 |
|---|---|---|
| dsv4 284B TG | NUMA mirror | ~33 t/s（主线 5.85） |
| dsv4 284B TG | NUMA-EP（内存减半） | 28.5 t/s |
| dsv4 284B PP | NUMA-EP | 310 t/s（主线 23.6） |
| GLM-5.2 745B TG | EP + GPU 4/5 层专家卸载 | 12.0 t/s（基线 11.06） |
| GLM-5.2 PP | EP | 33.7 t/s（+12.6% vs 无 EP） |
| Q5_K repack gemm nr=16 | 微基准 | ~4× vs legacy |

## 八、关键环境变量速查

| 变量 | 作用 | 默认 |
|---|---|---|
| `GGML_NUMA_EP=1` | 启用机箱内专家并行 | 关 |
| `GGML_NUMA_EP_MMAP=1` | mmap 文件映射的策略级专家放置 | 关 |
| `GGML_NUMA_EP_STEAL_MIN_TOKENS` | phase1 偷取阈值 | 32 |
| `GGML_NUMA_MIRROR_THREADS` | mirror 构建线程数 | — |
| `GGML_MM_PHASE=1` | mul_mat 分阶段计时（调试用） | 关 |
