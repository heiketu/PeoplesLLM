# Weight-stationary 超长 Prefill 设计

## 目标与边界

目标是在保留 `llama_decode` 和所有现有模型行为的前提下，为接近 1M token 的单请求
Prompt 增加可选的 layer-major 执行路径。当前 token-major 路径每个 ubatch 都遍历全部层；
host MoE 权重因此被重复 H2D。新路径让当前层权重驻留，按安全 token tile 扫完整 Prompt，
再进入下一层。

它不是单次 1M-token CUDA kernel，也不改变 Decode。任一能力或资源检查失败时，在写入 KV
前回退到现有 chunked Prefill。

## 已落地接口

`llama_decode` 当前断言单次输入不超过 `n_batch`，应用通常把长 Prompt 拆成多次调用。
执行器因此看不到后续 token，不能把调用顺序从 token-major 改成 layer-major。兼容方案是新增
显式实验接口，由已经持有完整 token 列表的调用方提交一个 sealed batch：

```c
int32_t llama_decode_layer_major(
        llama_context * ctx,
          llama_batch   batch,
               uint32_t tile_tokens);
```

该调用不受 context `n_batch` 限制，内部仍按不超过 `n_ubatch` 的 tile 运行。调用方负责在调用
前收集完整 Prompt；这样不引入跨调用 session 生命周期，也不改变任何现有公开 API 的语义。
接口位于实验性的 `llama-ext.h`。返回 `-1` 表示不满足资格，调用方可继续走原有 chunked
`llama_decode`；默认路径和 server 自动选择均未改变。

首版资格被刻意限制为 DSV4、初始空 sequence、单 sequence、从零连续 position、causal、
token 输入、只请求末 token logits、无 embedding/pooling/backend sampler/LoRA。完整 HC host
buffer 会在写 cache 前分配并触页；运行期失败会同步设备并删除该 sequence 的 raw/compressed
KV 和 compressor state，避免暴露半成品 cache。

## DSV4 最小执行图

DeepSeek-V4 先作为 opt-in 验证架构。其层间状态是 F32
`[n_embd, hc=4, n_tokens]`，当前模型每 token 16,384 个 float。1M token 单缓冲约 64GiB。
主机只保留一个可原地更新的大缓冲：同一层 tile 的输出写回已不再需要的输入区间。
当前实现按 tile 在该普通 host buffer 与设备间异步传输，并在层边界同步；后续可增加两个
小型 pinned staging tile 做 H2D/D2H 重叠，但不能锁页整个 64GiB。

CUDA MoE 预取槽会记录已上传的 immutable host tensor 及实际字节数。同一层的后续 token
tile 复用槽内权重，不再重复 H2D；scheduler-owned 目标仍逐 tile 执行 D2D commit，因为其
地址会被计算图复用，不能仅凭地址判定内容仍有效。

图参数增加可选的 `[layer_begin, layer_end)` 范围：

- `layer_begin == 0`：输入仍是 token/embedding，建立初始 hyper-connection state；
- `layer_begin > 0`：输入使用现有 `llm_graph_input_embd_h` 的 F32 HC state；
- `layer_end < n_layer`：导出 `l_out`，不构建 output head/logits；
- `layer_end == n_layer`：构建现有 head，仅请求所需输出行。

每个 tile 仍按位置升序执行当前层注意力并写最终 KV。下一层必须等待当前层全部 tile
完成。DSV4 memory context 会重放首次生成的 slot/split/compression 计划，并只让当前层图
写入其 KV；不会重新分配 token 位置。

## 内存与传输预算

- 模型实测 RSS：约 145GiB；
- DSV4 F32 HC state：约 64GiB / 1M token；
- host compute 静态估算：约 4.3GiB（ub4096）；
- 合计约 214GiB，251GiB 主机剩余不足以再做完整双缓冲或 NUMA mirror；
- staging 只允许 tile 级 pinned buffer；大状态用普通 NUMA 本地匿名内存；
- GPU 始终执行 1K/2K/4K tile，并遵守每卡 2GiB 保留线。

在当前 token-major ub4096 测量中，一次 ubatch 的完整 MoE 权重 H2D 约 147GB。1M Prompt
约 256 个 ubatch。Layer-major 理论上把每个专家层权重从“每 ubatch 一次”降为“一次”，
代价变为每层读写 64GiB HC state。是否净胜必须用端到端 nsys 验证，不能只用流量估算发布。

## 执行与失败语义

1. 校验完整 batch 和上下文资格，此阶段不改 KV，可安全回退。
2. 分配完整 HC state，生成一次 DSV4 slot/split/compression 计划。
3. 外层遍历 layer、内层重放相同 tile 计划；默认在每层边界同步 HC D2H/H2D。显式启用 `LLAMA_LAYER_MAJOR_DEVICE_HC=1` 时，显存预算允许才把完整 HC state 保存在 GPU，并以 D2D/P2P 跨层传递。
4. 最后一层发布末 token logits；所有层都已写好 Decode 可直接使用的 KV。
5. 任一计算失败时同步设备并清除该 sequence；绝不把半完成 KV 交给 Decode。

首版仅支持单 sequence、causal attention、无 LoRA 动态切换、无跨请求 continuous batching。
这些条件不满足时回退，不影响 llama.cpp 的广泛兼容性。

## 验收门槛

1. 先以 8K/32K Prompt 验证逐 token top-1 和最终 KV 后续 Decode 一致；
2. temp=0 固定 seed，通用路径与新路径生成至少 64 token 对拍；
3. 128K/256K/1M 阶梯记录 RAM、swap、每卡 VRAM 和失败清理；
4. 任何层中止后，同一 context 必须还能用通用路径重新 Prefill；
5. 端到端 Prefill 必须显著快于已完成的多槽 token-major 路径，否则不进入默认自动选择；
6. 非 DSV4 模型、CPU-only 和非 CUDA 构建的测试结果必须完全不变。

## 当前验证状态

- 非 CUDA AVX2 全量构建通过；layer-major、架构兼容和 repack 定向测试通过。
- 所有非 DSV4 架构都会被实验接口拒绝，随后原 `llama_decode` 测试继续通过。
- 真实 43 层 DSV4 GGUF 已完成 4-token / tile=2 的逐层 CUDA 对拍：Prefill logits
  `max_abs=0`，随后一 token Decode 的 logits 仍为 `max_abs=0`，验证最终 KV 可用。
- 双 RTX 3090、Q3_K_M 88.09GiB、16,384 token、tile=4096、44 GPU layer 的相同条件下，
  从 `101.616s / 161.234 tok/s` 提升到 `93.989s / 174.318 tok/s`；吞吐提升 8.1%，
  Prefill 和下一 token KV logits 均为 `max_abs=0`。
- 当前 profiling 的下一优先级是 Lightning Indexer 的逐行 CUB top-k 启动风暴和 Flash
  Attention；完整 segmented sort 替代方案会产生 NaN，已拒绝且未保留。
- 8K/32K、128K/256K/1M 阶梯性能和故障注入仍是发布/自动选择门槛；在这些数据完成前，
  该路径保持显式 opt-in，不接管 server 或通用 `llama_decode`。
- device HC 的 16K/tile4096 两次实测为 60.857/60.903s，平均 269.12 tok/s；与 host HC
  logits 逐字一致。Nsight 中 42GiB D2H 和 84GiB HC H2D 消失，替换为 64GiB D2D 和
  62GiB P2P。该路径额外占用约 1GiB/16K GPU 显存，并在分配前保留至少 4GiB 或 20%。
