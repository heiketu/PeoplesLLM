# 参与贡献 / Contributing

## 中文版

欢迎提 issue 和 PR。开始前请先了解本项目的特殊性：

- 本项目代码**几乎全部由 AI（Kimi K3 + Kimi Code CLI）编写**，作者只指定技术方向（EP、GEMM、指令集优化、NUMA 优化等）。代码可读性可能不佳，也可能存在潜在 BUG。
- 作者只对**个人平台**（双路 Xeon 8360Y + 2× RTX 3090，Ice Lake AVX512/VNNI/VBMI）上的运行负责。其他平台上的行为不保证。

### 提 issue 时请附上

- 平台信息：CPU 型号与支持的指令集（`lscpu | grep Flags` 或 `/proc/cpuinfo`）、内存容量与通道数、GPU 型号
- 模型与量化格式（如 DeepSeek-V4 Q3_K）
- 完整启动命令与相关环境变量（`GGML_NUMA_EP` 等）
- 关键日志片段（报错前后各几十行）

### 提交 PR 前请验证

- CPU 内核改动（repack/vec_dot 等）：`tests/test-repack-kernels` 必须全过（ALL TESTS PASSED），`test-backend-ops -b CPU -o MUL_MAT,MUL_MAT_ID` 不回归；性能结论需附 `tests/test-repack-kernels --perf [线程数]` 的 A/B 数据
- 改动只覆盖必要范围，风格与周边代码保持一致
- 数字诚实：提升和回退都写进提交说明

### 主线同步

本分支基于 llama.cpp `e8f19cc0a`（2026-07-16），`vendor` 分支保留基线。merge 主线时性能栈文件（ggml-cpu / repack / llama-graph）永远保留本分支实现。

---

## English

Issues and PRs are welcome. Please understand the nature of this project first:

- The code in this fork is **written almost entirely by AI (Kimi K3 + Kimi Code CLI)**; the author only sets the technical direction (EP, GEMM, ISA optimization, NUMA optimization, etc.). Readability may be poor and latent bugs may exist.
- The author takes responsibility only for their **own platform** (dual Xeon 8360Y + 2× RTX 3090, Ice Lake AVX512/VNNI/VBMI). Behavior on other platforms is not guaranteed.

### When filing an issue, please include

- Platform: CPU model and ISA flags (`lscpu | grep Flags` or `/proc/cpuinfo`), RAM capacity/channels, GPU model
- Model and quantization format (e.g. DeepSeek-V4 Q3_K)
- Full launch command and relevant environment variables (`GGML_NUMA_EP`, etc.)
- Key log excerpts (dozens of lines around the failure)

### Before opening a PR

- CPU kernel changes (repack/vec_dot etc.): `tests/test-repack-kernels` must pass (ALL TESTS PASSED), `test-backend-ops -b CPU -o MUL_MAT,MUL_MAT_ID` must not regress; performance claims need A/B data from `tests/test-repack-kernels --perf [threads]`
- Keep changes scoped and match the surrounding code style
- Be honest with numbers: report improvements and regressions alike in the commit message

### Upstream tracking

This fork is based on llama.cpp `e8f19cc0a` (2026-07-16); the `vendor` branch holds the base. When merging upstream, the performance-stack files (ggml-cpu / repack / llama-graph) always keep this fork's implementation.
