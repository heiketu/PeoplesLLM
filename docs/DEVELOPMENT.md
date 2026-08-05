# PeoplesLLM 开发与验证

本文只描述可复现的工程流程。功能参数、生产配方和经过复核的性能结论分别以
`docs/PEOPLESLLM-PARAMS.md` 和 `README.md` 为准。

## 仓库与分支

- `llama-src` / `local`：主集成工作树。
- `llama-upstream`：只读上游基线工作树。
- `llama-gpu-pp`、`llama-gpu-tp`、`llama-iq-traits`：独立实验 worktree。
- `vendor`：本地改动所基于的 llama.cpp 基线。

不要在多个 worktree 同时修改同一功能。合入前先确认：

```sh
git status --short --branch
git diff --check
git diff --cached --check
```

同时存在 staged 和 unstaged 修改的文件必须分别检查，避免提交半套改动。

## 干净 CPU 构建

不要复用来自其他目录或旧挂载点的 `CMakeCache.txt`。新建独立构建目录：

```sh
cmake -S . -B build-peoples-cpu \
    -DCMAKE_BUILD_TYPE=Release \
    -DGGML_CUDA=OFF \
    -DLLAMA_BUILD_TESTS=ON \
    -DLLAMA_BUILD_TOOLS=ON \
    -DLLAMA_BUILD_SERVER=ON
cmake --build build-peoples-cpu -j --target \
    llama-cli llama-bench llama-server llama-epd \
    test-chat test-backend-ops test-repack-kernels llama-ep-dealer-test
```

`test-chat` 使用 server 的请求转换代码，因此只在 `LLAMA_BUILD_SERVER=ON` 时生成。

## CUDA 构建

RTX 3090 使用计算能力 8.6：

```sh
cmake -S . -B build-peoples-cuda \
    -DCMAKE_BUILD_TYPE=Release \
    -DGGML_CUDA=ON \
    -DCMAKE_CUDA_ARCHITECTURES=86 \
    -DLLAMA_BUILD_TESTS=ON \
    -DLLAMA_BUILD_TOOLS=ON \
    -DLLAMA_BUILD_SERVER=ON
cmake --build build-peoples-cuda -j
```

测速前记录 `git rev-parse HEAD`、`git status --short`、完整 CMake 配置和运行命令。
不要把不同 commit 的二进制放进同一构建目录。

## 最小验证集

```sh
build-peoples-cpu/bin/llama-ep-dealer-test
build-peoples-cpu/bin/test-chat
build-peoples-cpu/bin/test-repack-kernels
```

CPU-only 构建直接运行 `test-backend-ops` 会跳过作为参考实现的 CPU backend，不能把它的
零退出状态当成算子正确性通过。完整 backend 对拍需要 CUDA 等第二个 backend；`-b CPU`
只适合定向调试 CPU 用例，不作为全套验收口径。

涉及 CUDA、NUMA、repack 或 remote EP 的修改还必须执行：

1. 对应 backend-op / selftest。
2. 固定 seed、temperature=0 的非空输出对拍。
3. ABBA 顺序性能复测。
4. 记录模型、量化格式、线程、NUMA、batch/ubatch 和 GPU offload 配置。

大模型进程继续使用项目约定的独占 benchmark 锁，避免并发加载造成 OOM 或污染结果。

## 文档职责

- `README.md`：项目定位和经过复核的代表性结果。
- `docs/CHANGES.md`：已合入能力，不记录未实现设想。
- `docs/PEOPLESLLM-PARAMS.md`：当前代码实际支持的参数和生产配方。
- `docs/LONG-CONTEXT-1M.md`：1M context 的 RAM/VRAM 硬预算与 release gate。
- `tools/epd/*DESIGN.md`：设计与实验路线，必须标注 implemented / experimental / rejected。
- 工作区 `HANDOVER.md`：本地研发日志和证据索引，不属于稳定用户接口。
