#!/bin/bash
# P0 GPU 验证（需安静窗口，严禁与生产 llama-server 同时跑）：
# BENCH_GPU_STREAM_REPACK 源模式 16K dense / 16K sparse / 2K dense，对指纹。
# 指纹参照（必须精确命中）：
#   16K dense  sum 4.091932428195e+05 top1 738
#   16K sparse sum 4.081505154114e+05 top1 738
#   2K  dense  sum 2.972226317873e+05 top1 223
# H2 修复（2026-08-23）：WT 原指向 ../llama-gpuprefill worktree（已不存在），改指本仓库根。
# 用法：MODEL_MX=/path/to/model.gguf ./bench-gpuprefill-repack.sh
set -x
LOCK=/tmp/xllama-bench.lock
MODEL_MX=${MODEL_MX:?set MODEL_MX to the DSV4 MXFP4 GGUF path}
WT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
OUT=/tmp/gpuprefill-repack
mkdir -p "$OUT"

run_scope() { # name cmd...
  local name=$1; shift
  flock -x "$LOCK" systemd-run --user --scope --unit="aq-$name" \
    -p ManagedOOMMemoryPressure=kill -p ManagedOOMMemoryPressureLimit=90% \
    --same-dir --collect -q "$@"
}

cd $WT
ENV_BASE="LLAMA_LAYER_MAJOR_DEVICE_HC=2 GGML_CUDA_MOE_PP_MIN_TOKENS=2048 GGML_CUDA_MOE_PP_PREFETCH=3 GGML_CUDA_MOE_PP_EP=1 GGML_CUDA_MOE_PP_EP_MIN_TOKENS=2048 GGML_CUDA_P2P=1 GGML_CUDA_BATCHED_TOPK=1 GGML_CUDA_DSV4_KV_REUSE=1 GGML_CUDA_MOE_PP_DEFER_PREFETCH=1 GGML_CUDA_MOE_PP_PIPE=1"

# 1. 16K dense（对 dense 指纹）
run_scope rp-d16k env $ENV_BASE \
  ./build-cuda/bin/test-layer-major "$MODEL_MX" 16384 16384 99 BENCH_GPU_STREAM_REPACK 72 64 \
  > $OUT/repack-dense-16384.log 2>&1

# 2. 16K sparse（对 sparse 指纹）
run_scope rp-s16k env $ENV_BASE LLAMA_DSV4_SPARSE_FA=1 GGML_CUDA_DSV4_SPARSE_RAW_COMPACT=1 \
  ./build-cuda/bin/test-layer-major "$MODEL_MX" 16384 16384 99 BENCH_GPU_STREAM_REPACK 72 64 \
  > $OUT/repack-sparse-16384.log 2>&1

# 3. 2K dense（对 2K 指纹）
run_scope rp-d2k env $ENV_BASE \
  ./build-cuda/bin/test-layer-major "$MODEL_MX" 2048 2048 99 BENCH_GPU_STREAM_REPACK 72 0 \
  > $OUT/repack-dense-2048.log 2>&1

# 4. 4K dense（设计预期 ~500 tok/s 参考点，无指纹约束）
run_scope rp-d4k env $ENV_BASE \
  ./build-cuda/bin/test-layer-major "$MODEL_MX" 4096 4096 99 BENCH_GPU_STREAM_REPACK 72 0 \
  > $OUT/repack-dense-4096.log 2>&1

echo "=== gpuprefill-repack done ==="
grep -H "sum\|top-1\|tok/s\|t/s" $OUT/repack-*.log | tail -40
