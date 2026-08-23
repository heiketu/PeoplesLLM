#!/bin/bash
# ep-selftest.sh — llama-epd --selftest 的 loopback 包装（无需从机）
#
# worker 内建自检：同一层 MoE FFN 本地直算 vs loopback TCP 走完整协议栈的
# 数值一致性检查，退出码即结果（0=一致）。覆盖：协议编解码、ragged 收集、
# repack 内核、融合 gate/up 与 clamp/SWIGLU 路径。
#
# 用法：
#   scripts/verify/ep-selftest.sh [额外 llama-epd 参数...]
#
# 覆盖变量：BIN / MODEL / SELFTEST_LAYER / SELFTEST_TOKENS
set -euo pipefail
cd "$(dirname "$0")/../.."

BIN=${BIN:-$PWD/build-noncuda/bin/llama-epd}
MODEL=${MODEL:-/media/heiketu/2922DB6548C1F185/DeepSeek-V4-Flash-Q4-mxfp4-0731.gguf}
SELFTEST_TOKENS=${SELFTEST_TOKENS:-4}

[ -x "$BIN" ] || { echo "llama-epd 不存在: $BIN（BIN=... 覆盖；需纯 CPU 构建 build-noncuda）" >&2; exit 1; }
[ -f "$MODEL" ] || { echo "模型不存在: $MODEL（MODEL=... 覆盖）" >&2; exit 1; }

ARGS=(--selftest --selftest-tokens "$SELFTEST_TOKENS")
[ -n "${SELFTEST_LAYER:-}" ] && ARGS+=(--selftest-layer "$SELFTEST_LAYER")

# 与全项目 bench 共用独占锁：模型加载防并发 OOM
exec flock -x /tmp/xllama-bench.lock \
  env LD_LIBRARY_PATH="$(dirname "$BIN")" \
  "$BIN" -m "$MODEL" "${ARGS[@]}" "$@"
