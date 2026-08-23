#!/bin/bash
# tg-fingerprint.sh — TG 输出指纹对拍（llama-cli 固定 seed/prompt 贪婪解码）
#
# 原理：llama-cli 固定 --seed + --temp 0 的 --single-turn 输出在代码无数值变化时
# 应逐字节一致；对 stdout 取 sha256 与基线对拍。数值验收的固化版本（原先手工做，
# 散在 /tmp 易丢）。
#
# 用法：
#   scripts/verify/tg-fingerprint.sh            # 对拍（无基线时提示先 --update）
#   scripts/verify/tg-fingerprint.sh --update   # 生成/覆盖基线 scripts/verify/baselines/
#   额外参数原样透传给 llama-cli（如 -ngl 99）
#
# 覆盖变量：BIN / MODEL / PROMPT / SEED / N_PREDICT / THREADS / TAG
# （TAG 默认取构建目录名，CPU 与 CUDA 构建数值路径不同，基线按 TAG 分开存）
#
# 已知坑（勿踩）：
#   - 必须 --single-turn 且 </dev/null，否则 llama-cli 进交互 TUI 空转
#   - 先 wc -c 检查输出非空再比对：空输出对拍是假阳性（进程可能根本没跑起来）
#   - llama-cli 用 --no-mmap；llama-bench 不认 --no-mmap，要写 --mmap 0
set -euo pipefail
cd "$(dirname "$0")/../.."

BIN=${BIN:-$PWD/build-noncuda/bin/llama-cli}
MODEL=${MODEL:-/media/heiketu/2922DB6548C1F185/DeepSeek-V4-Flash-Q4-mxfp4-0731.gguf}
PROMPT=${PROMPT:-"The quick brown fox jumps over the lazy dog."}
SEED=${SEED:-42}
N_PREDICT=${N_PREDICT:-64}
THREADS=${THREADS:-72}
TAG=${TAG:-$(basename "$(dirname "$BIN")")}

BASELINE_DIR=$PWD/scripts/verify/baselines
BASELINE=$BASELINE_DIR/tg-fingerprint.$TAG.sha256

UPDATE=0
if [ "${1:-}" = "--update" ]; then UPDATE=1; shift; fi

[ -x "$BIN" ] || { echo "llama-cli 不存在: $BIN（BIN=... 覆盖）" >&2; exit 1; }
[ -f "$MODEL" ] || { echo "模型不存在: $MODEL（MODEL=... 覆盖）" >&2; exit 1; }

OUT=$(mktemp)
trap 'rm -f "$OUT"' EXIT

# 与全项目 bench 共用独占锁：大模型加载防并发 OOM；验证请在安静窗口跑
flock -x /tmp/xllama-bench.lock \
  "$BIN" -m "$MODEL" -p "$PROMPT" -n "$N_PREDICT" --seed "$SEED" --temp 0 \
  -t "$THREADS" --single-turn --no-mmap "$@" </dev/null >"$OUT" 2>/dev/null

# 空输出对拍是假阳性：先验字节数（至少 prompt 回显 + 1 token 生成）
BYTES=$(wc -c <"$OUT")
if [ "$BYTES" -le $(( ${#PROMPT} + 1 )) ]; then
  echo "FAIL: 输出仅 $BYTES 字节，生成未发生；stderr 已被丢弃，去掉 2>/dev/null 重跑看日志" >&2
  exit 1
fi

HASH=$(sha256sum "$OUT" | cut -d' ' -f1)

if [ "$UPDATE" = 1 ]; then
  mkdir -p "$BASELINE_DIR"
  echo "$HASH" >"$BASELINE"
  echo "baseline written: $BASELINE"
  echo "  sha256=$HASH bytes=$BYTES seed=$SEED n_predict=$N_PREDICT bin=$BIN"
  exit 0
fi

if [ ! -f "$BASELINE" ]; then
  echo "无基线 $BASELINE；首次运行请先生成：$0 --update" >&2
  exit 2
fi

WANT=$(cat "$BASELINE")
if [ "$HASH" = "$WANT" ]; then
  echo "PASS: sha256=$HASH ($BYTES bytes, baseline $BASELINE)"
else
  echo "FAIL: 输出指纹漂移（数值路径有变化）" >&2
  echo "  want $WANT" >&2
  echo "  got  $HASH" >&2
  echo "  基线: $BASELINE（确认变化符合预期后用 --update 刷新）" >&2
  exit 1
fi
