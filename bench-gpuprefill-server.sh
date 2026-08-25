#!/bin/bash
# P1 server 集成验证（需安静窗口，严禁与生产 llama-server 同时跑）：
# 资格门路由（合格/不合格请求）、PP 提升、并发 slot TG 抖动、门限扫参。
# 用法: MODEL_MX=/path/to/model.gguf ./bench-gpuprefill-server.sh [MIN_TOKENS=4096]
# H2 修复（2026-08-23）：WT 原指向 ../llama-gpuprefill worktree（已不存在），改指本仓库根。
set -x
LOCK=/tmp/xllama-bench.lock
MODEL_MX=${MODEL_MX:?set MODEL_MX to the DSV4 MXFP4 GGUF path}
WT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
OUT=/tmp/gpuprefill-server
MIN_TOKENS=${1:-4096}
PORT=8090
mkdir -p "$OUT"

ENV_BASE="GGML_NUMA_EP=1 LLAMA_LAYER_MAJOR_DEVICE_HC=2 GGML_CUDA_MOE_PP_MIN_TOKENS=2048 GGML_CUDA_MOE_PP_PREFETCH=3 GGML_CUDA_MOE_PP_EP=1 GGML_CUDA_MOE_PP_EP_MIN_TOKENS=2048 GGML_CUDA_P2P=1 GGML_CUDA_BATCHED_TOPK=1 GGML_CUDA_DSV4_KV_REUSE=1 GGML_CUDA_MOE_PP_DEFER_PREFETCH=1 GGML_CUDA_MOE_PP_PIPE=1 LLAMA_GPU_PREFILL_MIN_TOKENS=$MIN_TOKENS"

# 生产对齐形态（缩小 ctx 以便窗口内跑；q8 KV + 0 pin + EP 全部保持）
flock -x "$LOCK" systemd-run --user --scope --unit="aq-gpuprefill-server" \
  -p ManagedOOMMemoryPressure=kill -p ManagedOOMMemoryPressureLimit=90% \
  --same-dir --collect -q \
  env $ENV_BASE ./build-cuda/bin/llama-server -m "$MODEL_MX" \
    -c 1048576 -np 2 -ngl 99 -ncmoe 99 -b 8192 -ub 1024 -fa 1 -t 72 \
    -ctk q8_0 -ctv q8_0 --numa distribute --no-mmap --port $PORT \
  > $OUT/server.log 2>&1 &

cd $WT
for i in $(seq 1 120); do
  curl -s -o /dev/null http://127.0.0.1:$PORT/health && break
  sleep 5
done

# 长 prompt（~16K token）：合格请求，应路由到 GPU 流式 prefill
python3 - <<'EOF' > $OUT/long_prompt.txt
print("DeepSeek V4 prefill verification. " * 4000)
EOF

echo "=== 1. eligible 16K prompt (expect 'trying GPU streaming prefill') ==="
curl -s http://127.0.0.1:$PORT/completion -H 'Content-Type: application/json' \
  -d "{\"prompt\": \"$(cat $OUT/long_prompt.txt | head -c 60000)\", \"n_predict\": 8, \"temperature\": 0}" \
  > $OUT/resp-16k.json

echo "=== 2. ineligible 1K prompt (expect NO streaming prefill line) ==="
curl -s http://127.0.0.1:$PORT/completion -H 'Content-Type: application/json' \
  -d '{"prompt": "Write a short story about a robot learning to paint.", "n_predict": 8, "temperature": 0}' \
  > $OUT/resp-1k.json

echo "=== 3. repeat 16K prompt with cache_prompt (LCP hit -> ineligible chunked) ==="
curl -s http://127.0.0.1:$PORT/completion -H 'Content-Type: application/json' \
  -d "{\"prompt\": \"$(cat $OUT/long_prompt.txt | head -c 60000)\", \"n_predict\": 8, \"temperature\": 0, \"cache_prompt\": true}" \
  > $OUT/resp-16k-cache.json

echo "=== 4. concurrent: slot A 16K prefill while slot B decodes (TG jitter) ==="
curl -s http://127.0.0.1:$PORT/completion -H 'Content-Type: application/json' \
  -d '{"prompt": "Count from 1 to 200, separated by commas.", "n_predict": 400, "temperature": 0}' \
  > $OUT/resp-tg.json &
TG_PID=$!
sleep 2
curl -s http://127.0.0.1:$PORT/completion -H 'Content-Type: application/json' \
  -d "{\"prompt\": \"$(cat $OUT/long_prompt.txt | head -c 60000)\", \"n_predict\": 4, \"temperature\": 0}" \
  > $OUT/resp-16k-conc.json
wait $TG_PID

echo "=== 5. routing + timings ==="
grep -c "trying GPU streaming prefill" $OUT/server.log
grep "GPU streaming prefill" $OUT/server.log | tail -10
for f in $OUT/resp-*.json; do
  echo "--- $f"; python3 -c "import json,sys; d=json.load(open('$f')); print(d.get('timings', d))"
done

echo "=== done; server left running in scope aq-gpuprefill-server ==="
echo "stop with: systemctl --user stop aq-gpuprefill-server.scope"
