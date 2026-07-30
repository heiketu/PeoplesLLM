#!/bin/bash
# GLM-5.2 master server 启动（build-epdev-rdma）。用法: glm-master.sh single|dual|mirror [LAYERS]
set -u
PROJ=/home/heiketu/x-llama.cpp/llama-src
MODEL=/models/glm5.2p/GLM-5.2-UD-Q2_K_MXFP4-00001-of-00007.gguf
MODE=${1:?usage: glm-master.sh single|dual|mirror [LAYERS]}
LAYERS=${2:-3-17}
BUILD=${BUILD:-build-epdev-rdma}
cd "$PROJ"
export LD_LIBRARY_PATH="$PROJ/$BUILD/bin"
export GGML_NUMA_EP=1 GGML_NUMA_EP_MMAP=1
if [ "$MODE" != single ]; then
  export GGML_REMOTE_EP=1 GGML_REMOTE_EP_HOST=10.0.0.2 GGML_REMOTE_EP_PORT=29200
  export GGML_REMOTE_EP_LAYERS="$LAYERS" GGML_REMOTE_EP_RDMA="${RDMA:-1}" GGML_REMOTE_EP_DEBUG=1
fi
if [ "$MODE" = mirror ]; then
  export GGML_REMOTE_EP_MIRROR=1
fi
if [ "${PIPELINE:-0}" = 1 ]; then
  export GGML_REMOTE_EP_PIPELINE=1
fi
exec flock -x /tmp/xllama-bench.lock numactl --interleave=all \
  ./$BUILD/bin/llama-server \
  -m "$MODEL" \
  -ngl 99 -ncmoe 99 -t 70 --threads-batch 70 -fa 1 -b 1024 -ub 512 \
  -ot "blk\.(29|3[0-2])\.ffn_(up|gate|down)_exps\.weight=CUDA0,blk\.(5[89]|6[0-2])\.ffn_(up|gate|down)_exps\.weight=CUDA1" \
  -c 8192 -np 1 --host 127.0.0.1 --port 18121 \
  --no-mmap --numa mirror --numa-mirror kv \
  --jinja --reasoning-format deepseek
