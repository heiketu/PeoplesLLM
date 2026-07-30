#!/bin/bash
# GLM-5.2 slave EPD worker。用法: glm-worker.sh [LAYERS] [THREADS] [NUMA] [RDMA]
set -u
LAYERS=${1:-3-17}
T=${2:-70}
NUMA=${3:-weighted}
RDMA=${4:-1}
cd ~/x-llama.cpp/llama-src/build-cpu-rdma/bin
export GGML_REMOTE_EP_DEBUG=1 GGML_EPD_NUMA="$NUMA" GGML_REMOTE_EP_RDMA="$RDMA"
exec flock -x /tmp/xllama-bench.lock ./llama-epd \
  -m ~/x-llama.cpp/glm5.2p/GLM-5.2-UD-Q2_K_MXFP4-00001-of-00007.gguf \
  --layers "$LAYERS" --no-mmap -t "$T" --port 29200
