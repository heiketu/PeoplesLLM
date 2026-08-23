#!/bin/bash
# PeoplesLLM 一键 profile 启动器 —— 把几十上百个参数收成一条命令。
#
# 用法:
#   tools/peoplesllm-run.sh <profile> [额外 llama-server 参数...]
#
# profile:
#   dsv4-prod       单机混合生产:双 GPU + 双路 NUMA EP + dspark,4 slot x 1M,Q8 KV(默认)
#   dsv4-prod-f16   同上但 F16 KV 单槽 1M(质量优先档)
#   dsv4-dual       双机 DSV4 EP 生产(classic 分层,需 slave 在线; slave 默认认领 36-42)
#   glm-dual        双机 GLM-5.2 EP 生产(需 slave 在线; slave 默认认领 3-17)
#   cpu-pure        纯 CPU 生产(build-noncuda; CUDA_VISIBLE_DEVICES="" 是铁律)
#   bench-hybrid    混合推理基准(llama-bench tg512/pp2048)
#
# 环境变量覆盖: MODEL / MODEL_DRAFT / MODEL_GLM / PORT / SLOTS / CTX_TOTAL
#               REMOTE_HOST / REMOTE_PORT / REMOTE_LAYERS / RDMA
set -u
cd "$(dirname "$0")/.."
CBIN=$PWD/build-cuda/bin
WBIN=$PWD/build-noncuda/bin

MODEL=${MODEL:-/media/heiketu/2922DB6548C1F185/DeepSeek-V4-Flash-Q4-mxfp4-0731.gguf}
MODEL_DRAFT=${MODEL_DRAFT:-/media/heiketu/2922DB6548C1F185/dspark-DeepSeek-V4-Flash-MXFP4.gguf}
MODEL_GLM=${MODEL_GLM:-/models/glm5.2p/GLM-5.2-UD-Q2_K_MXFP4-00001-of-00007.gguf}
PORT=${PORT:-18108}
SLOTS=${SLOTS:-4}
CTX_TOTAL=${CTX_TOTAL:-4194304}   # SLOTS x 1M
REMOTE_HOST=${REMOTE_HOST:-10.0.0.2}
REMOTE_PORT=${REMOTE_PORT:-29200}
RDMA=${RDMA:-1}

# 以下旋钮自 2026-08 起已默认开启(代码内 default ON),此处无需再设:
#   GGML_REPACK_GEMV_PREFETCH / GGML_NUMA_HIER_BARRIER / GGML_CUDA_BATCHED_TOPK / GGML_CUDA_DSV4_KV_REUSE
# 如需关闭: 显式 export XXX=0

worker_up() { # 等待 worker 监听
    for i in $(seq 1 200); do timeout 2 bash -c "</dev/tcp/$1/$2" 2>/dev/null && return 0; sleep 3; done
    return 1
}

case "${1:-}" in
  dsv4-prod|dsv4-prod-f16)
    [ -f "$MODEL" ] || { echo "模型不存在: $MODEL (用 MODEL=... 覆盖)"; exit 1; }
    pkill -9 -x llama-server 2>/dev/null; pkill -x llama-epd 2>/dev/null
    for i in $(seq 1 90); do pgrep -x llama-epd >/dev/null || break; sleep 2; done
    pgrep -x llama-epd >/dev/null && { echo "旧 worker 未退出"; exit 1; }

    setsid bash -c "cd $WBIN && export LD_LIBRARY_PATH=\$PWD GGML_NUMA_EP=1 && \
      exec $WBIN/llama-epd -m $MODEL --port 29202 --layers 0-42 --experts 0-255 -t 72 --no-autotune --no-mmap" \
      </dev/null >/tmp/peoplesllm-worker.log 2>&1 9>&- &
    worker_up 127.0.0.1 29202 || { echo "worker 启动失败, 见 /tmp/peoplesllm-worker.log"; exit 1; }
    echo "worker up (127.0.0.1:29202)"

    KV_ARGS="-c $CTX_TOTAL -np $SLOTS -ctk q8_0 -ctv q8_0"
    [ "${1}" = "dsv4-prod-f16" ] && KV_ARGS="-c 1048576 -np 1"
    exec systemd-run --user --scope --unit=peoplesllm-prod --same-dir --collect -q \
      env LD_LIBRARY_PATH=$CBIN GGML_REMOTE_EP=1 GGML_REMOTE_EP_LAYERS=0-42 GGML_REMOTE_EP_SCHED=1 \
      GGML_REMOTE_EP_SCHED_KLOCAL=0 GGML_REMOTE_EP_SCHED_MAX_EFFORT=1 GGML_REMOTE_EP_SCHED_PP=1 \
      GGML_REMOTE_EP_SCHED_ENDPOINTS=127.0.0.1:29202 \
      GGML_REMOTE_EP_RDMA=1 GGML_EP_RDMA_SPIN=1 GGML_REMOTE_EP_PIPE=0 GGML_REMOTE_EP_PARALLEL_IO=1 \
      GGML_REMOTE_EP_WEIGHT_ON_MASTER=1 GGML_REMOTE_EP_RECONNECT_TIMEOUT_MS=90000 \
      GGML_REMOTE_EP_SCHED_TG_REPEAT_COST=250 GGML_REMOTE_EP_SCHED_PP_REPEAT_COST=250 \
      $CBIN/llama-server -m "$MODEL" -md "$MODEL_DRAFT" --spec-type draft-dspark \
      --spec-draft-n-max 2 --spec-draft-p-min 0 \
      -dev CUDA0,CUDA1 -sm layer -ts 1,0 -ot "^token_embd\\.weight$=CUDA1,^output.*=CUDA1" \
      --spec-draft-device CUDA1 -ngl all -ncmoe 99 -ngld all -t 72 -tb 72 -b 4096 -ub 1024 \
      $KV_ARGS -fa on -fit off --no-kv-unified --numa distribute \
      --no-warmup --no-ui --host 0.0.0.0 --port "$PORT" "${@:2}"
    ;;

  dsv4-dual|glm-dual)
    # 双机 EP(master 侧): 不起本地 worker, 认领层发给 slave 的 llama-epd;
    # slave 需先在线(见下方打印的启动范例)。机制与坑见 docs/QUICKSTART.md §3
    DUAL_MODEL=$MODEL; DUAL_LAYERS=36-42   # DSV4(43 层 MoE)实测 slave 认领 7-8 层最优
    if [ "${1}" = "glm-dual" ]; then
      DUAL_MODEL=$MODEL_GLM; DUAL_LAYERS=3-17   # GLM-5.2: slave 15 层(43.5G)是内存上限
    fi
    DUAL_LAYERS=${REMOTE_LAYERS:-$DUAL_LAYERS}
    [ -f "$DUAL_MODEL" ] || { echo "模型不存在: $DUAL_MODEL (MODEL=/MODEL_GLM=... 覆盖)"; exit 1; }
    echo "等待 slave worker $REMOTE_HOST:$REMOTE_PORT (layers $DUAL_LAYERS)..."
    echo "  slave 启动范例(build-noncuda): GGML_EPD_NUMA=weighted \\"
    echo "    llama-epd -m <同内容gguf> --port $REMOTE_PORT --layers $DUAL_LAYERS -t 72 --no-mmap"
    echo "  有 RoCE 时 slave 同设 GGML_REMOTE_EP_RDMA=1 (master 侧 RDMA=$RDMA, RDMA=0 覆盖)"
    worker_up "$REMOTE_HOST" "$REMOTE_PORT" || { echo "slave worker 不在线, 先启动 slave"; exit 1; }

    if [ "${1}" = "dsv4-dual" ]; then
      KV_ARGS="-c $CTX_TOTAL -np $SLOTS -ctk q8_0 -ctv q8_0"
      exec systemd-run --user --scope --unit=peoplesllm-dsv4-dual --same-dir --collect -q \
        env LD_LIBRARY_PATH=$CBIN \
        GGML_REMOTE_EP=1 GGML_REMOTE_EP_HOST=$REMOTE_HOST GGML_REMOTE_EP_PORT=$REMOTE_PORT \
        GGML_REMOTE_EP_LAYERS=$DUAL_LAYERS GGML_REMOTE_EP_RDMA=$RDMA \
        GGML_NUMA_EP=1 \
        $CBIN/llama-server -m "$DUAL_MODEL" -md "$MODEL_DRAFT" --spec-type draft-dspark \
        --spec-draft-n-max 2 --spec-draft-p-min 0 \
        -dev CUDA0,CUDA1 -sm layer -ts 1,0 -ot "^token_embd\\.weight$=CUDA1,^output.*=CUDA1" \
        --spec-draft-device CUDA1 -ngl all -ncmoe 99 -ngld all -t 72 -tb 72 -b 4096 -ub 1024 \
        $KV_ARGS -fa on -fit off --no-kv-unified --numa distribute --no-mmap \
        --no-warmup --no-ui --host 0.0.0.0 --port "$PORT" "${@:2}"
    else
      # GLM-5.2: 对齐 tools/epd/bench-glm-master.sh 的 dual 模式(去掉了与 --no-mmap
      # 矛盾的 GGML_NUMA_EP_MMAP=1, 见 PARAMETERS §8); GPU 钉几层专家是实测配方
      exec systemd-run --user --scope --unit=peoplesllm-glm-dual --same-dir --collect -q \
        env LD_LIBRARY_PATH=$CBIN \
        GGML_REMOTE_EP=1 GGML_REMOTE_EP_HOST=$REMOTE_HOST GGML_REMOTE_EP_PORT=$REMOTE_PORT \
        GGML_REMOTE_EP_LAYERS=$DUAL_LAYERS GGML_REMOTE_EP_RDMA=$RDMA \
        GGML_NUMA_EP=1 \
        $CBIN/llama-server -m "$DUAL_MODEL" \
        -ngl 99 -ncmoe 99 -t 70 -tb 70 -fa 1 -b 1024 -ub 512 \
        -ot "blk\.(29|3[0-2])\.ffn_(up|gate|down)_exps\.weight=CUDA0,blk\.(5[89]|6[0-2])\.ffn_(up|gate|down)_exps\.weight=CUDA1" \
        -c ${CTX_GLM:-8192} -np 1 --no-mmap --numa mirror --numa-mirror kv \
        --jinja --reasoning-format deepseek \
        --no-warmup --no-ui --host 0.0.0.0 --port "${PORT_GLM:-18121}" "${@:2}"
    fi
    ;;

  cpu-pure)
    [ -f "$MODEL" ] || { echo "模型不存在: $MODEL (MODEL=... 覆盖)"; exit 1; }
    # CUDA_VISIBLE_DEVICES="" 是铁律(注意是 ="" 不是 =): CUDA 设备可见 + -ngl 0 时
    # DSA 稀疏注意力会被误分派到 CUDA 而 layer 在 CPU, 整段退化 ~2x (QUICKSTART §4)。
    # GGML_NUMA_HIER_BARRIER 已默认开, 无需再设
    exec systemd-run --user --scope --unit=peoplesllm-cpu-pure --same-dir --collect -q \
      env LD_LIBRARY_PATH=$WBIN CUDA_VISIBLE_DEVICES="" \
      GGML_NUMA_EP=1 GGML_NUMA_EP_GATE_UP_PARALLEL=1 \
      $WBIN/llama-server -m "$MODEL" \
      -ngl 0 -t 72 -tb 72 --numa distribute --no-mmap \
      -fa 1 -b 2048 -ub 512 -c ${CTX_PURE:-131072} -np 1 \
      --no-warmup --no-ui --host 0.0.0.0 --port "$PORT" "${@:2}"
    ;;

  bench-hybrid)
    [ -f "$MODEL" ] || { echo "模型不存在: $MODEL"; exit 1; }
    exec flock /tmp/xllama-bench.lock \
      env GGML_NUMA_EP=1 "$CBIN/llama-bench" -m "$MODEL" \
      -ngl 99 -ncmoe 99 -t 72 -fa 1 -b 4096 -ub 1024 --load-mode none -r 2 --numa distribute \
      -p 2048 -n 512 "${@:2}"
    ;;

  *)
    sed -n 2,16p "$0"
    exit 1
    ;;
esac
