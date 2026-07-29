#!/usr/bin/env bash
# ep-topo-run.sh — 六节点（GPU0/GPU1 + master node0/1 + slave node0/1）拓扑探测驱动
#
# 用法：tools/epd/ep-topo-run.sh [输出.json]   默认 tools/epd/ep-topo-profile.json
#
# 安全：全程持 flock -x /tmp/xllama-bench.lock；开始前检查本机与 slave 无 llama 进程。
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
BIN="$ROOT/build-epdev-topo/ep-topo-probe"
GPU_BIN="$ROOT/build-epdev-topo/ep-topo-gpu"
OUT="${1:-$ROOT/tools/epd/ep-topo-profile.json}"
SLAVE="heiketu@10.0.0.2"
SLAVE_BIN="/home/heiketu/x-llama.cpp/llama-src/build-cpu-topo/ep-topo-probe"
SSH="ssh -o BatchMode=yes $SLAVE"
PORT=29310

exec 9>/tmp/xllama-bench.lock
flock -x 9
echo "[ep-topo] lock acquired"

for host in local slave; do
    if [ "$host" = local ]; then chk="pgrep -x llama-cli; pgrep -x llama-server; pgrep -x llama-epd"
    else chk="$SSH 'pgrep -x llama-cli; pgrep -x llama-server; pgrep -x llama-epd'"; fi
    if eval "$chk" >/dev/null 2>&1; then
        echo "[ep-topo] ERROR: $host 上有 llama 进程在跑，退出" >&2; exit 1
    fi
done
echo "[ep-topo] no llama processes on master/slave"

# --- 构建（幂等）---
mkdir -p "$ROOT/build-epdev-topo"
gcc -O2 -Wall -o "$BIN" "$ROOT/tools/epd/ep-topo-probe.c" -lnuma -lpthread
$SSH 'mkdir -p /home/heiketu/x-llama.cpp/llama-src/build-cpu-topo && \
      gcc -O2 -Wall -o /home/heiketu/x-llama.cpp/llama-src/build-cpu-topo/ep-topo-probe \
          /home/heiketu/x-llama.cpp/llama-src/tools/epd/ep-topo-probe.c -lnuma -lpthread'
echo "[ep-topo] binaries built (master + slave)"

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"; $SSH "pkill -f ep-topo-probe.*tcpecho" 2>/dev/null || true' EXIT

# --- master 本地 ---
"$BIN" membw 0 512 10 2>/dev/null > "$TMP/m_membw0.json"
"$BIN" membw 1 512 10 2>/dev/null > "$TMP/m_membw1.json"
"$BIN" pingpong 0 0 5 2000 2>/dev/null > "$TMP/m_pp00.json"
"$BIN" pingpong 1 1 5 2000 2>/dev/null > "$TMP/m_pp11.json"
"$BIN" pingpong 0 1 5 2000 2>/dev/null > "$TMP/m_pp01.json"
echo "[ep-topo] master local probes done"

# --- GPU ---
if [ -x "$GPU_BIN" ]; then
    "$GPU_BIN" 2>/dev/null > "$TMP/gpu.jsonl" || echo '{"kind":"gpu","error":"failed"}' > "$TMP/gpu.jsonl"
else
    echo '{"kind":"gpu","error":"binary missing"}' > "$TMP/gpu.jsonl"
fi
echo "[ep-topo] gpu probe done"

# --- slave 本地 ---
$SSH "$SLAVE_BIN membw 0 512 10 2>/dev/null" > "$TMP/s_membw0.json"
$SSH "$SLAVE_BIN membw 1 512 10 2>/dev/null" > "$TMP/s_membw1.json"
$SSH "$SLAVE_BIN pingpong 0 0 5 2000 2>/dev/null" > "$TMP/s_pp00.json"
$SSH "$SLAVE_BIN pingpong 1 1 5 2000 2>/dev/null" > "$TMP/s_pp11.json"
$SSH "$SLAVE_BIN pingpong 0 1 5 2000 2>/dev/null" > "$TMP/s_pp01.json"
echo "[ep-topo] slave local probes done"

# --- 跨机 TCP ping：4 组合（master node × slave node） ---
for sn in 0 1; do
    $SSH "pkill -f 'ep-topo-probe.*tcpecho' 2>/dev/null; \
          nohup $SLAVE_BIN tcpecho $PORT $sn >/dev/null 2>&1 & \
          for i in \$(seq 50); do $SLAVE_BIN tcping 127.0.0.1 $PORT -1 >/dev/null 2>&1 && break; sleep 0.1; done"
    for mn in 0 1; do
        "$BIN" tcping 10.0.0.2 $PORT $mn 2>/dev/null > "$TMP/tcp_m${mn}_s${sn}.json"
        echo "[ep-topo] tcping master node$mn -> slave node$sn done"
    done
done
$SSH "pkill -f 'ep-topo-probe.*tcpecho' 2>/dev/null" || true

# --- 汇总 ---
python3 - "$TMP" "$OUT" <<'EOF'
import json, sys, glob, os, datetime
tmp, out = sys.argv[1], sys.argv[2]
def load(name):
    p = os.path.join(tmp, name)
    try:
        with open(p) as f:
            return json.loads(f.read().strip().splitlines()[0])
    except Exception as e:
        return {"error": f"{name}: {e}"}
profile = {
    "kind": "ep-topo-profile",
    "version": 1,
    "timestamp": datetime.datetime.now().astimezone().isoformat(),
    "nodes": ["gpu0", "gpu1", "mnode0", "mnode1", "snode0", "snode1"],
    "master": {
        "membw": [load("m_membw0.json"), load("m_membw1.json")],
        "pingpong": {"0-0": load("m_pp00.json"), "1-1": load("m_pp11.json"),
                     "0-1": load("m_pp01.json")},
    },
    "slave": {
        "membw": [load("s_membw0.json"), load("s_membw1.json")],
        "pingpong": {"0-0": load("s_pp00.json"), "1-1": load("s_pp11.json"),
                     "0-1": load("s_pp01.json")},
    },
    "tcping": {f"m{m}-s{s}": load(f"tcp_m{m}_s{s}.json") for m in (0,1) for s in (0,1)},
}
# GPU 输出是两行 JSON（transfer + p2p）
gpu_lines = open(os.path.join(tmp, "gpu.jsonl")).read().strip().splitlines()
profile["gpu"] = [json.loads(l) for l in gpu_lines if l.strip()]
with open(out, "w") as f:
    json.dump(profile, f, indent=2, ensure_ascii=False)
print(f"[ep-topo] profile written: {out}")
EOF
