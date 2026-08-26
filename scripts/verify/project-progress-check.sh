#!/usr/bin/env bash
# Read-only project health snapshot for the user-level progress timer.
set -uo pipefail

PROJECT_ROOT=/home/heiketu/x-llama.cpp
SOURCE_ROOT=$PROJECT_ROOT/llama-src
SNAPSHOT_DIR=$PROJECT_ROOT/progress-check
LOCK_FILE=/tmp/xllama-progress-check.lock
BENCH_LOCK_FILE=/tmp/xllama-bench.lock
UDNL_LOCK_FILE=/tmp/xllama-udnl-mx-fixed.lock
SLAVE_HOST=10.0.0.2
SOURCE_MODEL=/media/heiketu/21ABB1E86121C730/DSV4NEW/DeepSeek-V4-Flash-0731-abliterated
E4A_SUMMARY=$SOURCE_ROOT/quant-sweep/hot-cold-combo-20260824/e4a-slot-order-avx-final/summary.json
ARCH_PDF=$SOURCE_ROOT/paper/arch-xllama-ep.pdf
KERNEL_PDF=$SOURCE_ROOT/paper/kernel-udnl-formats.pdf
UDNL_MX_META=$SOURCE_ROOT/quant-sweep/udnl-mx-fixed-20260824/quantize.meta
UDNL_MX_OUTPUT='/mnt/storage/大型AI/dsv4 multi/DeepSeek-V4-Flash-0731-GGUF(1)/DeepSeek-V4-Flash-0731-UDNL_MX-fixed-imatrix-20260824.gguf'
UDNL_MX_PARTIAL=$UDNL_MX_OUTPUT.partial
UDNL_MX_RESULT_DIR=$SOURCE_ROOT/quant-sweep/udnl-mx-fixed-20260824
Q234_META=$SOURCE_ROOT/quant-sweep/q2q3q4-first-plan/quantize.meta
Q234_OUTPUT='/mnt/storage/大型AI/dsv4-q2q3q4-speed-first.gguf'
Q234_PARTIAL=$Q234_OUTPUT.partial
Q234_V2_META=$SOURCE_ROOT/quant-sweep/q2q3q4-first-plan-v2/quantize.meta
Q234_V2_OUTPUT='/mnt/storage/大型AI/dsv4-q2q3q4-speed-first-v2.gguf'
Q234_V2_PARTIAL=$Q234_V2_OUTPUT.partial
HOT_REMOTE_DIR=$SOURCE_ROOT/quant-sweep/hot-remote-ep-20260824

mkdir -p "$SNAPSHOT_DIR"
exec 9>"$LOCK_FILE"
flock -n 9 || exit 0

snapshot_tmp=$(mktemp "$SNAPSHOT_DIR/.latest.XXXXXX")
cleanup() {
    if [[ -n ${snapshot_tmp:-} && -e $snapshot_tmp ]]; then
        rm -f -- "$snapshot_tmp"
    fi
}
trap cleanup EXIT

{
    echo "timestamp=$(date --iso-8601=seconds)"
    echo "head=$(git -C "$SOURCE_ROOT" rev-parse --short=12 HEAD 2>/dev/null || echo unavailable)"
    echo "worktree_changes=$(git -C "$SOURCE_ROOT" status --porcelain 2>/dev/null | wc -l)"
    echo "handover_mtime=$(stat -c %y "$PROJECT_ROOT/HANDOVER.md" 2>/dev/null || echo missing)"
    echo "handover_sha256=$(sha256sum "$PROJECT_ROOT/HANDOVER.md" 2>/dev/null | awk '{print $1}' || echo missing)"

    echo "benchmark_lock:"
    if flock -n "$BENCH_LOCK_FILE" true; then
        echo "xllama_bench_lock=free"
    else
        echo "xllama_bench_lock=busy"
    fi
    if flock -n "$UDNL_LOCK_FILE" true; then
        echo "udnl_mx_lock=free"
    else
        echo "udnl_mx_lock=busy"
    fi
    lslocks | awk 'NR == 1 || /xllama-bench/'

    echo "local_processes:"
    pgrep -a -f '(^|/)(llama-server|llama-cli|llama-bench|llama-perplexity|llama-quantize|llama-epd)( |$)' || true
    pgrep -a -f '(^|/)(cmake|ninja|make|gmake)( |$)' || true

    echo "local_ports:"
    ss -ltn | awk 'NR == 1 || /:18108|:2920[0-3]/'

    echo "gpu:"
    nvidia-smi --query-gpu=index,pstate,memory.used,utilization.gpu --format=csv,noheader,nounits 2>/dev/null || true

    echo "cpu_policy:"
    printf 'governor '
    cat /sys/devices/system/cpu/cpu[0-9]*/cpufreq/scaling_governor 2>/dev/null | sort | uniq -c | xargs
    printf 'epp '
    cat /sys/devices/system/cpu/cpu[0-9]*/cpufreq/energy_performance_preference 2>/dev/null | sort | uniq -c | xargs
    printf 'no_turbo '
    cat /sys/devices/system/cpu/intel_pstate/no_turbo 2>/dev/null || true

    echo "key_artifacts:"
    for artifact in \
        "$SOURCE_ROOT/quant-sweep/FORMAL-TG-PATH-REVIEW.md" \
        "$SOURCE_ROOT/quant-sweep/Q2Q4-HYBRID-PHASEA.md" \
        "$SOURCE_ROOT/quant-sweep/q2q4-hybrid-materialize.py" \
        "$SOURCE_ROOT/quant-sweep/MXFP4-DUAL-MACHINE-EP-RESULT.md" \
        "$E4A_SUMMARY" \
        "$ARCH_PDF" \
        "$KERNEL_PDF"; do
        stat -c '%y %s %n' "$artifact" 2>/dev/null || echo "missing $artifact"
    done

    echo "udnl_mx_requantize:"
    sed -n '/^status=/p; /^started_at=/p; /^finished_at=/p; /^rc=/p; /^override_count=/p; /^output_size=/p; /^output_sha256=/p' "$UDNL_MX_META" 2>/dev/null || echo "meta=not_started"
    stat -c 'partial_size=%s partial_mtime=%y' "$UDNL_MX_PARTIAL" 2>/dev/null || true
    stat -c 'final_size=%s final_mtime=%y' "$UDNL_MX_OUTPUT" 2>/dev/null || true
    python3 - "$UDNL_MX_RESULT_DIR" <<'PY' 2>&1 || true
import json
import re
import sys
from pathlib import Path

root = Path(sys.argv[1])
try:
    perf = json.loads((root / "formal-perf/01-UDNL_MX.json").read_text())
    print(f"raw_pp2048={perf['pp2048']:.2f} raw_tg512={perf['tg512']:.2f}")
except Exception as exc:
    print(f"raw_perf=unavailable:{exc}")
try:
    modes = json.loads((root / "mode-scan/summary.json").read_text())
    print(f"mode_tensors={modes['tensor_count']} invalid_mode0={modes['invalid_mode0']} all_w3_rate={modes['all_w3_rate']:.8f}")
except Exception as exc:
    print(f"mode_scan=unavailable:{exc}")
try:
    match = re.findall(r"Final estimate: PPL = ([0-9.]+)", (root / "ppl20.log").read_text())
    print(f"ppl20={match[-1]}" if match else "ppl20=unavailable")
except Exception as exc:
    print(f"ppl20=unavailable:{exc}")
PY

    echo "q2q3q4_materialize:"
    sed -n '/^status=/p; /^started_at=/p; /^finished_at=/p; /^rc=/p; /^expected_counts=/p; /^actual_counts=/p; /^projected_gib=/p; /^dry_run_gib=/p; /^output_size=/p; /^output_sha256=/p' "$Q234_META" 2>/dev/null || echo "meta=not_started"
    stat -c 'partial_size=%s partial_mtime=%y' "$Q234_PARTIAL" 2>/dev/null || true
    stat -c 'final_size=%s final_mtime=%y' "$Q234_OUTPUT" 2>/dev/null || true
    echo "q2q3q4_v2_materialize:"
    sed -n '/^status=/p; /^started_at=/p; /^finished_at=/p; /^rc=/p; /^expected_counts=/p; /^actual_counts=/p; /^projected_gib=/p; /^dry_run_gib=/p; /^output_size=/p; /^output_sha256=/p' "$Q234_V2_META" 2>/dev/null || echo "meta=not_started"
    stat -c 'partial_size=%s partial_mtime=%y' "$Q234_V2_PARTIAL" 2>/dev/null || true
    stat -c 'final_size=%s final_mtime=%y' "$Q234_V2_OUTPUT" 2>/dev/null || true

    echo "hot_remote_ep:"
    python3 - "$HOT_REMOTE_DIR" <<'PY' 2>&1 || true
import statistics
import sys
from pathlib import Path

root = Path(sys.argv[1])
values = {"A": [], "B": []}
for path in sorted(root.glob("formal-*.meta")):
    fields = dict(line.split("=", 1) for line in path.read_text().splitlines() if "=" in line)
    if fields.get("rc") == "0" and fields.get("mode") in values:
        values[fields["mode"]].append(float(fields["tg"]))
if values["A"] and values["B"]:
    a = statistics.mean(values["A"])
    b = statistics.mean(values["B"])
    print(f"tg_A={a:.2f} tg_B={b:.2f} speedup_pct={(b/a-1)*100:.2f} runs_A={len(values['A'])} runs_B={len(values['B'])}")
for mode in ("A", "B"):
    path = root / f"ppl-{mode}.meta"
    if path.exists():
        fields = dict(line.split("=", 1) for line in path.read_text().splitlines() if "=" in line)
        print(f"ppl_{mode}={fields.get('ppl')} bridge_markers_{mode}={fields.get('bridge_marker_count')}")
PY

    echo "final_e4a_gates:"
    python3 - "$E4A_SUMMARY" <<'PY' 2>&1 || true
import json
import sys

try:
    with open(sys.argv[1], "r", encoding="utf-8") as handle:
        summary = json.load(handle)
    gates = summary.get("gates", {})
    required = ("tg", "pp", "ppl", "fixed_seed_response_repeatable")
    print(" ".join(f"{name}={str(bool(gates.get(name))).lower()}" for name in required))
    print(
        f"tg_A={summary['tg']['A_mean']:.4f} tg_B={summary['tg']['B_mean']:.4f} "
        f"pp_B={summary['pp']['B']:.2f} ppl_A={summary['ppl']['A']:.4f} "
        f"ppl_B={summary['ppl']['B']:.4f}"
    )
except Exception as exc:
    print(f"summary_error={exc}")
PY

    echo "paper_sha256:"
    sha256sum "$ARCH_PDF" "$KERNEL_PDF" 2>/dev/null || true

    echo "source_model:"
    if [[ -f $SOURCE_MODEL/model-00017-of-00048.safetensors ]]; then
        echo "shard17=canonical"
    elif [[ -f $SOURCE_MODEL/model-00017-of-00048.safetensors_ ]]; then
        echo "shard17=renamed_with_trailing_underscore"
    else
        echo "shard17=missing"
    fi
    echo "canonical_shards=$(find "$SOURCE_MODEL" -maxdepth 1 \( -type f -o -type l \) -name 'model-*-of-00048.safetensors' 2>/dev/null | wc -l)"

    echo "slave:"
    ssh -o BatchMode=yes -o ConnectTimeout=5 -o ServerAliveInterval=3 "heiketu@$SLAVE_HOST" '
        echo "host=$(hostname)"
        echo "available_kib=$(awk '\''/^MemAvailable:/ {print $2}'\'' /proc/meminfo 2>/dev/null || true)"
        pgrep -a -f '\''(^|/)(llama-server|llama-cli|llama-bench|llama-perplexity|llama-quantize|llama-epd)( |$)'\'' || true
        ss -ltn | awk '\''NR == 1 || /:18108|:2920[0-3]/'\''
    ' 2>&1 || echo "slave_ssh=failed"
} > "$snapshot_tmp"

mv -f -- "$snapshot_tmp" "$SNAPSHOT_DIR/latest.log"
snapshot_tmp=
cat "$SNAPSHOT_DIR/latest.log"
