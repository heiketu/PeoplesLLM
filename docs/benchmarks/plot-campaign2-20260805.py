#!/usr/bin/env python3
"""PeoplesLLM campaign2 复测图表（2026-08-05 晚，负载环境）：
Fig1 纯 CPU 推理 vs 主线；Fig2 多 slot 并发聚合吞吐 vs 主线。
数据直接解析 bench-20260805/*.log，缺数据的分组自动跳过。"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import os, re, glob

HERE = os.path.dirname(os.path.abspath(__file__))
LOGS = os.path.join(HERE, "..", "..", "..", "bench-20260805")
plt.rcParams.update({"figure.dpi": 110, "font.size": 10, "axes.grid": True, "grid.alpha": 0.3})
C_BASE, C_OURS, C_OURS2 = "#9e9e9e", "#4c8bf5", "#2e7d32"

def parse_bench(path):
    """llama-bench 表格 -> {test: t/s}"""
    out = {}
    if not os.path.exists(path):
        return out
    for line in open(path):
        m = re.match(r"\|.*\|\s*(pp512|tg512|tg64)\s*\|\s*([\d.]+)", line)
        if m:
            out[m.group(1)] = float(m.group(2))
    return out

# ---- Fig 1: pure CPU (recipeB -ngl 0, 72 threads, interleave) ----
ours = parse_bench(os.path.join(LOGS, "cpu-ours.log"))
up = parse_bench(os.path.join(LOGS, "cpu-upstream.log"))
tests = [t for t in ("pp512", "tg512", "tg64") if t in ours and t in up]
if tests:
    x = np.arange(len(tests)); w = 0.36
    fig, ax = plt.subplots(figsize=(8, 4.2))
    b1 = ax.bar(x - w / 2, [up[t] for t in tests], w, label="upstream llama.cpp (b10173)", color=C_BASE)
    b2 = ax.bar(x + w / 2, [ours[t] for t in tests], w, label="PeoplesLLM", color=C_OURS)
    for i, t in enumerate(tests):
        ax.text(i - w / 2, up[t] + max(ours.values()) * 0.01, f"{up[t]:.1f}", ha="center", fontsize=9)
        ax.text(i + w / 2, ours[t] + max(ours.values()) * 0.01, f"{ours[t]:.1f}", ha="center", fontsize=9, fontweight="bold")
        ax.text(i + w / 2, ours[t] * 0.5, f"+{(ours[t]/up[t]-1)*100:.0f}%", ha="center", fontsize=11,
                fontweight="bold", color="white")
    ax.set_xticks(x); ax.set_xticklabels(tests)
    ax.set_ylabel("tok/s")
    ax.set_title("Pure-CPU inference: DSV4 284B Q3_K, -ngl 0, 72 threads, interleave\n"
                 "(2026-08-05, loaded environment; upstream ran with mmap on — deprecated flag)")
    ax.legend(fontsize=9)
    ax.set_ylim(0, max(ours.values()) * 1.18)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "pure_cpu_vs_upstream.png"))
    plt.close(fig)
    print("pure_cpu_vs_upstream.png done", {t: (up[t], ours[t]) for t in tests})
else:
    print("Fig1 skipped: missing cpu logs", ours, up)

# ---- Fig 2: multi-slot aggregate throughput ----
def parse_slots():
    """-> {impl: {N: aggregate_tok_s}}"""
    res = {}
    for f in glob.glob(os.path.join(LOGS, "slots-*-*.log")):
        base = os.path.basename(f)
        m = re.match(r"slots-(ours|up|upstream)-(\d+)\.log", base)
        if not m:
            continue
        impl, n = m.group(1), int(m.group(2))
        impl = "upstream" if impl in ("up", "upstream") else "ours"
        agg = None
        for line in open(f):
            mm = re.search(r"aggregate=([\d.]+) tok/s", line)
            if mm:
                agg = float(mm.group(1))
        if agg is not None:
            res.setdefault(impl, {})[n] = agg
    return res

slots = parse_slots()
if slots:
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ns = sorted({n for v in slots.values() for n in v})
    w = 0.36; x = np.arange(len(ns))
    for k, (impl, color, label) in enumerate((("ours", C_OURS, "PeoplesLLM (mirror + row-pin)"),
                                              ("upstream", C_BASE, "upstream llama.cpp"))):
        if impl not in slots:
            continue
        vals = [slots[impl].get(n, 0) for n in ns]
        ax.bar(x + (k - 0.5) * w if len(slots) > 1 else x, vals, w if len(slots) > 1 else 0.5,
               label=label, color=color)
        for i, v in enumerate(vals):
            if v:
                ax.text((x[i] + (k - 0.5) * w) if len(slots) > 1 else x[i], v + 0.3, f"{v:.1f}",
                        ha="center", fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels([f"{n} slot{'s' if n>1 else ''}" for n in ns])
    ax.set_xlabel("concurrent requests (each 512 tok)")
    ax.set_ylabel("aggregate throughput (tok/s)")
    ax.set_title("Multi-slot concurrency: DSV4 284B hybrid (14 layers on 2x RTX 3090)\n"
                 "llama-server -np 8, 2026-08-05 loaded environment")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "multislot_concurrency.png"))
    plt.close(fig)
    print("multislot_concurrency.png done", slots)
else:
    print("Fig2 skipped: no slots logs yet")
