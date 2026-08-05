#!/usr/bin/env python3
"""PeoplesLLM 2026-08-04 长上下文优化成果图表（数据出自 进度.md/记录.md 实测）"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import os

OUT = os.path.dirname(os.path.abspath(__file__))
plt.rcParams.update({"figure.dpi": 110, "font.size": 10, "axes.grid": True, "grid.alpha": 0.3})

# ---- Chart 1: 16K PP progression (DSV4-Flash, 16384 tokens) ----
stages = [
    ("layer-major v1", 161.2),
    ("safe opts", 174.3),
    ("exact FA bounds\n+ device HC", 500.4),
    ("streaming GPU-MoE\n(3-slot prefetch)", 522.3),
    ("tile 8192", 538.1),
    ("true dual-GPU EP", 581.5),
    ("+ ordered K/V reuse", 604.2),
    ("+ sparse raw compact\n(opt-in)", 752.2),
]
fig, ax = plt.subplots(figsize=(10, 5))
x = np.arange(len(stages))
vals = [s[1] for s in stages]
bars = ax.bar(x, vals, color=["#9e9e9e"] * 2 + ["#4c8bf5"] * 5 + ["#2e7d32"])
for i, v in enumerate(vals):
    ax.text(i, v + 8, f"{v:.0f}", ha="center", fontsize=9)
ax.set_xticks(x)
ax.set_xticklabels([s[0] for s in stages], fontsize=8)
ax.set_ylabel("Prefill throughput (tok/s)")
ax.set_title("DSV4-Flash 16K prefill progression (2x RTX 3090 + dual-socket Xeon, layer-major path)")
ax.set_ylim(0, 830)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "longctx_pp_progression.png"))
plt.close(fig)

# ---- Chart 2: MXFP4 Hybrid CPU-MoE audit + dual-GPU EP scaling ----
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
labels = ["4K", "16K", "16K + exact Q8\nGPU/CPU boundary"]
before = [110.6, 127.8, 127.8]
after = [270.2, 232.3, 251.1]
x = np.arange(3)
w = 0.35
ax1.bar(x - w / 2, before, w, label="before CPU audit", color="#bdbdbd")
ax1.bar(x + w / 2, after, w, label="after audit", color="#4c8bf5")
for i, (b, a) in enumerate(zip(before, after)):
    ax1.text(i - w / 2, b + 4, f"{b:.0f}", ha="center", fontsize=8)
    ax1.text(i + w / 2, a + 4, f"{a:.0f}", ha="center", fontsize=8)
ax1.set_xticks(x); ax1.set_xticklabels(labels, fontsize=8)
ax1.set_ylabel("tok/s"); ax1.legend(fontsize=8)
ax1.set_title("MXFP4 Hybrid CPU-MoE prefill\n(NUMA EP audit: +144% @4K)")

ep = [("serial correct\n(single stream)", 277.2), ("true dual-GPU EP\n(CUDA0 || CUDA1)", 451.8)]
ax2.bar([0, 1], [e[1] for e in ep], color=["#bdbdbd", "#2e7d32"])
for i, e in enumerate(ep):
    ax2.text(i, e[1] + 6, f"{e[1]:.0f}", ha="center")
ax2.set_xticks([0, 1]); ax2.set_xticklabels([e[0] for e in ep], fontsize=8)
ax2.set_ylabel("tok/s")
ax2.set_title("Same-layer expert-axis EP @2K\n(+63.0%, bit-identical output)")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "mxfp4_hybrid_cpu_audit.png"))
plt.close(fig)

# ---- Chart 3: TG improvements @16K ----
fig, ax = plt.subplots(figsize=(10, 4.5))
groups = [
    ("q1 FA kernel\n(fixed TG64)", 7.66, 8.96, "8-head -> 32-head (+17.0%)"),
    ("raw-SWA decode ring\n(fixed TG64)", 9.67, 11.80, "+22.1%"),
    ("raw-SWA decode ring\n(fixed TG512)", 11.30, 12.18, "+7.8%"),
]
x = np.arange(len(groups))
w = 0.35
b = [g[1] for g in groups]
a = [g[2] for g in groups]
ax.bar(x - w / 2, b, w, label="before", color="#bdbdbd")
ax.bar(x + w / 2, a, w, label="after", color="#ff8f00")
for i, g in enumerate(groups):
    ax.text(i - w / 2, g[1] + 0.15, f"{g[1]:.2f}", ha="center", fontsize=8)
    ax.text(i + w / 2, g[2] + 0.15, f"{g[2]:.2f}", ha="center", fontsize=8)
    ax.text(i, max(g[1], g[2]) + 1.0, g[3], ha="center", fontsize=8, color="#555555")
ax.set_xticks(x); ax.set_xticklabels([g[0] for g in groups], fontsize=9)
ax.set_ylabel("Generation throughput (tok/s)")
ax.set_title("DSV4-Flash 16K decode (TG) kernel-level improvements, fixed-workload A/B")
ax.legend(fontsize=8)
ax.set_ylim(0, 15)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "longctx_tg_improvements.png"))
plt.close(fig)

# ---- Chart 4: 16K Nsight hotspot breakdown (604.2 tok/s path, 27.1s total) ----
fig, ax = plt.subplots(figsize=(7.5, 5))
parts = [
    ("Flash Attention", 11.892, "#4c8bf5"),
    ("H2D weight transfer", 7.480, "#ff8f00"),
    ("MXFP4 MoE MMQ", 3.381, "#2e7d32"),
    ("Q8 matmul", 1.614, "#8e24aa"),
    ("Lightning Indexer", 1.395, "#00838f"),
    ("other", 27.115 - 11.892 - 7.480 - 3.381 - 1.614 - 1.395, "#bdbdbd"),
]
wedges, _, autotexts = ax.pie(
    [p[1] for p in parts], labels=[f"{p[0]}\n{p[1]:.2f}s" for p in parts],
    colors=[p[2] for p in parts], autopct="%1.1f%%", pctdistance=0.75,
    textprops={"fontsize": 9})
for t in autotexts:
    t.set_fontsize(8)
ax.set_title("16K prefill GPU-side time breakdown (27.1s total, 604 tok/s path)\nNsight profile — FA ~50%, H2D ~28% are the next targets")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "longctx_hotspots.png"))
plt.close(fig)

print("charts written:", os.listdir(OUT))
