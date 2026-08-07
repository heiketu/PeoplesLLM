#!/usr/bin/env python3
"""PeoplesLLM FA decode 优化累积图（2026-08-06/07）：
Fig: DSV4 q1 decode fixed-TG64，raw-SWA ring 默认化（f16 KV）与 q8 compact 稀疏 FA（q8_0 KV）。
数据为本轮 fixed-workload A/B 实测定稿值。"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import os

HERE = os.path.dirname(os.path.abspath(__file__))
plt.rcParams.update({"figure.dpi": 110, "font.size": 10, "axes.grid": True, "grid.alpha": 0.3})
C_BASE, C_OURS, C_OURS2 = "#9e9e9e", "#ff8c00", "#4c8bf5"

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.4))

# ---- Panel A: f16 KV, raw-SWA decode ring cumulative (fixed TG64) ----
labels_a = ["baseline\n(before 08-06)", "+ raw-SWA ring\n(default on)", "+ mask-bounds\ntile crop"]
vals_a = [8.894, 11.837, 12.085]
colors_a = [C_BASE, C_OURS, C_OURS]
x = np.arange(len(labels_a))
ax1.bar(x, vals_a, 0.6, color=colors_a)
for i, v in enumerate(vals_a):
    ax1.text(i, v + 0.15, f"{v:.2f}", ha="center", fontsize=9,
             fontweight="bold" if i else "normal")
ax1.annotate(f"+{(vals_a[1]/vals_a[0]-1)*100:.0f}%", (1, vals_a[1] * 0.55),
             ha="center", fontsize=11, fontweight="bold", color="white")
ax1.annotate(f"+{(vals_a[2]/vals_a[0]-1)*100:.0f}%\n(vs base)", (2, vals_a[2] * 0.52),
             ha="center", fontsize=10.5, fontweight="bold", color="white")
ax1.set_xticks(x); ax1.set_xticklabels(labels_a, fontsize=9)
ax1.set_ylabel("Generation throughput (tok/s)")
ax1.set_ylim(0, max(vals_a) * 1.2)
ax1.set_title("q1 decode fixed TG64 — F16 KV\nraw-SWA decode ring default-on (multi-slot auto fallback)")

# ---- Panel B: q8_0 KV, q8 compact sparse FA (fixed TG64) ----
labels_b = ["dense top-k scan\n(q8_0 KV)", "q8 compact sparse FA\n(default on)"]
vals_b = [9.75, 10.69]
x = np.arange(len(labels_b))
ax2.bar(x, vals_b, 0.45, color=[C_BASE, C_OURS2])
for i, v in enumerate(vals_b):
    ax2.text(i, v + 0.15, f"{v:.2f}", ha="center", fontsize=9,
             fontweight="bold" if i else "normal")
ax2.annotate(f"+{(vals_b[1]/vals_b[0]-1)*100:.1f}%", (1, vals_b[1] * 0.55),
             ha="center", fontsize=11, fontweight="bold", color="white")
ax2.axhline(9.66, ls="--", color="#2e7d32", lw=1.4, label="f16 dense = 9.66 (reference)")
ax2.legend(fontsize=8.5, loc="upper left")
ax2.set_xticks(x); ax2.set_xticklabels(labels_b, fontsize=9)
ax2.set_ylim(0, max(vals_b) * 1.2)
ax2.set_title("q1 decode fixed TG64 — q8_0 KV\nonly top-k rows materialized; fastest q1 decode path")

fig.suptitle("DSV4 16K decode: FA structural fixes, fixed-workload A/B (2026-08-06/07)", fontsize=12)
fig.tight_layout(rect=(0, 0, 1, 0.94))
fig.savefig(os.path.join(HERE, "fa_decode_tg64_progression.png"))
plt.close(fig)
print("fa_decode_tg64_progression.png done", vals_a, vals_b)
