#!/usr/bin/env python3
"""PeoplesLLM README 图表（2026-08-05）：vs 主线基线 + 头条加速比 + CPU 内核加速比。
数据全部出自仓库公开、复核后的 README 与 benchmarks 记录。"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import os

OUT = os.path.dirname(os.path.abspath(__file__))
plt.rcParams.update({"figure.dpi": 110, "font.size": 10, "axes.grid": True, "grid.alpha": 0.3})
C_BASE, C_OURS, C_OURS2, C_ACC = "#9e9e9e", "#4c8bf5", "#2e7d32", "#ff8f00"

# ---- Fig 1: DSV4 284B vs upstream llama.cpp (same-machine A/B, 2026-08-01) ----
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
names = ["upstream\nllama.cpp", "PeoplesLLM\nrow-window EP", "PeoplesLLM\nNUMA mirror"]
pp = [161.3, 333.0, 349.0]
tg = [28.0, 30.48, 30.44]
for ax, vals, title, ylim in ((ax1, pp, "Prompt processing pp512 (tok/s)", 400),
                              (ax2, tg, "Generation tg512 (tok/s)", 36)):
    bars = ax.bar(range(3), vals, color=[C_BASE, C_OURS, C_OURS2])
    for i, v in enumerate(vals):
        ax.text(i, v + ylim * 0.01, f"{v:.0f}" if v > 50 else f"{v:.1f}", ha="center", fontsize=10)
    for i in (1, 2):
        ax.text(i, vals[i] * 0.5, f"{vals[i]/vals[0]:.1f}x" if vals[i]/vals[0] >= 1.5 else f"+{(vals[i]/vals[0]-1)*100:.0f}%",
                ha="center", fontsize=12, fontweight="bold", color="white")
    ax.set_xticks(range(3)); ax.set_xticklabels(names, fontsize=9)
    ax.set_title(title, fontsize=11); ax.set_ylim(0, ylim)
fig.suptitle("DeepSeek-V4 284B (Q3_K) — dual-socket Xeon 8360Y + 2x RTX 3090, same-methodology A/B", fontsize=11)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "dsv4_vs_upstream.png"))
plt.close(fig)

# ---- Fig 2: headline speedups (horizontal bars, log x) ----
items = [
    ("Batched top-k kernel (16K x 4K, k=512)", 21.6),
    ("Layer-major 16K prefill (vs first version)", 4.7),
    ("CPU gemm kernels (micro-bench, max)", 4.9),
    ("GLM-5.2 PP1020 (IQ2_XS/IQ3_XXS traits)", 4.1),
    ("MXFP4 Hybrid 4K prefill (CPU audit)", 2.44),
    ("DSV4 pp512 (vs upstream)", 2.17),
    ("Dual-GPU same-layer EP @2K", 1.63),
    ("DSV4 tg512 (vs upstream)", 1.09),
    ("GLM-5.2 tg512 (IQ traits)", 1.06),
]
items = items[::-1]
fig, ax = plt.subplots(figsize=(10, 5))
y = np.arange(len(items))
vals = [i[1] for i in items]
colors = [C_ACC if v >= 4 else C_OURS if v >= 1.5 else C_OURS2 for v in vals]
ax.barh(y, vals, color=colors)
for i, v in enumerate(vals):
    ax.text(v + 0.15, i, f"{v:.2f}x" if v < 4 else f"{v:.1f}x", va="center", fontsize=10, fontweight="bold")
ax.set_yticks(y); ax.set_yticklabels([i[0] for i in items], fontsize=9)
ax.set_xlabel("Speedup (x)")
ax.set_xlim(0, 24)
ax.axvline(1.0, color="#888888", lw=0.8, ls="--")
ax.set_title("PeoplesLLM headline speedups (all measured, same-methodology A/B)")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "headline_speedups.png"))
plt.close(fig)

# ---- Fig 3: CPU kernel gemm speedup (72 threads, production config) ----
fmts = ["Q2_K", "Q3_K*", "Q4_0", "Q4_K", "Q5_K*", "Q6_K*", "MXFP4", "Q8_0*", "IQ1_S*", "IQ1_M*"]
nr16 = [1.09, 1.38, 2.31, 1.72, 2.41, 2.12, 2.06, 2.09, 2.49, 2.71]
nr32 = [0.95, 1.54, 3.04, 2.01, 2.53, 2.96, 3.01, 2.79, 3.15, 3.11]
gemv = [1.32, 1.25, 1.12, 1.45, 0.96, 1.03, 1.45, 1.26, 1.23, 1.20]
x = np.arange(len(fmts))
w = 0.28
fig, ax = plt.subplots(figsize=(11, 4.5))
ax.bar(x - w, gemv, w, label="gemv (decode, batch=1)", color=C_BASE)
ax.bar(x, nr16, w, label="gemm nr=16 (prefill)", color=C_OURS)
ax.bar(x + w, nr32, w, label="gemm nr=32 (prefill)", color=C_OURS2)
for i in range(len(fmts)):
    ax.text(x[i] + w, nr32[i] + 0.05, f"{nr32[i]:.1f}", ha="center", fontsize=8)
ax.set_xticks(x); ax.set_xticklabels(fmts, fontsize=9)
ax.axhline(1.0, color="#888888", lw=0.8, ls="--")
ax.set_ylabel("Speedup vs upstream legacy vec_dot (x)")
ax.set_title("AVX512/VNNI/VBMI 8x8 repack kernels — 72 threads, nc=16384 k=8192 (DRAM-bound)\n* = kernel stack new in this fork")
ax.legend(fontsize=9)
ax.set_ylim(0, 3.6)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "cpu_kernel_speedup.png"))
plt.close(fig)

# ---- Fig 4: GLM-5.2 traits effect ----
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
labels = ["before traits", "+ IQ2_XS/IQ3_XXS\ntraits + gemm dispatch"]
pp1020 = [99.1, 406.0]
tg512 = [11.25, 11.95]
ax1.bar(range(2), pp1020, color=[C_BASE, C_OURS])
for i, v in enumerate(pp1020):
    ax1.text(i, v + 8, f"{v:.0f}", ha="center", fontsize=11)
ax1.text(1, 220, "4.1x", ha="center", fontsize=14, fontweight="bold", color="white")
ax1.set_xticks(range(2)); ax1.set_xticklabels(labels, fontsize=9)
ax1.set_title("GLM-5.2 745B pp1020 (tok/s)"); ax1.set_ylim(0, 470)
ax2.bar(range(2), tg512, color=[C_BASE, C_OURS2])
for i, v in enumerate(tg512):
    ax2.text(i, v + 0.2, f"{v:.2f}", ha="center", fontsize=11)
ax2.text(1, 7, "+6%", ha="center", fontsize=13, fontweight="bold", color="white")
ax2.set_xticks(range(2)); ax2.set_xticklabels(labels, fontsize=9)
ax2.set_title("GLM-5.2 745B tg512 (tok/s)"); ax2.set_ylim(0, 14)
fig.suptitle("GLM-5.2 (UD-Q2_K_MXFP4), single machine + 2x RTX 3090", fontsize=11)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "glm_traits.png"))
plt.close(fig)

print("done:", [f for f in os.listdir(OUT) if f.endswith('.png')])

# ---- Fig 5: two-machine EP (GLM-5.2, 2026-07-31, pre-traits) ----
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
labels = ["single\nmachine", "dual 15L\nTCP", "dual 15L\nRDMA", "dual 15L\nRDMA+MIRROR"]
tg = [13.2, 11.85, 12.16, 12.93]
colors = [C_BASE, C_OURS, C_OURS, C_OURS2]
ax1.bar(range(4), tg, color=colors)
for i, v in enumerate(tg):
    ax1.text(i, v + 0.15, f"{v:.2f}", ha="center", fontsize=9)
ax1.set_xticks(range(4)); ax1.set_xticklabels(labels, fontsize=8)
ax1.set_title("tg512 (tok/s)"); ax1.set_ylim(0, 15.5)
labels2 = ["dual 15L\nTCP", "dual 15L\nRDMA"]
pp = [33.4, 76.1]
ax2.bar(range(2), pp, color=[C_OURS, C_OURS2])
for i, v in enumerate(pp):
    ax2.text(i, v + 1.5, f"{v:.1f}", ha="center", fontsize=10)
ax2.text(1, 45, "2.3x", ha="center", fontsize=13, fontweight="bold", color="white")
ax2.set_xticks(range(2)); ax2.set_xticklabels(labels2, fontsize=8)
ax2.set_title("pp1020 (tok/s) — RDMA large-frame fix"); ax2.set_ylim(0, 88)
fig.suptitle("Two-machine expert-parallel: GLM-5.2 745B over 100G RoCEv2 direct link (pre-traits data)", fontsize=11)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "dual_machine_ep.png"))
plt.close(fig)
print("dual_machine_ep.png done")

# ---- Fig 6: NUMA locality — EP local-read vs upstream distribute ----
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
# A: access origin structure
impl = ["upstream\ndistribute", "PeoplesLLM\nrow-window EP", "PeoplesLLM\nNUMA mirror"]
local = [50, 100, 100]
cross = [50, 0, 0]
ax1.bar(range(3), local, color=C_OURS2, label="local-node read")
ax1.bar(range(3), cross, bottom=local, color="#d32f2f", label="cross-UPI read")
for i in range(3):
    ax1.text(i, 50, f"{local[i]}% local", ha="center", fontsize=11, fontweight="bold",
             color="white" if local[i] >= 50 else "#333")
ax1.set_xticks(range(3)); ax1.set_xticklabels(impl, fontsize=9)
ax1.set_ylabel("share of MoE weight reads (%)")
ax1.set_title("Weight-read locality (structural)")
ax1.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2)
ax1.set_ylim(0, 105)
# B: measured effective bandwidth per access pattern
pats = ["cross-socket\n(UPI path)", "upstream pattern\n(interleave, 152 thr)", "EP/mirror pattern\n(local x2 sockets)"]
bw = [54.1, 177.6, 279.2]
bars = ax2.bar(range(3), bw, color=["#d32f2f", C_BASE, C_OURS2])
for i, v in enumerate(bw):
    ax2.text(i, v + 5, f"{v:.0f}", ha="center", fontsize=11, fontweight="bold")
ax2.text(2, 140, "+57%", ha="center", fontsize=14, fontweight="bold", color="white")
ax2.set_xticks(range(3)); ax2.set_xticklabels(pats, fontsize=8.5)
ax2.set_ylabel("aggregate read bandwidth (GB/s)")
ax2.set_title("Measured effective read bandwidth\n(membw2, 76 thr/socket, 2026-08-05)")
ax2.set_ylim(0, 320)
fig.suptitle("NUMA locality: why row-window EP/mirror beat upstream distribute — dual Xeon 8360Y", fontsize=11)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "numa_locality.png"))
plt.close(fig)
print("numa_locality.png done")
