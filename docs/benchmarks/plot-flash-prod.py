#!/usr/bin/env python3
"""生产 server PP/TG 合并图（单张双面板，log-x）。
数据来自 bench-flash-prod.py 的 PROD 行。用法: plot-flash-prod.py <log文件> <标题后缀>"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import os, re, sys

HERE = os.path.dirname(os.path.abspath(__file__))
plt.rcParams.update({"figure.dpi": 110, "font.size": 10, "axes.grid": True, "grid.alpha": 0.3,
                     "grid.which": "both"})
C_PP, C_TG = "#4c8bf5", "#2e7d32"

def parse(path):
    L, pp, tg = [], [], []
    for line in open(path):
        m = re.search(r"PROD L=(\d+) pp=([\d.]+) tg=([\d.]+)", line)
        if m:
            L.append(int(m.group(1))); pp.append(float(m.group(2))); tg.append(float(m.group(3)))
    return np.array(L), np.array(pp), np.array(tg)

def fmt(x, _):
    if x >= 1048576: return f"{x/1048576:.0f}M"
    if x >= 1024: return f"{x/1024:.0f}K"
    return str(int(x))

log = sys.argv[1]
suffix = sys.argv[2] if len(sys.argv) > 2 else ""
L, pp, tg = parse(log)
order = np.argsort(L)
L, pp, tg = L[order], pp[order], tg[order]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.4))
ax1.semilogx(L, pp, "-o", color=C_PP, lw=2, ms=5)
for x, y in zip(L, pp):
    ax1.annotate(f"{y:.0f}", (x, y), textcoords="offset points", xytext=(0, 7), ha="center", fontsize=8)
ax1.set_title("Prompt processing (prefill)")
ax1.set_xlabel("context depth (tokens)")
ax1.set_ylabel("tok/s")
ax1.xaxis.set_major_formatter(plt.FuncFormatter(fmt))

ax2.semilogx(L, tg, "-s", color=C_TG, lw=2, ms=5)
for x, y in zip(L, tg):
    ax2.annotate(f"{y:.1f}", (x, y), textcoords="offset points", xytext=(0, 7), ha="center", fontsize=8)
ax2.set_title("Generation (decode) at depth")
ax2.set_xlabel("context depth (tokens)")
ax2.set_ylabel("tok/s")
ax2.xaxis.set_major_formatter(plt.FuncFormatter(fmt))

fig.suptitle(f"DSV4-Flash production server — 8 slots, Q8 KV cache, 2x RTX 3090 + dual Xeon 8360Y {suffix}", fontsize=11)
fig.tight_layout()
out = os.path.join(HERE, "flash_prod_pp_tg.png")
fig.savefig(out)
print("saved", out, dict(zip(L.tolist(), zip(pp.tolist(), tg.tolist()))))
