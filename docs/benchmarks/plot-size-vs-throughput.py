#!/usr/bin/env python3
# size-vs-throughput chart for the kernel paper: hybrid series (17-point matrix)
# plus CPU-only series from format-size-tg-20260825, with trend lines and
# stream-bake A/B markers. Output: docs/img/size-vs-throughput.png
import json
import pathlib
import re
import sys

import numpy as np

ROOT = pathlib.Path("/home/heiketu/x-llama.cpp/llama-src")
OUTDIR = ROOT / "docs/img"
SWEEP = ROOT / "quant-sweep/format-size-tg-20260825"
OUTDIR.mkdir(parents=True, exist_ok=True)

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    print("matplotlib missing", file=sys.stderr)
    sys.exit(2)

HYBRID = [  # name, size_gib, pp2048, tg512
    ("IQ1_S", 76.87, 251.18, 29.45),
    ("IQ1_M", 80.93, 241.38, 29.03),
    ("IQ2_XXS", 84.62, 247.17, 28.89),
    ("IQ2_M", 84.68, 246.83, 29.24),
    ("Q2_K_XL", 90.18, 232.22, 28.03),
    ("Q2_K-0731(实为IQ2_XXS)", 90.89, 223.60, 22.87),
    ("IQ3_XXS", 97.05, 208.96, 26.74),
    ("IQ3_S", 108.10, 254.48, 23.99),
    ("Q3_K_M", 119.28, 207.20, 25.49),
    ("Q3_K_XL", 119.40, 207.69, 25.34),
    ("IQ4_XS", 127.28, 215.93, 21.10),
    ("IQ4_NL", 127.28, 223.53, 21.36),
    ("Q4_K_XL", 144.44, 314.15, 25.65),
    ("MXFP4", 145.26, 312.48, 26.87),
    ("E4A", 145.26, 362.92, 24.83),
    ("UDNL_W4", 146.36, 370.87, 25.10),
    ("UDNL_MX", 116.13, 418.95, 26.90),
    ("Q8_K_XL", 150.75, 306.17, 21.39),
]

SIZE_OVERRIDE = {
    "UD-IQ1_S": 76.87, "UD-IQ1_M": 80.93, "UD-IQ2_XXS": 84.62, "UD-IQ2_M": 84.68,
    "UD-Q2_K_XL": 90.18, "Q2_K-0731": 90.89, "UD-IQ3_XXS": 97.05, "UD-IQ3_S": 108.10,
    "UD-Q3_K_M": 119.28, "UD-Q3_K_XL": 119.40, "UD-IQ4_XS": 127.28, "UD-IQ4_NL": 127.28,
    "UD-Q4_K_XL": 144.44, "UD-Q8_K_XL": 150.75,
    "IQ2_XS-Experts-Q8_0": 81.90, "IQ3_XXS-Experts-Q8_0": 106.10,
    "IQ2_XS-IQ3_XXS-MXFP4-Q8_0": 112.10, "UDNL_MX-fixed": 116.13,
    "IQ3_S-Experts-Q8_0": 118.10, "MXFP4_MOE-Q8_0": 145.60, "MXFP4_MOE-BF16": 150.80,
}

def parse_sweep():
    cpu = {}  # name -> {size, pp, tg, bake}
    for f in sorted(SWEEP.glob("cpu-*.json")):
        m = re.match(r"cpu-(.+)-b[01]$", f.stem)
        if not m:
            continue
        name = m.group(1)
        try:
            rows = json.loads(f.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        pp = tg = None
        for row in rows:
            if row.get("n_prompt") and not row.get("n_gen"):
                pp = row.get("avg_ts")
            elif row.get("n_gen") and not row.get("n_prompt"):
                tg = row.get("avg_ts")
        if pp is None or tg is None:
            continue
        if name not in cpu:
            cpu[name] = {"size": SIZE_OVERRIDE.get(name, 0.0), "runs": []}
        cpu[name]["runs"].append((float(pp), float(tg)))
    out = {}
    for name, info in cpu.items():
        if info["runs"]:
            pp = np.mean([r[0] for r in info["runs"]])
            tg = np.mean([r[1] for r in info["runs"]])
            out[name] = (info["size"], pp, tg)
    return out

def trend(xs, ys):
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    b = np.polyfit(xs, ys, 1)
    pred = np.polyval(b, xs)
    ss_res = np.sum((ys - pred) ** 2)
    ss_tot = np.sum((ys - ys.mean()) ** 2)
    return b, 1 - ss_res / ss_tot

def draw(ax, is_tg, cpu, title, ylabel):
    hx = [h[1] for h in HYBRID]
    hy = [h[3] if is_tg else h[2] for h in HYBRID]
    ax.scatter(hx, hy, marker="o", color="#377eb8", s=42,
               label="hybrid tg512" if is_tg else "hybrid pp2048", zorder=3)
    b_h, r2_h = trend(hx, hy)
    xs = np.linspace(min(hx) - 5, max(hx) + 5, 50)
    ax.plot(xs, np.polyval(b_h, xs), color="#377eb8", lw=1.2, ls="--",
            label=f"hybrid trend (R²={r2_h:.2f})")

    if cpu:
        cx = [v[0] for v in cpu.values()]
        cy = [v[2] if is_tg else v[1] for v in cpu.values()]
        ax.scatter(cx, cy, marker="^", color="#e41a1c", s=48,
                   label="CPU-only tg128" if is_tg else "CPU-only pp512", zorder=3)
        if len(cx) > 2:
            b_c, r2_c = trend(cx, cy)
            ax.plot(xs, np.polyval(b_c, xs), color="#e41a1c", lw=1.2, ls=":",
                    label=f"CPU-only trend (R²={r2_c:.2f})")

    ax.set_xlabel("model file size (GiB)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, loc="best")

def main():
    cpu = parse_sweep()
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 4.6))
    draw(axes[0], True, cpu, "decode TG vs model size", "TG (tok/s)")
    draw(axes[1], False, cpu, "prefill PP vs model size", "PP (tok/s)")
    fig.tight_layout()
    out = OUTDIR / "size-vs-throughput.png"
    fig.savefig(out, dpi=170)
    print("wrote", out)
    print("cpu points:", len(cpu))

if __name__ == "__main__":
    main()
