#!/usr/bin/env python3
"""Regenerate the public project overview charts.

Requires matplotlib and numpy.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


OUT = Path(__file__).resolve().parents[1] / "img"
OUT.mkdir(parents=True, exist_ok=True)

GREEN = "#4d8b57"
BLUE = "#3e70a8"
ORANGE = "#c46239"
PURPLE = "#7f5aaa"
GRAY = "#8b9098"


def finish(fig, name):
    fig.savefig(OUT / name, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def evolution():
    stages = [
        "Baseline\nMXFP4",
        "Spec-decode\ntune (n2/p0)",
        "Op fusion +\nasync readback",
        "arec GEMM\n(UDNL_MX)",
        "64-row claim +\ntail batching",
        "Hot-expert GPU\nresidency",
    ]
    x = np.arange(len(stages))
    pp = [309, 309, 309, 385, 385, 385]
    raw = [24.9, 24.9, 25.9, 25.9, 26.4, 28.5]
    spec = [23.9, 26.8, 26.8, 26.8, 30.1, 30.1]

    fig, left = plt.subplots(figsize=(10, 4.8))
    right = left.twinx()
    left.step(x, pp, where="post", marker="o", color=GREEN, linewidth=2.2, label="pp2048 (tok/s, left)")
    right.step(x, raw, where="post", marker="s", color=BLUE, linewidth=2.2, label="tg512 raw (tok/s, right)")
    right.step(x, spec, where="post", marker="^", linestyle="--", color=ORANGE, linewidth=2.2, label="tg512 speculative (tok/s, right)")
    left.set_ylim(250, 430)
    right.set_ylim(20, 33)
    left.set_ylabel("Prefill pp2048 (tok/s)", color=GREEN)
    right.set_ylabel("Decode tg512 (tok/s)")
    left.set_xticks(x, stages)
    left.grid(alpha=0.25)
    lines = left.lines + right.lines
    left.legend(lines, [line.get_label() for line in lines], loc="upper left")
    left.set_title("DSV4-Flash performance milestones (2 x Xeon ICX-SP + 2 x RTX 3090)\nindependent measured tracks; values are not multiplicative")
    for px, py, label, offset in [(0, 309, "309", (-12, 10)), (3, 385, "385", (0, 10))]:
        left.annotate(label, (px, py), xytext=offset, textcoords="offset points", ha="center", color=GREEN, fontweight="bold")
    for px, py, label, offset in [(0, 24.9, "24.9", (0, -18)), (2, 25.9, "25.9", (0, -16)), (4, 26.4, "26.4", (0, -16)), (5, 28.5, "28.5", (0, -16))]:
        right.annotate(label, (px, py), xytext=offset, textcoords="offset points", ha="center", color=BLUE, fontweight="bold")
    for px, py, label in [(0, 23.9, "23.9"), (1, 26.8, "26.8"), (4, 30.1, "30.1")]:
        right.annotate(label, (px, py), xytext=(0, 10), textcoords="offset points", ha="center", color=ORANGE, fontweight="bold")
    finish(fig, "evolution-staircase.png")


def numa_tp():
    labels = ["hybrid tg512", "hybrid pp2048", "pure-CPU tg128", "pure-CPU pp512"]
    off = np.array([16.59, 267.05, 7.23, 101.41])
    on = np.array([25.01, 298.98, 11.65, 107.91])
    changes = ["+51%", "+12%", "+61%", "+6.4%"]
    x = np.arange(len(labels))
    width = 0.36
    fig, ax = plt.subplots(figsize=(9, 4.5))
    bars_off = ax.bar(x - width / 2, off, width, color=GRAY, label="off (interleave + round-robin)")
    bars_on = ax.bar(x + width / 2, on, width, color=BLUE, label="on (row-window TP + dynamic claim)")
    ax.set_xticks(x, labels)
    ax.set_ylabel("tok/s")
    ax.set_ylim(0, 345)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="upper right")
    ax.set_title("Intra-device NUMA tensor parallelism, DSV4-Flash MXFP4 (ABAB verified)\nlocal+bound 313.5 GB/s vs interleave 122 GB/s (2.57 x)")
    for bar, value in zip(bars_off, off):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 4, f"{value:.2f}", ha="center")
    for bar, value, change in zip(bars_on, on, changes):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 4, f"{value:.2f}\n{change}", ha="center", color=BLUE, fontweight="bold")
    finish(fig, "numa-tp-onoff.png")


def formats():
    # Full-format matrix measured on 2026-08-21 with one hybrid setup.
    rows = [
        ("UD-IQ1_S",                         76.9, 256.86, 27.94),
        ("UD-IQ1_M",                         80.9, 242.84, 26.44),
        ("UD-IQ2_XXS",                       84.6, 251.75, 27.17),
        ("UD-IQ2_M",                         84.7, 244.99, 26.60),
        ("UD-Q2_K_XL",                       90.2, 236.31, 26.56),
        ("UD-IQ3_XXS",                       97.1, 209.67, 24.77),
        ("UD-IQ3_S",                        108.1, 265.20, 22.97),
        ("mix-IQ2XS/IQ3XXS/MXFP4",         112.1, 233.71, 25.14),
        ("UD-Q3_K_M",                       119.3, 209.77, 24.05),
        ("UD-Q3_K_XL",                      119.4, 209.62, 23.92),
        ("UD-IQ4_XS",                       127.3, 226.92, 21.05),
        ("UD-IQ4_NL",                       127.3, 229.91, 20.74),
        ("UD-Q4_K_XL",                      144.4, 318.49, 25.17),
        ("mix-MXFP4-MoE/Q8",                145.6, 313.28, 24.49),
        ("UD-Q8_K_XL",                      150.8, 307.48, 20.78),
        ("mix-MXFP4-MoE/BF16",              150.8, 306.81, 20.59),
    ]
    rows.sort(key=lambda row: (row[1], row[0]))

    def smaller_but_slower(metric):
        result = []
        for index, row in enumerate(rows):
            size = row[1]
            speed = row[metric]
            result.append(any(other[1] > size + 0.05 and other[metric] > speed * 1.03 for other in rows[index + 1:]))
        return result

    sizes = np.array([row[1] for row in rows])
    pp = np.array([row[2] for row in rows])
    tg = np.array([row[3] for row in rows])
    counter_pp = smaller_but_slower(2)
    counter_tg = smaller_but_slower(3)

    fig, axes = plt.subplots(2, 1, figsize=(11.5, 8.7), sharex=True)
    close_size_label_dx = {3: -10, 4: 10, 9: -10, 10: 10, 11: -10, 12: 10, 15: -10, 16: 10}
    for ax, values, counter, color, ylabel, trend_x_pos, trend_ha in (
        (axes[0], pp, counter_pp, GREEN, "pp2048 (tok/s)", 0.015, "left"),
        (axes[1], tg, counter_tg, BLUE, "tg512 (tok/s)", 0.985, "right"),
    ):
        trend_coeff = np.polyfit(sizes, values, 1)
        trend_x = np.linspace(sizes.min(), sizes.max(), 200)
        fitted = np.polyval(trend_coeff, sizes)
        residual = np.sum((values - fitted) ** 2)
        total = np.sum((values - values.mean()) ** 2)
        r_squared = 1.0 - residual / total
        ax.plot(trend_x, np.polyval(trend_coeff, trend_x), color=color, linewidth=2.0, linestyle="--", alpha=0.8, zorder=1)
        for index, (size, speed, is_counterexample) in enumerate(zip(sizes, values, counter), 1):
            point_color = "#d62728" if is_counterexample else color
            marker = "D" if is_counterexample else "o"
            ax.scatter(size, speed, s=70, marker=marker, color=point_color, edgecolor="black", linewidth=0.7, zorder=3)
            label_dx = close_size_label_dx.get(index, 0)
            ax.annotate(str(index), (size, speed), xytext=(label_dx, 7), textcoords="offset points", ha="center", fontsize=7.5)
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25)
        ax.text(trend_x_pos, 0.94, f"linear trend: {trend_coeff[0]:+.3f} tok/s/GiB, R²={r_squared:.2f}",
                transform=ax.transAxes, ha=trend_ha, va="top", color=color, fontsize=9, fontweight="bold")

    axes[0].set_ylim(160, 335)
    axes[1].set_ylim(17.5, 29.5)
    axes[1].set_xlim(73, 154)
    axes[1].set_xlabel("Model size (GiB)")
    axes[0].set_title(
        "DSV4-Flash format scatter and least-squares trend: size alone does not explain speed\n"
        "red diamond = a smaller format is at least 3% slower than a larger format"
    )

    entries = [f"{index:>2}  {name} ({size:.1f}G)" for index, (name, size, _, _) in enumerate(rows, 1)]
    columns = 2
    rows_per_column = (len(entries) + columns - 1) // columns
    legend_lines = []
    for row_index in range(rows_per_column):
        cells = []
        for column in range(columns):
            entry_index = column * rows_per_column + row_index
            cells.append(entries[entry_index] if entry_index < len(entries) else "")
        legend_lines.append("    ".join(f"{cell:<39}" for cell in cells))
    fig.text(0.5, 0.005, "\n".join(legend_lines), ha="center", va="bottom", family="monospace", fontsize=7.3)
    fig.tight_layout(rect=(0, 0.145, 1, 1))
    finish(fig, "quant-formats.png")


def gpu_prefill():
    ubatch = np.array([2048, 4096, 8192, 16384])
    gpu = np.array([241.1, 349.8, 399.7, 327.2])
    cpu = np.array([175.4, 173.3, 154.5, 134.7])
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(ubatch, gpu, marker="o", linewidth=2.4, color=ORANGE, label="GPU layer-major streaming MoE prefill")
    ax.plot(ubatch, cpu, marker="s", linewidth=2.4, color=GRAY, label="CPU repack GEMM reference")
    ax.set_xticks(ubatch, ["2K", "4K", "8K", "16K"])
    ax.set_xlabel("Micro-batch size (16K-token prompt)")
    ax.set_ylabel("Prefill (tok/s)")
    ax.set_ylim(110, 420)
    ax.grid(alpha=0.25)
    ax.legend(loc="upper left")
    ax.set_title("GPU streaming prefill of host-resident MoE weights (DSV4-Flash, 2 x RTX 3090)\nfull 16K prompt: chunked 213 -> layer-major + EP + prefetch 334.5 tok/s (+57%)")
    for x, value in zip(ubatch, gpu):
        offset = (0, -18) if x == 8192 else (0, 8)
        ax.annotate(f"{value:.0f}", (x, value), xytext=offset, textcoords="offset points", ha="center", color=ORANGE)
    for x, value in zip(ubatch, cpu):
        ax.annotate(f"{value:.0f}", (x, value), xytext=(0, -16), textcoords="offset points", ha="center", color=GRAY)
    for x, g, c in zip(ubatch, gpu, cpu):
        ax.text(x, (g + c) / 2, f"{g / c:.2f} x", ha="center", fontweight="bold")
    finish(fig, "gpu-prefill-streaming.png")


def remote_ep():
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.1))
    pairs = [
        ([24.13, 40.59], ["1 machine\n2 NUMA workers", "2 machines\n4 NUMA workers"], "GLM-5.2 MoE pp512\ntrue expert parallelism", "tok/s"),
        ([235.37, 269.36], ["UB64", "UB256"], "DSV4-Flash 16K prefill\n4-worker RDMA EP", "tok/s"),
        ([58.0, 11.5], ["TCP", "RDMA\nRoCEv2"], "Cross-machine RPC RTT\n64 B message", "us"),
    ]
    for ax, (values, labels, title, ylabel) in zip(axes, pairs):
        bars = ax.bar(labels, values, color=[GRAY, PURPLE], width=0.55)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_ylim(0, max(values) * 1.18)
        ax.grid(axis="y", alpha=0.25)
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, value * 1.03, f"{value:g}", ha="center", fontweight="bold")
    axes[0].text(0.5, 0.68, "1.682 x", transform=axes[0].transAxes, ha="center", color=PURPLE, fontsize=13, fontweight="bold")
    axes[1].text(0.5, 0.74, "+14.4%", transform=axes[1].transAxes, ha="center", color=PURPLE, fontsize=13, fontweight="bold")
    fig.subplots_adjust(top=0.72, wspace=0.35)
    fig.suptitle("Multi-machine expert parallelism over a 100 GbE direct link (TCP fallback available)", y=0.98)
    finish(fig, "remote-ep.png")


if __name__ == "__main__":
    evolution()
    numa_tp()
    formats()
    gpu_prefill()
    remote_ep()
