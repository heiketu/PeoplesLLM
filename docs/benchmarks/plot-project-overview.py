#!/usr/bin/env python3
"""Regenerate the public project overview charts.

Requires matplotlib and numpy.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


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
        "MXFP4\nbaseline",
        "DSpark\ntune (n2/p0)",
        "Raw GPU\nop fusion",
        "E4A raw\nnative layout",
        "DSpark claim +\ntail batching",
        "E4A raw\nstrict K24 hot",
    ]
    x = np.arange(len(stages))
    pp = [309, np.nan, np.nan, 362.92, np.nan, 361.47]
    raw = [24.9, np.nan, 25.9, 24.83, np.nan, 30.23]
    spec = [23.9, 26.8, np.nan, np.nan, 30.1, np.nan]

    fig, left = plt.subplots(figsize=(10, 4.8))
    right = left.twinx()
    left.plot(x, pp, marker="o", color=GREEN, linewidth=2.2, label="PP2048 (prefill, left)")
    right.plot(x, raw, marker="s", color=BLUE, linewidth=2.2, label="TG512 raw/no-DSpark (right)")
    right.plot(x, spec, marker="^", linestyle="--", color=ORANGE, linewidth=2.2, label="TG512 DSpark speculative (right)")
    left.set_ylim(250, 430)
    right.set_ylim(20, 33)
    left.set_ylabel("Prefill pp2048 (tok/s)", color=GREEN)
    right.set_ylabel("Decode TG512 (raw/spec; see legend)")
    left.set_xticks(x, stages)
    left.grid(alpha=0.25)
    lines = left.lines + right.lines
    left.legend(lines, [line.get_label() for line in lines], loc="upper left")
    left.set_title("DSV4-Flash independently measured performance tracks (2 x Xeon ICX-SP + 2 x RTX 3090)\nblank points are untested cross-combinations; tracks must not be multiplied")
    for px, py, label, offset in [(0, 309, "309", (-12, 10)), (3, 362.92, "362.92", (0, 10)), (5, 361.47, "361.47", (12, -16))]:
        left.annotate(label, (px, py), xytext=offset, textcoords="offset points", ha="center", color=GREEN, fontweight="bold")
    for px, py, label, offset in [(0, 24.9, "24.9", (0, -18)), (2, 25.9, "25.9", (0, -16)), (3, 24.83, "24.83", (-12, -16)), (5, 30.23, "30.23", (0, -16))]:
        right.annotate(label, (px, py), xytext=offset, textcoords="offset points", ha="center", color=BLUE, fontweight="bold")
    for px, py, label in [(0, 23.9, "23.9"), (1, 26.8, "26.8"), (4, 30.1, "30.1")]:
        right.annotate(label, (px, py), xytext=(0, 10), textcoords="offset points", ha="center", color=ORANGE, fontweight="bold")
    finish(fig, "evolution-staircase.png")


def numa_tp():
    labels = ["hybrid raw TG512\n(no DSpark)", "hybrid PP2048", "pure-CPU raw TG128\n(no DSpark)", "pure-CPU PP512"]
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
    ax.set_title("Intra-device NUMA tensor parallelism, DSV4-Flash MXFP4 (ABAB verified)\nraw/no-DSpark decode; local+bound 313.5 GB/s vs interleave 122 GB/s (2.57 x)")
    for bar, value in zip(bars_off, off):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 4, f"{value:.2f}", ha="center")
    for bar, value, change in zip(bars_on, on, changes):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 4, f"{value:.2f}\n{change}", ha="center", color=BLUE, fontweight="bold")
    finish(fig, "numa-tp-onoff.png")


def formats():
    # Formal performance sweep, 2026-08-24. Values are intentionally embedded
    # so the public chart generator does not depend on private benchmark artifacts.
    rows = [
        ("UD-IQ1_S",      76.87065544724464, 251.18, 29.45, "ud"),
        ("UD-IQ1_M",      80.93315544724464, 241.38, 29.03, "ud"),
        ("UD-IQ2_XXS",    84.62065544724464, 247.17, 28.89, "ud"),
        ("UD-IQ2_M",      84.68230095505714, 246.83, 29.24, "ud"),
        ("UD-Q2_K_XL",    90.18230098485947, 232.22, 28.03, "ud"),
        ("UD-IQ3_XXS",    97.05112132430077, 208.96, 26.74, "ud"),
        ("UD-IQ3_S",     108.09799629449844, 254.48, 23.99, "ud"),
        ("UDNL_MX",       116.12836688756943, 418.95, 26.90, "xllama"),
        ("UD-Q3_K_M",    119.28238350152970, 207.20, 25.49, "ud"),
        ("UD-Q3_K_XL",   119.40182167291641, 207.69, 25.34, "ud"),
        # Weighted across the 3-run formal batch and the 5-run confirmation batch.
        ("UD-IQ4_XS",    127.27682167291641, 215.93, 21.10, "ud"),
        ("UD-IQ4_NL",    127.27682167291641, 223.53, 21.36, "ud"),
        ("UD-Q4_K_XL",   144.44369927048683, 314.15, 25.65, "ud"),
        ("UD-Q8_K_XL",   150.75282707810402, 306.17, 21.39, "ud"),
        ("MXFP4",        145.26439723372460, 312.48, 26.87, "baseline"),
        ("UDNL_W4",      146.36274161934853, 370.87, 25.10, "xllama"),
        ("E4A",          145.26439723372460, 362.92, 24.83, "xllama"),
    ]
    rows.sort(key=lambda row: (row[1], row[0]))

    def smaller_but_slower(metric):
        result = []
        for row in rows:
            size = row[1]
            speed = row[metric]
            result.append(any(other[1] > size + 0.05 and other[metric] > speed * 1.03 for other in rows))
        return result

    sizes = np.array([row[1] for row in rows])
    pp = np.array([row[2] for row in rows])
    tg = np.array([row[3] for row in rows])
    kinds = np.array([row[4] for row in rows])
    counter_pp = smaller_but_slower(2)
    counter_tg = smaller_but_slower(3)

    pp_offsets = {
        "UD-IQ1_S": (-8, 14), "UD-IQ1_M": (-18, -19),
        "UD-IQ2_XXS": (27, 13), "UD-IQ2_M": (28, -18),
        "UD-Q2_K_XL": (0, 10), "UD-IQ3_XXS": (0, -17),
        "UD-IQ3_S": (-8, 10),
        "UDNL_MX": (7, 10),
        "UD-Q3_K_M": (-46, -18), "UD-Q3_K_XL": (11, 9),
        "UD-IQ4_XS": (-49, -18), "UD-IQ4_NL": (10, 9),
        "UD-Q4_K_XL": (-58, -18), "MXFP4": (-26, 10),
        "E4A": (-21, -18), "UDNL_W4": (20, 10), "UD-Q8_K_XL": (0, -18),
    }
    tg_offsets = {
        "UD-IQ1_S": (-6, 13), "UD-IQ1_M": (-24, -19),
        "UD-IQ2_XXS": (27, -13), "UD-IQ2_M": (28, 13),
        "UD-Q2_K_XL": (0, 10), "UD-IQ3_XXS": (-3, -17),
        "UD-IQ3_S": (-10, -17),
        "UDNL_MX": (-8, 10),
        "UD-Q3_K_M": (-46, 10), "UD-Q3_K_XL": (10, -17),
        "UD-IQ4_XS": (-48, -17), "UD-IQ4_NL": (11, 9),
        "UD-Q4_K_XL": (-58, -18), "MXFP4": (-24, 10),
        "E4A": (-22, -18), "UDNL_W4": (20, 10), "UD-Q8_K_XL": (0, -18),
    }

    fig, axes = plt.subplots(2, 1, figsize=(13.2, 9.7), sharex=True)
    trend_mask = kinds != "xllama"
    for ax, values, counter, ylabel, offsets, trend_label_x, trend_ha in (
        (axes[0], pp, counter_pp, "pp2048 (tok/s)", pp_offsets, 0.015, "left"),
        (axes[1], tg, counter_tg, "tg512 raw/no-DSpark (tok/s)", tg_offsets, 0.985, "right"),
    ):
        trend_sizes = sizes[trend_mask]
        trend_values = values[trend_mask]
        trend_coeff = np.polyfit(trend_sizes, trend_values, 1)
        trend_x = np.linspace(sizes.min(), sizes.max(), 200)
        fitted = np.polyval(trend_coeff, trend_sizes)
        residual = np.sum((trend_values - fitted) ** 2)
        total = np.sum((trend_values - trend_values.mean()) ** 2)
        r_squared = 1.0 - residual / total
        ax.plot(trend_x, np.polyval(trend_coeff, trend_x), color=GRAY, linewidth=1.8, linestyle="--", alpha=0.85, zorder=1)
        for row, size, speed, kind, is_counterexample in zip(rows, sizes, values, kinds, counter):
            name = row[0]
            if kind == "ud":
                marker, face, point_size, width = "o", BLUE, 58, 0.65
            elif kind == "baseline":
                marker, face, point_size, width = "X", "black", 100, 0.8
            else:
                marker, face, point_size, width = "*", ORANGE, 190, 1.0
            ax.scatter(size, speed, s=point_size, marker=marker, color=face, edgecolor="black", linewidth=width, zorder=3)
            if is_counterexample:
                ax.scatter(size, speed, s=130 if kind != "xllama" else 245, marker="D", facecolors="none",
                           edgecolors="#d62728", linewidth=1.45, zorder=4)
            dx, dy = offsets[name]
            text_color = "#8f3f1f" if kind == "xllama" else ("black" if kind == "baseline" else "#244b73")
            ax.annotate(f"{name}  {speed:.2f}", (size, speed), xytext=(dx, dy), textcoords="offset points",
                        ha="center", va="center", fontsize=7.2, color=text_color,
                        fontweight="bold" if kind != "ud" else "normal",
                        arrowprops=dict(arrowstyle="-", color=text_color, alpha=0.45, linewidth=0.55) if abs(dx) > 15 else None)
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25)
        ax.text(trend_label_x, 0.95, f"UD + MXFP4 OLS: {trend_coeff[0]:+.3f} tok/s/GiB, R²={r_squared:.2f}",
                transform=ax.transAxes, ha=trend_ha, va="top", color="#555b63", fontsize=8.8, fontweight="bold")

    axes[0].set_ylim(185, 445)
    axes[1].set_ylim(19.8, 30.3)
    axes[1].set_xlim(73.5, 154.0)
    axes[1].set_xlabel("Model size (GiB)")
    fig.suptitle(
        "DSV4-Flash formal format sweep (matched performance recipe; corrected UDNL_MX rerun)\n"
        "red open diamond: a smaller format is at least 3% slower than a larger format",
        y=0.995,
    )
    legend = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=BLUE, markeredgecolor="black", markersize=7, label="Unsloth UD"),
        Line2D([0], [0], marker="X", color="none", markerfacecolor="black", markeredgecolor="black", markersize=8, label="MXFP4 anchor"),
        Line2D([0], [0], marker="*", color="none", markerfacecolor=ORANGE, markeredgecolor="black", markersize=13, label="PeoplesLLM format"),
        Line2D([0], [0], marker="D", color="none", markerfacecolor="none", markeredgecolor="#d62728", markersize=8, label="smaller but >=3% slower"),
        Line2D([0], [0], color=GRAY, linestyle="--", linewidth=1.8, label="OLS on UD + MXFP4 only"),
    ]
    fig.legend(handles=legend, ncol=5, loc="upper center", bbox_to_anchor=(0.5, 0.948), fontsize=8, framealpha=0.94)
    fig.text(0.5, 0.008, "UDNL_MX is the corrected imatrix build: 116.13 GiB, pp 418.95, tg 26.90. Its PPL 4.6047 is shown separately. UD-IQ4_XS is repeat-weighted.",
             ha="center", va="bottom", fontsize=8, color="#555b63")
    fig.tight_layout(rect=(0, 0.035, 1, 0.905))
    finish(fig, "quant-formats.png")


def udnl_mx_tradeoff():
    labels = ["Q3_K_XL", "UDNL_MX\ncorrected"]
    colors = [BLUE, ORANGE]
    metrics = [
        ("Model size", [119.40, 116.13], [0, 0], "GiB", "lower is better", False),
        ("PP2048", [207.69, 418.95], [0, 3.50], "tok/s", "2.02 x faster", True),
        ("TG512 raw\n(no DSpark)", [25.34, 26.90], [0, 0.25], "tok/s", "+6.2%", True),
        ("WikiText-2 PPL", [4.0189, 4.6047], [0, 0.1782], "PPL", "+14.6% worse", False),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(13.2, 3.9))
    for ax, (title, values, errors, unit, callout, higher_is_better) in zip(axes, metrics):
        bars = ax.bar(labels, values, color=colors, width=0.58)
        if errors[1] > 0:
            ax.errorbar([1], [values[1]], yerr=[errors[1]], fmt="none", ecolor="black", capsize=4, linewidth=1.0)
        ax.set_title(title)
        ax.set_ylabel(unit)
        ax.set_ylim(0, max(values) * 1.23)
        ax.grid(axis="y", alpha=0.25)
        for index, (bar, value) in enumerate(zip(bars, values)):
            digits = 4 if title.endswith("PPL") else 2
            label_y = value + errors[index] + max(values)*0.025
            ax.text(bar.get_x() + bar.get_width() / 2, label_y, f"{value:.{digits}f}",
                    ha="center", fontsize=9, fontweight="bold")
        callout_color = GREEN if (higher_is_better or title == "Model size") and "worse" not in callout else ORANGE
        ax.text(0.5, 0.94, callout, transform=ax.transAxes, ha="center", va="top",
                fontsize=9, color=callout_color, fontweight="bold")
    fig.suptitle(
        "Corrected raw UDNL_MX trades a small size reduction and much higher PP for worse quality\n"
        "not recommended as a Q3_K_XL replacement",
        y=1.03,
    )
    fig.tight_layout()
    finish(fig, "udnl-mx-tradeoff.png")


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
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.8))
    axes = axes.ravel()
    pairs = [
        ([22.1, 25.2], ["1 machine\n2 NUMA workers", "2 machines\n4 NUMA workers"], "DSV4-Flash raw TG512 (no DSpark)\nmatched pure EP", "tok/s"),
        ([25.25, 28.85], ["4-worker\nremote-only (A)", "CUDA1 K24 hot\n+ remote cold (B)"], "DSV4-Flash raw TG512 (no DSpark)\nstrict hot + remote bridge", "tok/s"),
        ([64.3, 75.6], ["1 machine\n2 NUMA workers", "2 machines\n4 NUMA workers"], "DSV4-Flash 31-token prompt\nmatched pure EP", "tok/s"),
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
    axes[0].text(0.5, 0.68, "+14.03%", transform=axes[0].transAxes, ha="center", color=PURPLE, fontsize=13, fontweight="bold")
    axes[1].text(0.5, 0.66, "+14.26%", transform=axes[1].transAxes, ha="center", color=PURPLE, fontsize=13, fontweight="bold")
    axes[1].text(0.5, 0.54, "paired PPL -0.85%", transform=axes[1].transAxes, ha="center", color=ORANGE, fontsize=10, fontweight="bold")
    axes[2].text(0.5, 0.68, "+17.57%", transform=axes[2].transAxes, ha="center", color=PURPLE, fontsize=13, fontweight="bold")
    fig.subplots_adjust(top=0.84, wspace=0.28, hspace=0.52)
    fig.suptitle(
        "Independent DSV4-Flash single-slot EP comparisons over a 100 GbE direct link\n"
        "combined hot + remote result is strict-cover REQ4, raw/no-DSpark, and not MAX_EFFORT",
        y=0.985,
    )
    finish(fig, "remote-ep.png")


if __name__ == "__main__":
    evolution()
    numa_tp()
    formats()
    udnl_mx_tradeoff()
    gpu_prefill()
    remote_ep()
