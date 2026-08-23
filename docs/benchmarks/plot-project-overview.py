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
    rows = [
        ("MXFP4", 145.3, 307.58, 3.5830),
        ("UDNL_W4 + arec", 146.4, 366.58, 3.7997),
        ("UDNL_MX + arec", 116.1, 384.87, 4.6274),
        ("E4A", 147.2, 370.0, 3.5830),
        ("Q3_R (MoE)", 114.1, 174.60, 4.7636),
        ("IQ2_XXS (file named Q2_K)", 90.9, 223.60, 6.6338),
    ]
    fig, ax = plt.subplots(figsize=(8.7, 5.1))
    sizes = [row[1] for row in rows]
    speeds = [row[2] for row in rows]
    ppls = [row[3] for row in rows]
    scatter = ax.scatter(sizes, speeds, c=ppls, s=170, cmap="RdYlGn_r", edgecolor="black", vmin=3.4, vmax=6.8, zorder=3)
    offsets = {
        "MXFP4": (-10, -24),
        "UDNL_W4 + arec": (-12, 14),
        "UDNL_MX + arec": (0, 12),
        "E4A": (-10, -24),
        "Q3_R (MoE)": (0, -24),
        "IQ2_XXS (file named Q2_K)": (10, 12),
    }
    for name, size, speed, ppl in rows:
        dx, dy = offsets[name]
        ha = "right" if dx < 0 else ("left" if dx > 0 else "center")
        ax.annotate(f"{name}\n{size:.1f} GiB, PPL {ppl:.2f}", (size, speed), xytext=(dx, dy), textcoords="offset points", ha=ha, fontsize=9)
    ax.set_xlim(80, 160)
    ax.set_ylim(120, 430)
    ax.set_xlabel("Model size (GiB)")
    ax.set_ylabel("pp2048 (tok/s)")
    ax.grid(alpha=0.25)
    colorbar = fig.colorbar(scatter, ax=ax)
    colorbar.set_label("Perplexity (lower is better)")
    ax.set_title("MoE weight formats: size, prefill speed, and quality\nsmaller files are not automatically faster")
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
