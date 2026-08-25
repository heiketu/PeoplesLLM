#!/usr/bin/env python3
"""Build fixed-size sparse EP ownership maps from activation-frequency CSV.

The input is the CSV emitted by GGML_REMOTE_EP_FREQ_FILE:

    layer,expert,count

Primary ownership is an exact, non-overlapping cover with the same number of
experts on every worker.  A multidimensional greedy pass balances every layer,
then pairwise swaps improve the worst normalized per-layer load.  Optional
secondary copies are a max-effort hint: each worker receives up to N extra hot
experts, while the reported expected load assumes a replicated expert splits
its traffic equally among its holders.  The runtime dealer still makes the
final per-token least-load decision.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Iterable


def read_freq(path: Path, n_expert: int | None) -> tuple[list[int], list[list[float]]]:
    rows: list[tuple[int, int, float]] = []
    layers: set[int] = set()
    max_expert = -1
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        required = {"layer", "expert", "count"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError("CSV must have layer,expert,count columns")
        for row in reader:
            layer = int(row["layer"])
            expert = int(row["expert"])
            count = float(row["count"])
            if layer < 0 or expert < 0 or count < 0:
                raise ValueError("layer, expert, and count must be non-negative")
            rows.append((layer, expert, count))
            layers.add(layer)
            max_expert = max(max_expert, expert)

    if not rows:
        raise ValueError("frequency CSV is empty")
    if n_expert is None:
        n_expert = max_expert + 1
    if n_expert < 1 or max_expert >= n_expert:
        raise ValueError(f"--experts {n_expert} does not cover expert id {max_expert}")

    layer_ids = sorted(layers)
    layer_pos = {layer: i for i, layer in enumerate(layer_ids)}
    freq = [[0.0] * len(layer_ids) for _ in range(n_expert)]
    seen: set[tuple[int, int]] = set()
    for layer, expert, count in rows:
        key = (layer, expert)
        if key in seen:
            raise ValueError(f"duplicate row for layer {layer}, expert {expert}")
        seen.add(key)
        freq[expert][layer_pos[layer]] = count
    return layer_ids, freq


def score(loads: list[list[float]], totals: list[float]) -> tuple[float, float]:
    n_workers = len(loads)
    ratios: list[float] = []
    squared = 0.0
    for layer, total in enumerate(totals):
        if total <= 0:
            continue
        ideal = total / n_workers
        for worker in range(n_workers):
            ratio = loads[worker][layer] / ideal
            ratios.append(ratio)
            squared += (ratio - 1.0) ** 2
    return (max(ratios, default=0.0), squared)


def add_vec(dst: list[float], src: list[float], scale: float = 1.0) -> None:
    for i, value in enumerate(src):
        dst[i] += scale * value


def loads_for(owner: list[int], freq: list[list[float]], n_workers: int) -> list[list[float]]:
    loads = [[0.0] * len(freq[0]) for _ in range(n_workers)]
    for expert, worker in enumerate(owner):
        add_vec(loads[worker], freq[expert])
    return loads


def balanced_primary(freq: list[list[float]], n_workers: int) -> tuple[list[int], list[list[float]]]:
    n_expert = len(freq)
    if n_expert % n_workers != 0:
        raise ValueError("expert count must be divisible by worker count for equal-size primary shards")
    capacity = n_expert // n_workers
    totals = [sum(freq[e][layer] for e in range(n_expert)) for layer in range(len(freq[0]))]
    norm_weight = [
        sum(freq[e][layer] / totals[layer] for layer in range(len(totals)) if totals[layer] > 0)
        for e in range(n_expert)
    ]
    order = sorted(range(n_expert), key=lambda e: (-norm_weight[e], -sum(freq[e]), e))

    owner = [-1] * n_expert
    count = [0] * n_workers
    loads = [[0.0] * len(totals) for _ in range(n_workers)]
    for expert in order:
        best: tuple[tuple[float, float], int, int] | None = None
        for worker in range(n_workers):
            if count[worker] >= capacity:
                continue
            add_vec(loads[worker], freq[expert])
            candidate = (score(loads, totals), count[worker], worker)
            add_vec(loads[worker], freq[expert], -1.0)
            if best is None or candidate < best:
                best = candidate
        assert best is not None
        worker = best[2]
        owner[expert] = worker
        count[worker] += 1
        add_vec(loads[worker], freq[expert])

    # Best-improvement pair swaps preserve exact shard cardinality.  The first
    # score component targets the slowest worker at the worst layer; the second
    # smooths all remaining layer/worker deviations.
    ideals = [total / n_workers if total > 0 else 0.0 for total in totals]
    current = score(loads, totals)
    for _ in range(128):
        best_score = current
        best_pair: tuple[int, int] | None = None

        # A swap changes only two worker rows. Cache the contribution of the
        # unaffected rows once per worker pair, then score a candidate in
        # O(layers) instead of rescanning O(workers * layers). This matters for
        # full-model profiles (43+ MoE layers) while preserving the exact score.
        row_squared = [0.0] * n_workers
        pair_unaffected_peak: dict[tuple[int, int], float] = {}
        for worker in range(n_workers):
            row_squared[worker] = sum(
                (loads[worker][layer] / ideals[layer] - 1.0) ** 2
                for layer in range(len(totals)) if ideals[layer] > 0)
        for wa in range(n_workers):
            for wb in range(wa + 1, n_workers):
                pair_unaffected_peak[(wa, wb)] = max(
                    (loads[worker][layer] / ideals[layer]
                     for worker in range(n_workers) if worker != wa and worker != wb
                     for layer in range(len(totals)) if ideals[layer] > 0),
                    default=0.0)

        for a in range(n_expert):
            wa = owner[a]
            for b in range(a + 1, n_expert):
                wb = owner[b]
                if wa == wb:
                    continue
                pair = (min(wa, wb), max(wa, wb))
                peak = pair_unaffected_peak[pair]
                squared = current[1] - row_squared[wa] - row_squared[wb]
                for layer, ideal in enumerate(ideals):
                    if ideal <= 0:
                        continue
                    ratio_a = (loads[wa][layer] - freq[a][layer] + freq[b][layer]) / ideal
                    ratio_b = (loads[wb][layer] - freq[b][layer] + freq[a][layer]) / ideal
                    peak = max(peak, ratio_a, ratio_b)
                    squared += (ratio_a - 1.0) ** 2 + (ratio_b - 1.0) ** 2
                candidate = (peak, squared)
                if candidate < best_score:
                    best_score = candidate
                    best_pair = (a, b)
        if best_pair is None:
            break
        a, b = best_pair
        wa, wb = owner[a], owner[b]
        add_vec(loads[wa], freq[a], -1.0)
        add_vec(loads[wb], freq[b], -1.0)
        add_vec(loads[wa], freq[b])
        add_vec(loads[wb], freq[a])
        owner[a], owner[b] = wb, wa
        current = best_score
    return owner, loads


def add_hot_replicas(
    owner: list[int],
    freq: list[list[float]],
    n_workers: int,
    extra_per_worker: int,
    max_holders: int = 2,
) -> tuple[list[set[int]], list[list[float]]]:
    replicas = [set() for _ in range(n_workers)]
    loads = loads_for(owner, freq, n_workers)
    if extra_per_worker <= 0:
        return replicas, loads

    totals = [sum(freq[e][layer] for e in range(len(freq))) for layer in range(len(freq[0]))]
    holders = [{primary} for primary in owner]
    current = score(loads, totals)
    # Finish all beneficial second-holder placements before considering third
    # holders, and so on.  Mixing holder counts in one greedy pass can take an
    # attractive third copy too early and strand the much broader two-holder
    # solution in a local optimum.
    for target_holders in range(2, max_holders + 1):
        for _ in range(n_workers * extra_per_worker):
            best: tuple[tuple[float, float], float, int, int] | None = None
            for expert in range(len(freq)):
                n_holders = len(holders[expert])
                if n_holders != target_holders - 1:
                    continue
                for worker in range(n_workers):
                    if worker in holders[expert] or len(replicas[worker]) >= extra_per_worker:
                        continue
                    # Equal-split traffic is only an expectation used to choose
                    # a useful additional holder.  Going from H to H+1 holders
                    # moves 1/(H*(H+1)) of the expert's traffic off every old
                    # holder and gives 1/(H+1) to the new holder.  The online
                    # dealer uses the actual per-token least-load plan and is
                    # free to do better.
                    old_delta = -1.0 / (n_holders * (n_holders + 1))
                    new_delta =  1.0 / (n_holders + 1)
                    for old_worker in holders[expert]:
                        add_vec(loads[old_worker], freq[expert], old_delta)
                    add_vec(loads[worker], freq[expert], new_delta)
                    candidate = (score(loads, totals), -sum(freq[expert]), expert, worker)
                    for old_worker in holders[expert]:
                        add_vec(loads[old_worker], freq[expert], -old_delta)
                    add_vec(loads[worker], freq[expert], -new_delta)
                    if best is None or candidate < best:
                        best = candidate
            if best is None or best[0] >= current:
                break
            current, _, expert, worker = best
            n_holders = len(holders[expert])
            old_delta = -1.0 / (n_holders * (n_holders + 1))
            new_delta =  1.0 / (n_holders + 1)
            for old_worker in holders[expert]:
                add_vec(loads[old_worker], freq[expert], old_delta)
            add_vec(loads[worker], freq[expert], new_delta)
            replicas[worker].add(expert)
            holders[expert].add(worker)
    return replicas, loads


def compress_ids(ids: Iterable[int]) -> str:
    values = sorted(ids)
    if not values:
        return ""
    parts: list[str] = []
    first = last = values[0]
    for value in values[1:]:
        if value == last + 1:
            last = value
            continue
        parts.append(str(first) if first == last else f"{first}-{last}")
        first = last = value
    parts.append(str(first) if first == last else f"{first}-{last}")
    return ",".join(parts)


def metric(loads: list[list[float]], totals: list[float]) -> dict[str, float]:
    peak, squared = score(loads, totals)
    return {"peak_layer_ratio": peak, "normalized_rms": math.sqrt(squared / (len(loads) * len(totals)))}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path, help="GGML_REMOTE_EP_FREQ_FILE CSV")
    parser.add_argument("-w", "--workers", type=int, default=4)
    parser.add_argument("-e", "--experts", type=int, default=None,
                        help="full expert count (needed when trailing zero experts are absent)")
    parser.add_argument("--extra-per-worker", type=int, default=0,
                        help="max-effort secondary hot-expert copies on each worker")
    parser.add_argument("--max-holders", type=int, default=2,
                        help="maximum total holders per expert (primary included)")
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    parser.add_argument("--output", type=Path, help="write output to a file instead of stdout")
    args = parser.parse_args()
    if args.workers < 1 or args.workers > 63 or args.extra_per_worker < 0 or \
            args.max_holders < 1 or args.max_holders > args.workers:
        parser.error("workers must be in [1,63], extra-per-worker non-negative, and max-holders in [1,workers]")

    layers, freq = read_freq(args.csv, args.experts)
    owner, primary_loads = balanced_primary(freq, args.workers)
    replicas, expected_loads = add_hot_replicas(
        owner, freq, args.workers, args.extra_per_worker, args.max_holders)
    totals = [sum(freq[e][layer] for e in range(len(freq))) for layer in range(len(layers))]
    modulo_owner = [e % args.workers for e in range(len(freq))]

    worker_ids: list[list[int]] = []
    for worker in range(args.workers):
        ids = {e for e, primary in enumerate(owner) if primary == worker}
        ids.update(replicas[worker])
        worker_ids.append(sorted(ids))

    result = {
        "layers": layers,
        "n_expert": len(freq),
        "n_workers": args.workers,
        "extra_per_worker": args.extra_per_worker,
        "max_holders": args.max_holders,
        "metrics": {
            "modulo": metric(loads_for(modulo_owner, freq, args.workers), totals),
            "balanced_primary": metric(primary_loads, totals),
            "expected_with_replicas": metric(expected_loads, totals),
        },
        "workers": [
            {
                "worker": worker,
                "primary_count": sum(primary == worker for primary in owner),
                "replica_count": len(replicas[worker]),
                "experts": worker_ids[worker],
                "expert_list": compress_ids(worker_ids[worker]),
            }
            for worker in range(args.workers)
        ],
    }
    if args.json:
        output = json.dumps(result, indent=2, sort_keys=True) + "\n"
    else:
        lines = [f"layers={layers} experts={len(freq)} workers={args.workers}"]
        for name, values in result["metrics"].items():
            lines.append(f"{name}: peak_layer_ratio={values['peak_layer_ratio']:.6f} "
                         f"normalized_rms={values['normalized_rms']:.6f}")
        for worker in result["workers"]:
            lines.append(f"worker{worker['worker']} primary={worker['primary_count']} "
                         f"replicas={worker['replica_count']} --expert-list {worker['expert_list']}")
        output = "\n".join(lines) + "\n"
    if args.output:
        args.output.write_text(output)
    else:
        print(output, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
