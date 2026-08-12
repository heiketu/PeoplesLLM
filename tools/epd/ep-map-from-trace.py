#!/usr/bin/env python3
"""Add EP expert replicas using real small-batch router co-activation traces.

The aggregate-frequency mapper balances long-run traffic per layer.  Decode
verification uses only a few tokens, however, so its latency is determined by
the most loaded worker in each individual router batch.  This tool consumes
the optional ``GGML_REMOTE_EP_TRACE_ROUTER=1`` debug lines and greedily adds
replicas that reduce the sum of those per-batch critical loads.

The base JSON is an existing map from ``ep-map-from-freq.py``.  Primary
ownership is never removed, and every added copy remains optional at runtime;
therefore coverage and model math are unchanged.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path


TRACE_RE = re.compile(
    r"layer\s+(?P<layer>\d+)\s+sched\s+k=(?P<k>\d+).*?"
    r"n_tokens=(?P<tokens>\d+).*?ids=\[(?P<ids>[0-9,]*)\]"
)
MASK32 = (1 << 32) - 1


@dataclass(frozen=True)
class Batch:
    layer: int
    n_tokens: int
    k: int
    ids: tuple[int, ...]


def read_trace(path: Path, max_tokens: int) -> list[Batch]:
    batches: list[Batch] = []
    with path.open(errors="replace") as stream:
        for line in stream:
            match = TRACE_RE.search(line)
            if match is None:
                continue
            n_tokens = int(match.group("tokens"))
            if max_tokens > 0 and n_tokens > max_tokens:
                continue
            k = int(match.group("k"))
            ids_text = match.group("ids")
            ids = tuple(int(value) for value in ids_text.split(",") if value)
            if len(ids) != n_tokens * k:
                raise ValueError(
                    f"bad trace row: got {len(ids)} ids, expected {n_tokens * k}"
                )
            batches.append(Batch(int(match.group("layer")), n_tokens, k, ids))
    if not batches:
        raise ValueError("no matching router batches in trace")
    return batches


def pick_worker(
    expert: int,
    slot: int,
    token: int,
    owners: set[int],
    load: list[int],
    seen: list[set[int]],
    repeat_cost: int,
) -> int:
    salt = (
        ((expert * 0x9E3779B1) & MASK32)
        ^ ((slot * 0x85EBCA6B) & MASK32)
        ^ ((token * 0xC2B2AE35) & MASK32)
    )
    n_workers = len(load)
    start = salt % n_workers
    best = -1
    for step in range(n_workers):
        worker = (start + step) % n_workers
        if worker not in owners:
            continue
        increment = 1 if repeat_cost == 0 else (
            repeat_cost if expert in seen[worker] else 1000
        )
        score = load[worker] + increment
        best_score = (
            load[best] + (
                1 if repeat_cost == 0 else (
                    repeat_cost if expert in seen[best] else 1000
                )
            )
            if best >= 0 else 0
        )
        if best < 0 or score < best_score:
            best = worker
    if best < 0:
        raise ValueError(f"expert {expert} has no owner")
    return best


def simulate(
    batch: Batch,
    holders: list[set[int]],
    n_workers: int,
    repeat_cost: int,
) -> tuple[list[int], list[int]]:
    load = [0] * n_workers
    seen = [set() for _ in range(n_workers)]
    assignment: list[int] = []
    for token in range(batch.n_tokens):
        for slot in range(batch.k):
            expert = batch.ids[token * batch.k + slot]
            worker = pick_worker(
                expert, slot, token, holders[expert], load, seen, repeat_cost
            )
            load[worker] += (
                1 if repeat_cost == 0 else (
                    repeat_cost if expert in seen[worker] else 1000
                )
            )
            seen[worker].add(expert)
            assignment.append(worker)

    if repeat_cost == 0:
        return load, assignment

    # Match the production dealer's deterministic single-slot refinement. A
    # move is accepted only when it lexicographically lowers the endpoint loads
    # sorted from slowest to fastest.
    counts = [defaultdict(int) for _ in range(n_workers)]
    for pos, worker in enumerate(assignment):
        counts[worker][batch.ids[pos]] += 1

    def better(lhs: list[int], rhs: list[int]) -> bool:
        return sorted(lhs, reverse=True) < sorted(rhs, reverse=True)

    while True:
        best_pos = -1
        best_worker = -1
        best_load = load
        for pos, src in enumerate(assignment):
            expert = batch.ids[pos]
            remove_cost = repeat_cost if counts[src][expert] > 1 else 1000
            for dst in sorted(holders[expert]):
                if dst == src:
                    continue
                add_cost = repeat_cost if counts[dst][expert] > 0 else 1000
                candidate = load.copy()
                candidate[src] -= remove_cost
                candidate[dst] += add_cost
                if better(candidate, best_load):
                    best_pos = pos
                    best_worker = dst
                    best_load = candidate
        if best_worker < 0:
            break
        expert = batch.ids[best_pos]
        src = assignment[best_pos]
        counts[src][expert] -= 1
        counts[best_worker][expert] += 1
        assignment[best_pos] = best_worker
        load = best_load
    return load, assignment


def evaluate(
    batches: list[Batch], holders: list[set[int]], n_workers: int,
    keep_assignments: bool, repeat_cost: int,
) -> tuple[tuple[int, int, int], list[tuple[list[int], list[int]]]]:
    sum_max = 0
    sum_square = 0
    worst = 0
    details: list[tuple[list[int], list[int]]] = []
    for batch in batches:
        load, assignment = simulate(batch, holders, n_workers, repeat_cost)
        critical = max(load)
        sum_max += critical
        sum_square += critical * critical
        worst = max(worst, critical)
        if keep_assignments:
            details.append((load, assignment))
    return (sum_max, sum_square, worst), details


def percentile(values: list[int], fraction: float) -> int:
    ordered = sorted(values)
    pos = max(0, min(len(ordered) - 1, math.ceil(fraction * len(ordered)) - 1))
    return ordered[pos]


def metrics(
    batches: list[Batch], holders: list[set[int]], n_workers: int, repeat_cost: int,
) -> dict[str, float | int]:
    critical: list[int] = []
    normalized: list[float] = []
    for batch in batches:
        load, _ = simulate(batch, holders, n_workers, repeat_cost)
        value = max(load)
        critical.append(value)
        normalized.append(value / (sum(load) / n_workers))
    return {
        "batches": len(batches),
        "mean_critical_work": sum(critical) / len(critical),
        "mean_critical_over_ideal": sum(normalized) / len(normalized),
        "p95_critical_work": percentile(critical, 0.95),
        "p99_critical_work": percentile(critical, 0.99),
        "max_critical_work": max(critical),
    }


def candidate_votes(
    batches: list[Batch],
    details: list[tuple[list[int], list[int]]],
    holders: list[set[int]],
    worker_sizes: list[int],
    target_size: int,
    repeat_cost: int,
) -> dict[tuple[int, int], int]:
    votes: dict[tuple[int, int], int] = defaultdict(int)
    for batch, (load, assignment) in zip(batches, details):
        high = max(load)
        low = min(load)
        if high <= low + max(1, repeat_cost):
            continue
        high_workers = {worker for worker, value in enumerate(load) if value == high}
        low_workers = [
            worker for worker, value in enumerate(load)
            if value == low and worker_sizes[worker] < target_size
        ]
        if not low_workers:
            continue
        weight = high - low
        for pos, owner in enumerate(assignment):
            if owner not in high_workers:
                continue
            expert = batch.ids[pos]
            for worker in low_workers:
                if worker not in holders[expert]:
                    votes[(expert, worker)] += weight
    return votes


def optimize(
    batches: list[Batch],
    holders: list[set[int]],
    worker_experts: list[set[int]],
    target_size: int,
    candidate_width: int,
    repeat_cost: int,
) -> tuple[list[tuple[int, int]], tuple[int, int, int]]:
    n_workers = len(worker_experts)
    additions: list[tuple[int, int]] = []
    current, details = evaluate(batches, holders, n_workers, True, repeat_cost)
    while min(len(experts) for experts in worker_experts) < target_size:
        sizes = [len(experts) for experts in worker_experts]
        votes = candidate_votes(
            batches, details, holders, sizes, target_size, repeat_cost
        )
        ranked = sorted(
            votes,
            key=lambda pair: (-votes[pair], len(holders[pair[0]]), pair[0], pair[1]),
        )[:candidate_width]
        if not ranked:
            break

        best: tuple[tuple[int, int, int], int, int] | None = None
        for expert, worker in ranked:
            holders[expert].add(worker)
            trial, _ = evaluate(
                batches, holders, n_workers, False, repeat_cost
            )
            holders[expert].remove(worker)
            candidate = (trial, expert, worker)
            if best is None or candidate < best:
                best = candidate
        assert best is not None
        score, expert, worker = best
        if score >= current:
            break
        holders[expert].add(worker)
        worker_experts[worker].add(expert)
        additions.append((expert, worker))
        current, details = evaluate(
            batches, holders, n_workers, True, repeat_cost
        )
    return additions, current


def compress_ids(values: set[int]) -> str:
    ids = sorted(values)
    parts: list[str] = []
    first = last = ids[0]
    for value in ids[1:]:
        if value == last + 1:
            last = value
            continue
        parts.append(str(first) if first == last else f"{first}-{last}")
        first = last = value
    parts.append(str(first) if first == last else f"{first}-{last}")
    return ",".join(parts)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    parser.add_argument("base_map", type=Path)
    parser.add_argument("--target-experts-per-worker", type=int, default=96)
    parser.add_argument("--candidate-width", type=int, default=24)
    parser.add_argument("--max-tokens", type=int, default=4,
                        help="ignore larger prefill batches (0 keeps all)")
    parser.add_argument("--repeat-cost", type=int, default=0,
                        help="same-worker repeated expert cost relative to a new stream=1000; "
                             "0 preserves the legacy assignment-count objective")
    args = parser.parse_args()
    if (args.target_experts_per_worker < 1 or args.candidate_width < 1 or
            args.max_tokens < 0 or not 0 <= args.repeat_cost <= 1000):
        parser.error("target/candidate-width must be positive, max-tokens non-negative, "
                     "and repeat-cost in [0, 1000]")

    base = json.loads(args.base_map.read_text())
    n_workers = int(base["n_workers"])
    n_expert = int(base["n_expert"])
    worker_experts = [set(map(int, worker["experts"])) for worker in base["workers"]]
    if len(worker_experts) != n_workers:
        raise ValueError("worker count mismatch in base map")
    holders = [set() for _ in range(n_expert)]
    for worker, experts in enumerate(worker_experts):
        for expert in experts:
            if expert < 0 or expert >= n_expert:
                raise ValueError(f"expert {expert} is out of range")
            holders[expert].add(worker)
    if any(not owners for owners in holders):
        raise ValueError("base map does not cover every expert")

    batches = read_trace(args.trace, args.max_tokens)
    if any(expert >= n_expert for batch in batches for expert in batch.ids):
        raise ValueError("trace contains an expert outside the base map")
    before = metrics(batches, holders, n_workers, args.repeat_cost)
    additions, _ = optimize(
        batches, holders, worker_experts,
        args.target_experts_per_worker, args.candidate_width, args.repeat_cost,
    )
    after = metrics(batches, holders, n_workers, args.repeat_cost)

    result = dict(base)
    result["coactivation_trace"] = {
        "source": str(args.trace),
        "max_tokens": args.max_tokens,
        "target_experts_per_worker": args.target_experts_per_worker,
        "candidate_width": args.candidate_width,
        "repeat_cost": args.repeat_cost,
        "additions": len(additions),
        "metrics_before": before,
        "metrics_after": after,
    }
    result["workers"] = []
    for worker, experts in enumerate(worker_experts):
        primary_count = int(base["workers"][worker].get("primary_count", 0))
        result["workers"].append({
            "worker": worker,
            "primary_count": primary_count,
            "replica_count": len(experts) - primary_count,
            "experts": sorted(experts),
            "expert_list": compress_ids(experts),
        })
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
