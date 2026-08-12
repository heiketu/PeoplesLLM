#!/usr/bin/env python3

"""Deterministic invariant checks for ep-map-from-freq.py."""

from __future__ import annotations

import importlib.util
from pathlib import Path


MODULE_PATH = Path(__file__).with_name("ep-map-from-freq.py")
SPEC = importlib.util.spec_from_file_location("ep_map_from_freq", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
epmap = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(epmap)


def main() -> int:
    n_expert = 16
    n_workers = 4
    # Make modulo ownership deliberately poor in different layers, while also
    # retaining enough cross-layer tension to exercise the pair-swap pass.
    freq = [
        [
            float(50 + expert if expert % 4 == 0 else 1),
            float(40 + expert if expert % 4 == 1 else 2),
            float(30 + expert if expert % 4 == 2 else 3),
            float(20 + expert if expert % 4 == 3 else 4),
        ]
        for expert in range(n_expert)
    ]
    totals = [sum(freq[e][layer] for e in range(n_expert)) for layer in range(4)]

    owner, primary_loads = epmap.balanced_primary(freq, n_workers)
    assert sorted(owner) == [0] * 4 + [1] * 4 + [2] * 4 + [3] * 4
    modulo_loads = epmap.loads_for([e % n_workers for e in range(n_expert)], freq, n_workers)
    assert epmap.score(primary_loads, totals) < epmap.score(modulo_loads, totals)

    replicas, replica_loads = epmap.add_hot_replicas(owner, freq, n_workers, 2)
    assert all(len(worker) <= 2 for worker in replicas)
    assert all(owner[e] != worker for worker, ids in enumerate(replicas) for e in ids)
    assert len(set().union(*replicas)) == sum(len(ids) for ids in replicas)
    assert epmap.score(replica_loads, totals) <= epmap.score(primary_loads, totals)

    replicas4, replica4_loads = epmap.add_hot_replicas(
        owner, freq, n_workers, 3, max_holders=4)
    assert all(len(worker) <= 3 for worker in replicas4)
    assert all(owner[e] != worker for worker, ids in enumerate(replicas4) for e in ids)
    for expert in range(n_expert):
        assert 1 + sum(expert in ids for ids in replicas4) <= 4
    # The staged pass first reaches the complete two-holder local optimum and
    # accepts higher-order copies only when they improve it further.
    assert epmap.score(replica4_loads, totals) <= epmap.score(replica_loads, totals)

    assert epmap.compress_ids([7, 3, 4, 5, 10]) == "3-5,7,10"
    print("ep-map-from-freq-test: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
