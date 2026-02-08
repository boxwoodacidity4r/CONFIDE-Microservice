"""Diagnose Plants temporal (S_temp) quality: glue classes + top inter-service pairs.

Goal
----
Help raise Plants `temp` intra/inter ratio by identifying:
1) Glue / hub classes: classes with largest row-sum / degree / weighted degree in S_temp.
2) Strongest inter-service pairs: high similarity between classes that belong to different
   ground-truth microservices.

Inputs (repo conventions)
------------------------
- data/processed/temporal/plants_S_temp.npy
- data/processed/fusion/plants_class_order.json
- data/processed/groundtruth/plants_ground_truth.json

Usage (PowerShell)
-----------------
  python scripts/temporal/diagnose_plants_temp_glue.py
  python scripts/temporal/diagnose_plants_temp_glue.py --top-k 15 --min-sim 0.05

Notes
-----
- We treat S_temp as an undirected weighted graph using the max(S[i,j], S[j,i]).
- "Degree" here means number of non-zero off-diagonal edges.
- "Weighted degree" is sum of off-diagonal similarities.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Pair:
    i: int
    j: int
    sim: float


def _load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _labels_from_gt(gt: Dict[str, int], order: List[str]) -> List[int]:
    # gt: fqcn -> service_id
    return [int(gt.get(c, -1)) for c in order]


def _graph_stats(S: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return (degree, weighted_degree) per node (off-diagonal)."""
    A = S.copy()
    np.fill_diagonal(A, 0.0)
    # symmetrize by max to avoid directionality artifacts
    A = np.maximum(A, A.T)
    degree = (A > 0).sum(axis=1)
    wdeg = A.sum(axis=1)
    return degree.astype(int), wdeg.astype(float)


def _top_pairs(
    S: np.ndarray,
    labels: List[int],
    *,
    top_k: int,
    min_sim: float,
    only_inter: bool,
) -> List[Pair]:
    A = S.copy()
    np.fill_diagonal(A, 0.0)
    A = np.maximum(A, A.T)

    n = A.shape[0]
    pairs: List[Pair] = []

    for i in range(n):
        li = labels[i]
        for j in range(i + 1, n):
            sim = float(A[i, j])
            if sim < min_sim:
                continue
            if only_inter:
                lj = labels[j]
                if li == -1 or lj == -1:
                    continue
                if li == lj:
                    continue
            pairs.append(Pair(i=i, j=j, sim=sim))

    pairs.sort(key=lambda p: p.sim, reverse=True)
    return pairs[:top_k]


def main() -> int:
    ap = argparse.ArgumentParser(description="Diagnose Plants S_temp glue classes and inter-service pairs")
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--min-sim", type=float, default=0.02)
    args = ap.parse_args()

    S_path = ROOT / "data" / "processed" / "temporal" / "plants_S_temp.npy"
    order_path = ROOT / "data" / "processed" / "fusion" / "plants_class_order.json"
    gt_path = ROOT / "data" / "processed" / "groundtruth" / "plants_ground_truth.json"

    if not S_path.is_file():
        raise SystemExit(f"Missing: {S_path}")
    if not order_path.is_file():
        raise SystemExit(f"Missing: {order_path}")
    if not gt_path.is_file():
        raise SystemExit(f"Missing: {gt_path}")

    order: List[str] = _load_json(order_path)
    gt: Dict[str, int] = _load_json(gt_path)
    labels = _labels_from_gt(gt, order)

    S = np.load(S_path)
    if S.shape[0] != len(order) or S.shape[1] != len(order):
        raise SystemExit(f"Shape mismatch: S={S.shape}, order={len(order)}")

    degree, wdeg = _graph_stats(S)

    # Top glue classes by weighted degree
    idxs = np.argsort(-wdeg)

    print("== Plants S_temp hub/glue candidates (by weighted degree) ==")
    print("rank | idx | deg | wdeg | gt_label | class")
    for rank, i in enumerate(idxs[: args.top_k], start=1):
        print(
            f"{rank:>4} | {int(i):>3} | {int(degree[i]):>3} | {wdeg[i]:>6.3f} | {labels[i]:>8} | {order[int(i)]}"
        )

    # Pairs: strongest inter-service similarities
    top_inter = _top_pairs(S, labels, top_k=args.top_k, min_sim=args.min_sim, only_inter=True)
    print("\n== Strongest inter-service pairs (by similarity) ==")
    print("rank | sim | (i,j) | gt(i),gt(j) | class(i) || class(j)")
    for rank, p in enumerate(top_inter, start=1):
        print(
            f"{rank:>4} | {p.sim:>5.3f} | ({p.i:>2},{p.j:>2}) | {labels[p.i]:>5},{labels[p.j]:>5} | {order[p.i]} || {order[p.j]}"
        )

    # Also show strongest pairs overall (can reveal single-flow cliques)
    top_all = _top_pairs(S, labels, top_k=args.top_k, min_sim=args.min_sim, only_inter=False)
    print("\n== Strongest pairs overall (by similarity) ==")
    print("rank | sim | (i,j) | gt(i),gt(j) | class(i) || class(j)")
    for rank, p in enumerate(top_all, start=1):
        print(
            f"{rank:>4} | {p.sim:>5.3f} | ({p.i:>2},{p.j:>2}) | {labels[p.i]:>5},{labels[p.j]:>5} | {order[p.i]} || {order[p.j]}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
