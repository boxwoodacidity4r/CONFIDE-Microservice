"""Diagnose Plants S_temp quality issues (glue classes + strongest inter-service pairs).

Usage (from repo root):
  python scripts/temporal/diagnose_plants_temp_glue_pairs.py

Inputs (defaults align with current pipeline outputs):
  - data/processed/temporal/plants_S_temp.npy
  - data/processed/fusion/plants_class_order.json
  - data/processed/groundtruth/plants_ground_truth.json

Outputs:
  - Top glue classes by (weighted/unweighted) degree in S_temp
  - Top strongest inter-service class pairs by S_temp similarity

Notes:
  - S_temp is assumed symmetric, non-negative. Diagonal is ignored.
  - "Heat" is estimated via degree; if you want true "#traces containing class",
    add per-trace counting in the builder and persist that evidence.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

DEFAULT_S_TEMP = os.path.join(REPO_ROOT, "data", "processed", "temporal", "plants_S_temp.npy")
DEFAULT_CLASS_ORDER = os.path.join(
    REPO_ROOT, "data", "processed", "fusion", "plants_class_order.json"
)
DEFAULT_GROUNDTRUTH = os.path.join(
    REPO_ROOT, "data", "processed", "groundtruth", "plants_ground_truth.json"
)


@dataclass(frozen=True)
class Pair:
    i: int
    j: int
    score: float


def _load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_class_order(path: str) -> List[str]:
    data = _load_json(path)
    if isinstance(data, list):
        # expected: ["com.foo.ClassA", ...]
        return [str(x) for x in data]

    # fallback: allow dict-ish structures
    for key in ("classes", "class_order", "classOrder"):
        if key in data and isinstance(data[key], list):
            return [str(x) for x in data[key]]

    raise ValueError(f"Unrecognized class order format in {path}")


def _load_groundtruth_service_by_class(path: str) -> Dict[str, str]:
    gt = _load_json(path)

    # expected in this repo: mapping from class -> service
    if isinstance(gt, dict) and all(isinstance(k, str) for k in gt.keys()):
        # two possible shapes:
        # 1) { "com.foo.Class": "serviceA", ... }
        # 2) { "serviceA": ["Class1", ...], ... }
        # 3) { "com.foo.Class": 2, ... }  (numeric service ids)
        sample_v = next(iter(gt.values())) if gt else None
        if isinstance(sample_v, (str, int, float)):
            return {c: str(svc) for c, svc in gt.items()}
        if isinstance(sample_v, list):
            m: Dict[str, str] = {}
            for svc, classes in gt.items():
                for c in classes:
                    m[str(c)] = str(svc)
            return m

    raise ValueError(f"Unrecognized ground truth format in {path}")


def _top_glue_classes(
    s: np.ndarray,
    classes: List[str],
    topk: int = 5,
    min_edge: float = 0.0,
) -> List[Tuple[str, float, int]]:
    """Return [(class, weighted_degree, unweighted_degree), ...]."""
    if s.ndim != 2 or s.shape[0] != s.shape[1]:
        raise ValueError(f"S_temp must be square; got {s.shape}")

    a = s.copy()
    np.fill_diagonal(a, 0.0)

    if min_edge > 0:
        a = np.where(a >= min_edge, a, 0.0)

    weighted_degree = a.sum(axis=1)
    unweighted_degree = (a > 0).sum(axis=1)

    idx = np.argsort(-weighted_degree)[:topk]
    out: List[Tuple[str, float, int]] = []
    for i in idx:
        out.append((classes[i], float(weighted_degree[i]), int(unweighted_degree[i])))
    return out


def _top_inter_service_pairs(
    s: np.ndarray,
    classes: List[str],
    service_of: Dict[str, str],
    topk: int = 20,
    min_score: float = 0.0,
) -> List[Tuple[str, str, str, str, float]]:
    """Return [(class_i, svc_i, class_j, svc_j, score), ...]."""
    if s.ndim != 2 or s.shape[0] != s.shape[1]:
        raise ValueError(f"S_temp must be square; got {s.shape}")

    n = s.shape[0]
    a = s.copy()
    np.fill_diagonal(a, 0.0)

    pairs: List[Pair] = []

    # Only upper triangle; skip intra-service
    for i in range(n):
        ci = classes[i]
        si = service_of.get(ci, "<unknown>")
        row = a[i, i + 1 :]
        if row.size == 0:
            continue

        # Only consider candidate js where service differs and score >= min_score
        js = np.where(row >= min_score)[0]
        for off in js.tolist():
            j = i + 1 + off
            cj = classes[j]
            sj = service_of.get(cj, "<unknown>")
            if si == sj:
                continue
            score = float(a[i, j])
            if score <= 0:
                continue
            pairs.append(Pair(i=i, j=j, score=score))

    pairs.sort(key=lambda p: p.score, reverse=True)
    pairs = pairs[:topk]

    out: List[Tuple[str, str, str, str, float]] = []
    for p in pairs:
        ci = classes[p.i]
        cj = classes[p.j]
        out.append(
            (
                ci,
                service_of.get(ci, "<unknown>"),
                cj,
                service_of.get(cj, "<unknown>"),
                float(p.score),
            )
        )
    return out


def main(
    s_temp_path: str = DEFAULT_S_TEMP,
    class_order_path: str = DEFAULT_CLASS_ORDER,
    groundtruth_path: str = DEFAULT_GROUNDTRUTH,
    top_glue_k: int = 5,
    top_pair_k: int = 20,
    min_edge: float = 0.0,
    min_pair_score: float = 0.0,
) -> None:
    if not os.path.exists(s_temp_path):
        raise FileNotFoundError(s_temp_path)
    if not os.path.exists(class_order_path):
        raise FileNotFoundError(class_order_path)
    if not os.path.exists(groundtruth_path):
        raise FileNotFoundError(groundtruth_path)

    s = np.load(s_temp_path)
    classes = _load_class_order(class_order_path)

    if s.shape[0] != len(classes):
        raise ValueError(
            "Dimension mismatch: plants_S_temp.npy is "
            f"{s.shape}, but class_order has {len(classes)} entries"
        )

    service_of = _load_groundtruth_service_by_class(groundtruth_path)

    print("=== Plants S_temp diagnostics ===")
    print(f"S_temp: {s_temp_path}")
    print(f"Classes: {class_order_path} (n={len(classes)})")
    print(f"Ground truth: {groundtruth_path} (mapped={len(service_of)})")
    print()

    print(f"--- Top {top_glue_k} glue classes (degree centrality) ---")
    print("(weighted_degree = sum of similarities; degree = count of nonzero edges)")
    for cls, wdeg, deg in _top_glue_classes(s, classes, topk=top_glue_k, min_edge=min_edge):
        svc = service_of.get(cls, "<unknown>")
        print(f"  {wdeg:12.6f}  deg={deg:5d}  svc={svc:20s}  {cls}")
    print()

    print(f"--- Top {top_pair_k} strongest inter-service pairs ---")
    for ci, si, cj, sj, score in _top_inter_service_pairs(
        s,
        classes,
        service_of,
        topk=top_pair_k,
        min_score=min_pair_score,
    ):
        print(f"  {score:12.6f}  {si} :: {ci}    <->    {sj} :: {cj}")


if __name__ == "__main__":
    main()
