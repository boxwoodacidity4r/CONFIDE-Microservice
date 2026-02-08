"""Generic S_temp diagnostics: glue classes + strongest inter-service pairs.

Usage:
  python scripts/temporal/diagnose_s_temp_glue_pairs.py --system plants
  python scripts/temporal/diagnose_s_temp_glue_pairs.py --system daytrader

Inputs (by --system canonical key):
  - data/processed/temporal/{physical}_S_temp.npy
  - data/processed/fusion/{system}_class_order.json
  - data/processed/groundtruth/{system}_ground_truth.json

Notes:
  - Ground truth formats supported:
      * { "ClassFQCN": "serviceName" }
      * { "ClassFQCN": 2 }  (numeric service ids)
      * { "serviceName": ["ClassFQCN", ...] }
  - "Glue" is approximated via weighted degree from S_temp.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

SYSTEM_NAME_MAP = {
    "daytrader7": "daytrader",
    "plantsbywebsphere": "plants",
}


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
        return [str(x) for x in data]
    for key in ("classes", "class_order", "classOrder"):
        if key in data and isinstance(data[key], list):
            return [str(x) for x in data[key]]
    raise ValueError(f"Unrecognized class order format in {path}")


def _load_groundtruth_service_by_class(path: str) -> Dict[str, str]:
    gt = _load_json(path)
    if isinstance(gt, dict) and all(isinstance(k, str) for k in gt.keys()):
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
    topk: int,
    min_edge: float,
) -> List[Tuple[str, float, int]]:
    a = np.array(s, dtype=float, copy=True)
    np.fill_diagonal(a, 0.0)
    if min_edge > 0:
        a = np.where(a >= min_edge, a, 0.0)
    wdeg = a.sum(axis=1)
    deg = (a > 0).sum(axis=1)
    idx = np.argsort(-wdeg)[: max(0, min(topk, len(classes)))]
    return [(classes[i], float(wdeg[i]), int(deg[i])) for i in idx]


def _top_inter_pairs(
    s: np.ndarray,
    classes: List[str],
    service_of: Dict[str, str],
    topk: int,
    min_score: float,
) -> List[Tuple[str, str, str, str, float]]:
    a = np.array(s, dtype=float, copy=True)
    np.fill_diagonal(a, 0.0)

    n = a.shape[0]
    out: List[Pair] = []

    for i in range(n):
        ci = classes[i]
        si = service_of.get(ci, "<unknown>")
        row = a[i, i + 1 :]
        if row.size == 0:
            continue
        js = np.where(row >= min_score)[0]
        for off in js.tolist():
            j = i + 1 + off
            cj = classes[j]
            sj = service_of.get(cj, "<unknown>")
            if si == sj:
                continue
            score = float(a[i, j])
            if score > 0:
                out.append(Pair(i=i, j=j, score=score))

    out.sort(key=lambda p: p.score, reverse=True)
    out = out[: max(0, topk)]

    return [
        (
            classes[p.i],
            service_of.get(classes[p.i], "<unknown>"),
            classes[p.j],
            service_of.get(classes[p.j], "<unknown>"),
            float(p.score),
        )
        for p in out
    ]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", required=True, help="Canonical system key: acmeair/daytrader/jpetstore/plants")
    ap.add_argument("--top-glue", type=int, default=5)
    ap.add_argument("--top-pairs", type=int, default=20)
    ap.add_argument("--min-edge", type=float, default=0.0)
    ap.add_argument("--min-pair-score", type=float, default=0.0)
    return ap.parse_args()


def main() -> None:
    args = parse_args()

    system = str(args.system)
    physical = SYSTEM_NAME_MAP.get(system, system)

    s_temp_path = os.path.join(REPO_ROOT, "data", "processed", "temporal", f"{physical}_S_temp.npy")
    class_order_path = os.path.join(REPO_ROOT, "data", "processed", "fusion", f"{system}_class_order.json")
    groundtruth_path = os.path.join(REPO_ROOT, "data", "processed", "groundtruth", f"{system}_ground_truth.json")

    if not os.path.exists(s_temp_path):
        raise FileNotFoundError(s_temp_path)
    if not os.path.exists(class_order_path):
        raise FileNotFoundError(class_order_path)
    if not os.path.exists(groundtruth_path):
        raise FileNotFoundError(groundtruth_path)

    s = np.load(s_temp_path)
    classes = _load_class_order(class_order_path)
    if s.shape[0] != len(classes):
        raise ValueError(f"Dimension mismatch: S_temp is {s.shape}, class_order has {len(classes)}")

    service_of = _load_groundtruth_service_by_class(groundtruth_path)

    print(f"=== {system} S_temp diagnostics ===")
    print(f"S_temp: {s_temp_path}")
    print(f"Classes: {class_order_path} (n={len(classes)})")
    print(f"Ground truth: {groundtruth_path} (mapped={len(service_of)})")
    print()

    print(f"--- Top {args.top_glue} glue classes (weighted degree) ---")
    for cls, wdeg, deg in _top_glue_classes(s, classes, args.top_glue, args.min_edge):
        print(f"  {wdeg:12.6f}  deg={deg:5d}  svc={service_of.get(cls, '<unknown>'):20s}  {cls}")

    print()
    print(f"--- Top {args.top_pairs} inter-service pairs ---")
    for ci, si, cj, sj, score in _top_inter_pairs(s, classes, service_of, args.top_pairs, args.min_pair_score):
        print(f"  {score:12.6f}  {si} :: {ci}    <->    {sj} :: {cj}")


if __name__ == "__main__":
    main()
