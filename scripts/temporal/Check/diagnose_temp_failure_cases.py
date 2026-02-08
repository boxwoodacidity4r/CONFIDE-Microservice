"""Diagnose why temporal similarity (S_temp) has poor intra/inter contrast.

This script is designed to answer:
1) Are top-similar pairs mostly cross-domain (indicating co-occurrence bias)?
2) Which class-name keywords dominate top pairs (web/servlet/facade etc)?
3) Does filtering out obvious presentation/infra classes improve contrast?

Usage (PowerShell):
  python scripts/temporal/diagnose_temp_failure_cases.py --system daytrader
  python scripts/temporal/diagnose_temp_failure_cases.py --system jpetstore

Outputs:
  - prints summary to stdout
  - writes a JSON report to data/processed/temporal/<system>_temp_diagnosis.json
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


ROOT = Path(__file__).resolve().parents[2]


DEFAULT_SYSTEM_PATHS = {
    "acmeair": {
        "S": "data/processed/temporal/acmeair_S_temp.npy",
        "order": "data/processed/fusion/acmeair_class_order.json",
        "gt": "data/processed/groundtruth/acmeair_ground_truth.json",
    },
    "daytrader": {
        "S": "data/processed/temporal/daytrader_S_temp.npy",
        "order": "data/processed/fusion/daytrader_class_order.json",
        "gt": "data/processed/groundtruth/daytrader_ground_truth.json",
    },
    "jpetstore": {
        "S": "data/processed/temporal/jpetstore_S_temp.npy",
        "order": "data/processed/fusion/jpetstore_class_order.json",
        "gt": "data/processed/groundtruth/jpetstore_ground_truth.json",
    },
    "plants": {
        "S": "data/processed/temporal/plants_S_temp.npy",
        "order": "data/processed/fusion/plants_class_order.json",
        "gt": "data/processed/groundtruth/plants_ground_truth.json",
    },
}


# common noisy/presentation keywords for filtering
DEFAULT_FILTER_KEYWORDS = [
    "web",
    "servlet",
    "jsf",
    "jsp",
    "controller",
    "action",
    "view",
    "filter",
    "listener",
    "config",
    "dto",
    "util",
    "utils",
    "helper",
    "common",
    "facade",
    "gateway",
    "interceptor",
    "producer",
]


def _load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _label_list(gt: Dict[str, int], order: List[str]) -> List[int]:
    return [int(gt.get(c, -1)) for c in order]


def _intra_inter_stats(S: np.ndarray, labels: List[int]) -> Tuple[float, float, float, int, int]:
    n = len(labels)
    intra: List[float] = []
    inter: List[float] = []
    for i in range(n):
        for j in range(i + 1, n):
            if labels[i] == -1 or labels[j] == -1:
                continue
            if labels[i] == labels[j]:
                intra.append(float(S[i, j]))
            else:
                inter.append(float(S[i, j]))
    intra_avg = float(np.mean(intra)) if intra else float("nan")
    inter_avg = float(np.mean(inter)) if inter else float("nan")
    ratio = (intra_avg / inter_avg) if (inter_avg and inter_avg > 0) else float("inf")
    return intra_avg, inter_avg, ratio, len(intra), len(inter)


def _top_pairs(S: np.ndarray, top_k: int) -> List[Tuple[int, int, float]]:
    n = S.shape[0]
    pairs: List[Tuple[int, int, float]] = []
    # simple O(n^2) scan is fine for our dataset sizes
    for i in range(n):
        row = S[i]
        for j in range(i + 1, n):
            pairs.append((i, j, float(row[j])))
    pairs.sort(key=lambda x: x[2], reverse=True)
    return pairs[:top_k]


def _keyword_hits(cls: str, keywords: List[str]) -> List[str]:
    s = cls.lower()
    return [k for k in keywords if k in s]


def _filtered_indices(order: List[str], keywords: List[str], *, mode: str = "substring") -> List[int]:
    """Return indices to keep (i.e., NOT dropped).

    mode:
      - 'raw'      : do not filter; keep all (diagnose the saved matrix as-is)
      - 'substring': drop if any keyword is a substring of FQCN
      - 'temporal' : use the same token-based predicate as temporal builder
    """
    if mode == "raw":
        return list(range(len(order)))

    if mode == "temporal":
        # keep in sync with scripts/temporal/build_S_temp.py
        try:
            from ..build_S_temp import _should_drop_temporal_class  # type: ignore
        except Exception:  # pragma: no cover
            from build_S_temp import _should_drop_temporal_class  # type: ignore

        keep_idx = []
        for i, c in enumerate(order):
            if not _should_drop_temporal_class(c, keywords):
                keep_idx.append(i)

        # Avoid the misleading degenerate case (kept=0) which makes stats NaN.
        # If the predicate becomes too strict for a dataset, keep all and warn.
        if not keep_idx:
            print("[WARN] filter_mode=temporal dropped all classes; falling back to keep-all for diagnosis")
            return list(range(len(order)))
        return keep_idx

    # fallback to legacy substring behavior
    idx: List[int] = []
    for i, c in enumerate(order):
        hits = _keyword_hits(c, keywords)
        if hits:
            continue
        idx.append(i)
    return idx


def _submatrix(S: np.ndarray, idx: List[int]) -> np.ndarray:
    return S[np.ix_(idx, idx)]


def diagnose(system: str, top_k: int, filter_keywords: List[str], filter_mode: str = "substring") -> Dict[str, object]:
    p = DEFAULT_SYSTEM_PATHS.get(system)
    if not p:
        raise ValueError(f"Unknown system: {system}")

    S_path = ROOT / p["S"]
    order_path = ROOT / p["order"]
    gt_path = ROOT / p["gt"]

    S = np.load(str(S_path))
    order: List[str] = _load_json(order_path)
    gt: Dict[str, int] = _load_json(gt_path)

    labels = _label_list(gt, order)
    assert S.shape[0] == S.shape[1] == len(order), f"shape mismatch: S={S.shape} order={len(order)}"

    intra, inter, ratio, n_intra, n_inter = _intra_inter_stats(S, labels)

    # top pairs analysis
    pairs = _top_pairs(S, top_k=top_k)

    cross = 0
    labeled = 0
    kw_counter: Counter[str] = Counter()
    top_rows: List[Dict[str, object]] = []

    for i, j, sim in pairs:
        ci, cj = order[i], order[j]
        li, lj = labels[i], labels[j]
        if li != -1 and lj != -1:
            labeled += 1
            if li != lj:
                cross += 1

        hits_i = _keyword_hits(ci, filter_keywords)
        hits_j = _keyword_hits(cj, filter_keywords)
        kw_counter.update(hits_i)
        kw_counter.update(hits_j)

        top_rows.append(
            {
                "i": i,
                "j": j,
                "sim": sim,
                "class_i": ci,
                "class_j": cj,
                "label_i": li,
                "label_j": lj,
                "is_cross_domain": (li != -1 and lj != -1 and li != lj),
                "hits_i": hits_i,
                "hits_j": hits_j,
            }
        )

    cross_ratio = (cross / labeled) if labeled else None

    # filtered stats
    keep_idx = _filtered_indices(order, filter_keywords, mode=filter_mode)
    S2 = _submatrix(S, keep_idx)
    order2 = [order[i] for i in keep_idx]
    labels2 = [labels[i] for i in keep_idx]
    intra2, inter2, ratio2, n_intra2, n_inter2 = _intra_inter_stats(S2, labels2)

    report: Dict[str, object] = {
        "system": system,
        "filter_mode": filter_mode,
        "paths": {"S": str(S_path), "order": str(order_path), "gt": str(gt_path)},
        "overall": {
            "intra": intra,
            "inter": inter,
            "ratio": ratio,
            "pairs_intra": n_intra,
            "pairs_inter": n_inter,
        },
        "top_pairs": {
            "top_k": top_k,
            "labeled_pairs_in_top_k": labeled,
            "cross_domain_pairs_in_top_k": cross,
            "cross_domain_ratio_in_top_k": cross_ratio,
            "dominant_keywords_in_top_k": kw_counter.most_common(20),
            "rows": top_rows,
        },
        "filtered": {
            "filter_keywords": filter_keywords,
            "kept_classes": len(order2),
            "dropped_classes": len(order) - len(order2),
            "intra": intra2,
            "inter": inter2,
            "ratio": ratio2,
            "pairs_intra": n_intra2,
            "pairs_inter": n_inter2,
        },
    }

    return report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", required=True, choices=list(DEFAULT_SYSTEM_PATHS.keys()))
    ap.add_argument("--top-k", type=int, default=200)
    ap.add_argument(
        "--filter-mode",
        choices=["raw", "substring", "temporal"],
        default="temporal",
        help="How to apply keyword filtering for the 'filtered' stats section. Use 'raw' to skip filtering.",
    )
    ap.add_argument(
        "--filter-keywords",
        nargs="*",
        default=DEFAULT_FILTER_KEYWORDS,
        help="Keywords to drop classes containing them when computing filtered stats",
    )
    args = ap.parse_args()

    report = diagnose(args.system, top_k=args.top_k, filter_keywords=args.filter_keywords, filter_mode=args.filter_mode)

    out_path = ROOT / "data" / "processed" / "temporal" / f"{args.system}_temp_diagnosis.json"
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    ov = report["overall"]
    fv = report["filtered"]
    tp = report["top_pairs"]

    print(f"[{args.system}] overall ratio={ov['ratio']:.3f} intra={ov['intra']:.4f} inter={ov['inter']:.4f} pairs={ov['pairs_intra']}/{ov['pairs_inter']}")
    print(
        f"[{args.system}] top{tp['top_k']} labeled_pairs={tp['labeled_pairs_in_top_k']} cross_domain={tp['cross_domain_pairs_in_top_k']} cross_ratio={tp['cross_domain_ratio_in_top_k']}"
    )
    print(f"[{args.system}] dominant keywords in top{tp['top_k']}: {tp['dominant_keywords_in_top_k']}")
    print(f"[{args.system}] filter_mode={report.get('filter_mode')}")
    print(
        f"[{args.system}] filtered(drop_keywords) ratio={fv['ratio']:.3f} intra={fv['intra']:.4f} inter={fv['inter']:.4f} pairs={fv['pairs_intra']}/{fv['pairs_inter']} kept={fv['kept_classes']} dropped={fv['dropped_classes']}"
    )
    print(f"Wrote report: {out_path}")


if __name__ == "__main__":
    main()
