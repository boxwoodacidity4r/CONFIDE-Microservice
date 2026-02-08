"""Generate a paper-ready master table for temporal modality quality across systems.

Outputs
-------
- results/temporal_master_table.csv
- results/temporal_master_table.md

The script runs the same computations as `temporal_gate_report.py` (matrix stats +
GT-based intra/inter where available) and keeps output stable for paper artifacts.

Usage (PowerShell)
------------------
  python .\scripts\temporal\generate_temporal_master_table.py

Assumptions
-----------
- S_temp matrices already exist under data/processed/temporal/*_S_temp.npy
- class orders exist under data/processed/fusion/*_class_order.json
- ground-truth files exist under data/processed/groundtruth/*_ground_truth.json
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[2]

SYSTEMS: List[str] = ["plants", "daytrader", "acmeair", "jpetstore"]

# Gate thresholds used for paper readiness (must align with temporal_gate_report.py defaults)
MIN_OFFDIAG_NONZERO_DEFAULT = 200
MIN_DENSITY_DEFAULT = 1e-3
MIN_RATIO_DEFAULT = 1.0

# Systems with size-aware offdiag minimum (copied from gate logic)
def _min_offdiag(system: str, n: int) -> int:
    if system in {"acmeair", "jpetstore"}:
        return max(80, int(0.02 * n * (n - 1)))
    if system in {"plants", "plantsbywebsphere"}:
        return max(50, int(0.08 * n * (n - 1)))
    # DayTrader tends to have concentrated servlet-centered trace evidence; use a mild size-aware floor
    if system in {"daytrader", "daytrader7"}:
        return max(150, int(0.01 * n * (n - 1)))
    return MIN_OFFDIAG_NONZERO_DEFAULT


PATHS: Dict[str, Dict[str, str]] = {
    "plants": {
        "S": "data/processed/temporal/plants_S_temp.npy",
        "order": "data/processed/fusion/plants_class_order.json",
        "gt": "data/processed/groundtruth/plants_ground_truth.json",
    },
    "daytrader": {
        "S": "data/processed/temporal/daytrader_S_temp.npy",
        "order": "data/processed/fusion/daytrader_class_order.json",
        "gt": "data/processed/groundtruth/daytrader_ground_truth.json",
    },
    "acmeair": {
        "S": "data/processed/temporal/acmeair_S_temp.npy",
        "order": "data/processed/fusion/acmeair_class_order.json",
        "gt": "data/processed/groundtruth/acmeair_ground_truth.json",
    },
    "jpetstore": {
        "S": "data/processed/temporal/jpetstore_S_temp.npy",
        "order": "data/processed/fusion/jpetstore_class_order.json",
        "gt": "data/processed/groundtruth/jpetstore_ground_truth.json",
    },
}


@dataclass
class Row:
    system: str
    n: int
    offdiag_nonzero: int
    density: float
    intra_avg: float
    inter_avg: float
    ratio: float


def _load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _label_list(gt: Dict[str, int], order: List[str]) -> List[int]:
    return [int(gt.get(c, -1)) for c in order]


def _offdiag_stats(S: np.ndarray) -> Tuple[int, float]:
    n = S.shape[0]
    off = S.copy()
    np.fill_diagonal(off, 0.0)
    nonzero = int((off > 0).sum())
    density = nonzero / float(n * n - n)
    return nonzero, float(density)


def _intra_inter_stats(S: np.ndarray, labels: List[int]) -> Tuple[float, float, float]:
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
    return intra_avg, inter_avg, float(ratio)


def _fmt(x: float) -> str:
    if x != x:  # NaN
        return "nan"
    if x == float("inf"):
        return "inf"
    return f"{x:.4f}"


def main() -> None:
    rows: List[Row] = []
    failures: List[str] = []

    for system in SYSTEMS:
        p = PATHS[system]
        S_path = ROOT / p["S"]
        order_path = ROOT / p["order"]
        gt_path = ROOT / p["gt"]

        S = np.load(str(S_path))
        order = _load_json(order_path)
        gt = _load_json(gt_path)

        off_nz, density = _offdiag_stats(S)
        labels = _label_list(gt, order)
        intra, inter, ratio = _intra_inter_stats(S, labels)

        # gate checks
        min_off = _min_offdiag(system, int(S.shape[0]))
        ratio_ok = True
        if system not in {"plants", "plantsbywebsphere"}:
            ratio_ok = (ratio == float("inf")) or (ratio >= MIN_RATIO_DEFAULT)

        if off_nz < min_off or density < MIN_DENSITY_DEFAULT or not ratio_ok:
            failures.append(
                f"{system}: offdiag_nonzero={off_nz} (min {min_off}), density={density:.6f} (min {MIN_DENSITY_DEFAULT}), ratio={ratio:.3f} (min {MIN_RATIO_DEFAULT})"
            )

        rows.append(
            Row(
                system=system,
                n=int(S.shape[0]),
                offdiag_nonzero=int(off_nz),
                density=float(density),
                intra_avg=float(intra),
                inter_avg=float(inter),
                ratio=float(ratio),
            )
        )

    if failures:
        raise SystemExit(
            "Temporal gate not satisfied for one or more systems; refusing to write a paper-ready master table.\n"
            + "\n".join("- " + f for f in failures)
        )

    out_csv = ROOT / "results" / "temporal_master_table.csv"
    out_md = ROOT / "results" / "temporal_master_table.md"
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(asdict(rows[0]).keys()))
        w.writeheader()
        for r in rows:
            w.writerow(asdict(r))

    md_lines = []
    md_lines.append("# Temporal Evidence Reliability & Quality Metrics (4 Systems)\n")
    md_lines.append("| system | n | offdiag_nonzero | density | intra_avg | inter_avg | ratio |")
    md_lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        md_lines.append(
            "| "
            + " | ".join(
                [
                    r.system,
                    str(r.n),
                    str(r.offdiag_nonzero),
                    f"{r.density:.6f}",
                    _fmt(r.intra_avg),
                    _fmt(r.inter_avg),
                    _fmt(r.ratio),
                ]
            )
            + " |"
        )

    out_md.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    print(f"Wrote: {out_csv.as_posix()}")
    print(f"Wrote: {out_md.as_posix()}")


if __name__ == "__main__":
    main()
