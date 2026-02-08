"""Aggregate per-system mono baselines vs ours tables into one all-system table.

Input:
- results/ablation/mono_baselines_vs_ours_<system>_<ts>.csv

Output:
- results/ablation/mono_baselines_vs_ours_ALL_<ts>.csv/.md

This script is intentionally simple and deterministic:
- For each system, pick the latest CSV by timestamp in filename.
- Concatenate rows (16 rows: 4 systems × 4 methods).

Usage:
  python scripts/multimodal/phase4/make_all_systems_mono_table.py

Optional:
  --systems acmeair,daytrader,plants,jpetstore
  --use_paths path1.csv,path2.csv,...  (manual override)
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
from pathlib import Path
from typing import Dict, List


ROOT = Path(__file__).resolve().parents[3]
ABL = ROOT / "results" / "ablation"


def _pick_latest_csv(system: str) -> Path:
    pats = sorted(ABL.glob(f"mono_baselines_vs_ours_{system}_*.csv"))
    if not pats:
        raise FileNotFoundError(f"No CSV found for system={system}. Run run_mono_baselines_and_ours_table.py first.")
    return pats[-1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--systems", type=str, default="acmeair,daytrader,plants,jpetstore")
    ap.add_argument(
        "--use_paths",
        type=str,
        default=None,
        help="Comma-separated explicit CSV paths to use (manual override).",
    )
    args = ap.parse_args()

    systems = [s.strip() for s in args.systems.split(",") if s.strip()]

    csv_paths: List[Path] = []
    if args.use_paths:
        csv_paths = [Path(p.strip()) for p in args.use_paths.split(",") if p.strip()]
    else:
        for s in systems:
            csv_paths.append(_pick_latest_csv(s))

    rows: List[Dict[str, str]] = []
    for p in csv_paths:
        with p.open("r", encoding="utf-8") as f:
            r = csv.DictReader(f)
            for row in r:
                rows.append(dict(row))

    if not rows:
        raise RuntimeError("No rows loaded.")

    # stable sort: by system then method
    rows.sort(key=lambda d: (d.get("system", ""), d.get("method", "")))

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_csv = ABL / f"mono_baselines_vs_ours_ALL_{ts}.csv"
    out_md = ABL / f"mono_baselines_vs_ours_ALL_{ts}.md"

    fieldnames = list(rows[0].keys())
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    # markdown
    with out_md.open("w", encoding="utf-8") as f:
        f.write("# Mono baselines vs ours (ALL systems)\n\n")
        f.write(f"Generated: {ts}\n\n")
        f.write("| System | Method | BCubedF1 | MoJoSim | K | GT_K | K-Diff | mu_override | U | cap |\n")
        f.write("|---|---|---:|---:|---:|---:|---:|---:|---|---:|\n")
        for r in rows:
            f.write(
                "| {sys} | {m} | {f1:.4f} | {mj:.2f} | {k} | {gtk} | {kd:+d} | {mu} | {u} | {cap} |\n".format(
                    sys=r.get("system"),
                    m=r.get("method"),
                    f1=float(r.get("bcubed_f1", 0.0) or 0.0),
                    mj=float(r.get("mojosim", 0.0) or 0.0),
                    k=int(float(r.get("pred_k", 0) or 0)),
                    gtk=int(float(r.get("gt_k", 0) or 0)),
                    kd=int(float(r.get("k_diff", 0) or 0)),
                    mu=r.get("mu_override", "-"),
                    u=r.get("u_ablation", ""),
                    cap=float(r.get("cap", 0.0) or 0.0),
                )
            )

    print(f"[OK] saved: {out_csv}")
    print(f"[OK] saved: {out_md}")


if __name__ == "__main__":
    main()
