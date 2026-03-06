"""Phase4: sweep `mu_override` (structural weight) for all systems.

This generalizes the existing DayTrader-only sweep so reviewers can see the
structural-weight sensitivity across all four case-study systems.

What this does:
- For each system in `--systems`, run Phase3 CAC evaluation multiple times with
  fixed Phase3 hyperparams (cap/mode/alpha/merge_small_clusters), overriding
  `mu_override`.
- After each run, evaluate against ground truth using Phase4 evaluator.
- Write one CSV and one Markdown table per system.

Outputs (timestamped):
- results/ablation/<system>_mu_sweep_<ts>.csv
- results/ablation/<system>_mu_sweep_<ts>.md

Usage:
  python scripts/multimodal/phase4/run_mu_sweep_all_systems.py
  python scripts/multimodal/phase4/run_mu_sweep_all_systems.py --systems acmeair,plants --mu_values 0.1,0.2,0.3

Notes:
- This script does NOT rerun Phase1/2.
- It relies on `phase3_cac_evaluation.py` producing
  data/processed/fusion/<system>_cac-final_partition.json.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List


ROOT = Path(__file__).resolve().parents[3]
PHASE3 = ROOT / "scripts" / "multimodal" / "phase3" / "phase3_cac_evaluation.py"
PHASE4 = ROOT / "scripts" / "multimodal" / "phase4" / "evaluate_partition_f1.py"

SYSTEMS_DEFAULT = ["acmeair", "daytrader", "plants", "jpetstore"]


def _python() -> str:
    return sys.executable


def _run(cmd: List[str], cwd: Path) -> str:
    p = subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True, encoding="utf-8", errors="replace")
    out = (p.stdout or "") + ("\n" + p.stderr if p.stderr else "")
    if p.returncode != 0:
        raise RuntimeError(f"Command failed ({p.returncode}): {' '.join(cmd)}\n{out}")
    return out


def _phase3_run(
    system: str,
    *,
    cap: float,
    mode: str,
    alpha: float,
    merge: bool,
    min_cluster: int,
    mu_override: float,
) -> None:
    cmd = [
        _python(),
        str(PHASE3),
        system,
        "--dpep_cap",
        str(cap),
        "--mode",
        str(mode),
        "--alpha",
        str(alpha),
        "--u_ablation",
        "with_u",
        "--target_from_gt",
        "--mu_override",
        str(mu_override),
    ]
    if merge:
        cmd += ["--merge_small_clusters", "--min_cluster_size", str(min_cluster)]
    _run(cmd, ROOT)


def _evaluate(system: str, pred_path: Path) -> Dict[str, Any]:
    gt = ROOT / "data" / "processed" / "groundtruth" / f"{system}_ground_truth.json"
    class_order = ROOT / "data" / "processed" / "fusion" / f"{system}_class_order.json"
    out_json = ROOT / "results" / "ablation" / "_tmp" / f"metrics_{system}.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        _python(),
        str(PHASE4),
        "--gt",
        str(gt),
        "--pred",
        str(pred_path),
        "--class_order",
        str(class_order),
        "--out_json",
        str(out_json),
    ]
    _run(cmd, ROOT)
    return json.loads(out_json.read_text(encoding="utf-8"))


def _cap_for_system(system: str, fallback: float) -> float:
    best_meta = ROOT / "data" / "processed" / "fusion" / f"{system}_cac_best_by_bcubed.json"
    if best_meta.exists():
        try:
            return float(json.loads(best_meta.read_text(encoding="utf-8")).get("best_cap", fallback))
        except Exception:
            return fallback
    return fallback


def run_system(system: str, args, ts: str) -> tuple[Path, Path]:
    out_dir = ROOT / "results" / "ablation"
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / f"{system}_mu_sweep_{ts}.csv"
    md_path = out_dir / f"{system}_mu_sweep_{ts}.md"

    cap = _cap_for_system(system, float(args.dpep_cap))
    mus = [float(x) for x in args.mu_values.split(",") if x.strip()]

    rows: List[Dict[str, Any]] = []
    for mu in mus:
        _phase3_run(
            system,
            cap=cap,
            mode=args.mode,
            alpha=args.alpha,
            merge=args.merge_small_clusters,
            min_cluster=args.min_cluster_size,
            mu_override=float(mu),
        )
        part = ROOT / "data" / "processed" / "fusion" / f"{system}_cac-final_partition.json"
        if not part.exists():
            raise FileNotFoundError(f"Expected partition not found: {part}")
        m = _evaluate(system, part)
        rows.append(
            {
                "system": system,
                "mu_override": float(mu),
                "cap": float(cap),
                "bcubed_f1": float(m.get("bcubed_f1", 0.0)),
                "mojosim": float(m.get("mojosim", 0.0)),
                "pred_k": int(m.get("pred_k", 0) or 0),
                "gt_k": int(m.get("gt_k", 0) or 0),
                "k_diff": int(m.get("pred_k", 0) or 0) - int(m.get("gt_k", 0) or 0),
            }
        )

    # keep a copy sorted by mu for plots; but in table, show best first
    rows_by_mu = sorted(rows, key=lambda r: r["mu_override"])
    rows_best = sorted(rows, key=lambda r: (r["bcubed_f1"], r["mojosim"]), reverse=True)

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_by_mu[0].keys()))
        w.writeheader()
        w.writerows(rows_by_mu)

    with md_path.open("w", encoding="utf-8") as f:
        f.write(f"# {system}: mu_override sweep (structural weight)\n\n")
        f.write(f"Generated: {ts}\n\n")
        f.write(
            f"Fixed cap={cap:.3f}, mode={args.mode}, alpha={args.alpha}, merge_small_clusters={args.merge_small_clusters}, min_cluster_size={args.min_cluster_size}\n\n"
        )
        f.write("| mu_override | BCubedF1 | MoJoSim | K | GT_K | K-Diff |\n")
        f.write("|---:|---:|---:|---:|---:|---:|\n")
        for r in rows_best:
            f.write(
                "| {mu:.2f} | {f1:.4f} | {mj:.2f} | {k} | {gtk} | {kd:+d} |\n".format(
                    mu=float(r["mu_override"]),
                    f1=float(r["bcubed_f1"]),
                    mj=float(r["mojosim"]),
                    k=int(r["pred_k"]),
                    gtk=int(r["gt_k"]),
                    kd=int(r["k_diff"]),
                )
            )

        f.write("\n**Best by BCubedF1:**\n\n")
        f.write(json.dumps(rows_best[0], indent=2, ensure_ascii=False))
        f.write("\n")

    return csv_path, md_path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--systems", type=str, default=",".join(SYSTEMS_DEFAULT))
    ap.add_argument("--mode", type=str, default="sigmoid")
    ap.add_argument("--alpha", type=float, default=15.0)
    ap.add_argument("--dpep_cap", type=float, default=0.18)
    ap.add_argument("--merge_small_clusters", action="store_true", default=True)
    ap.add_argument("--min_cluster_size", type=int, default=3)
    ap.add_argument(
        "--mu_values",
        type=str,
        default="0.10,0.20,0.30,0.40,0.50,0.60,0.70,0.80,0.90",
    )

    args = ap.parse_args()
    systems = [s.strip() for s in args.systems.split(",") if s.strip()]

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    for system in systems:
        csv_path, md_path = run_system(system, args, ts)
        print(f"[OK] {system}: {csv_path}")
        print(f"[OK] {system}: {md_path}")


if __name__ == "__main__":
    main()
