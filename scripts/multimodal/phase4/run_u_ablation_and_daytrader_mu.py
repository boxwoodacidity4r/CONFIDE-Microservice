"""Phase4 experiment runner: (1) U ablation (B) + auto case study, (2) DayTrader mu tuning.

Focus:
- Ablation B: CAC(with U) vs CAC(U≡0) under identical Phase3 hyperparams.
- Case study: automatically select "most uncertain edges" and show how U changes weights
  and cluster assignments.
- DayTrader: sweep mu_override to increase topological weight, evaluate against GT,
  and diagnose K-collapse.

Outputs (timestamped):
- results/ablation/u_ablation_B_<ts>.csv / .md
- results/ablation/daytrader_mu_sweep_<ts>.csv / .md
- results/ablation/case_study_most_uncertain_<system>_<ts>.json

This script does NOT rerun Phase1/2.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
PHASE3 = ROOT / "scripts" / "multimodal" / "phase3" / "phase3_cac_evaluation.py"
PHASE4 = ROOT / "scripts" / "multimodal" / "phase4" / "evaluate_partition_f1.py"

SYSTEMS_DEFAULT = ["acmeair", "daytrader", "plants", "jpetstore"]


def _run(cmd: List[str], cwd: Path) -> str:
    p = subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True, encoding="utf-8", errors="replace")
    out = (p.stdout or "") + ("\n" + p.stderr if p.stderr else "")
    if p.returncode != 0:
        raise RuntimeError(f"Command failed ({p.returncode}): {' '.join(cmd)}\n{out}")
    return out


def _python() -> str:
    return sys.executable


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


def _load_partition(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _top_uncertain_edges(system: str, topk: int) -> List[Tuple[int, int, float]]:
    U = np.load(str(ROOT / "data" / "processed" / "edl" / f"{system}_edl_uncertainty.npy"))
    # normalize like Phase3
    umin, umax = float(U.min()), float(U.max())
    if umax > umin:
        U = (U - umin) / (umax - umin)

    n = U.shape[0]
    iu = np.triu_indices(n, k=1)
    vals = U[iu]
    idx = np.argsort(vals)[::-1][: int(topk)]
    edges = [(int(iu[0][k]), int(iu[1][k]), float(vals[k])) for k in idx]
    return edges


def _phase3_run(system: str, *, cap: float, u_ablation: str, mode: str, alpha: float, merge: bool, min_cluster: int, mu_override: float | None) -> None:
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
        str(u_ablation),
        "--target_from_gt",
    ]
    if merge:
        cmd += ["--merge_small_clusters", "--min_cluster_size", str(min_cluster)]
    if mu_override is not None:
        cmd += ["--mu_override", str(mu_override)]
    _run(cmd, ROOT)


def run_u_ablation_B(args) -> Tuple[Path, Path, Path]:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = ROOT / "results" / "ablation"
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / f"u_ablation_B_{ts}.csv"
    md_path = out_dir / f"u_ablation_B_{ts}.md"

    rows: List[Dict[str, Any]] = []

    for system in args.systems:
        # pick cap from existing best-by-bcubed meta if possible
        best_meta = ROOT / "data" / "processed" / "fusion" / f"{system}_cac_best_by_bcubed.json"
        cap = float(args.dpep_cap)
        if best_meta.exists():
            try:
                cap = float(json.loads(best_meta.read_text(encoding="utf-8")).get("best_cap", cap))
            except Exception:
                pass

        fusion_dir = ROOT / "data" / "processed" / "fusion"
        with_u_part = fusion_dir / f"{system}_cac-final_partition.json"
        no_u_part = fusion_dir / f"{system}_cac-final_no_u_partition.json"

        # Run with U
        _phase3_run(
            system,
            cap=cap,
            u_ablation="with_u",
            mode=args.mode,
            alpha=args.alpha,
            merge=args.merge_small_clusters,
            min_cluster=args.min_cluster_size,
            mu_override=None,
        )
        if not with_u_part.exists():
            raise FileNotFoundError(f"Expected partition not found after with-U run: {with_u_part}")
        m_with = _evaluate(system, with_u_part)

        # Run without U (U≡0)
        _phase3_run(
            system,
            cap=cap,
            u_ablation="no_u",
            mode=args.mode,
            alpha=args.alpha,
            merge=args.merge_small_clusters,
            min_cluster=args.min_cluster_size,
            mu_override=None,
        )
        # Phase3 always writes *_cac-final_partition.json; snapshot it to the no-U filename
        if not with_u_part.exists():
            raise FileNotFoundError(f"Expected partition not found after no-U run: {with_u_part}")
        no_u_part.write_text(with_u_part.read_text(encoding="utf-8"), encoding="utf-8")
        m_no = _evaluate(system, no_u_part)

        rows.append(
            {
                "system": system,
                "cap": cap,
                "with_u_bcubed_f1": float(m_with.get("bcubed_f1", 0.0)),
                "with_u_mojosim": float(m_with.get("mojosim", 0.0)),
                "with_u_k": int(m_with.get("pred_k", 0) or 0),
                "no_u_bcubed_f1": float(m_no.get("bcubed_f1", 0.0)),
                "no_u_mojosim": float(m_no.get("mojosim", 0.0)),
                "no_u_k": int(m_no.get("pred_k", 0) or 0),
                "delta_bcubed_f1": float(m_with.get("bcubed_f1", 0.0)) - float(m_no.get("bcubed_f1", 0.0)),
                "delta_mojosim": float(m_with.get("mojosim", 0.0)) - float(m_no.get("mojosim", 0.0)),
            }
        )

        # Case study JSON per system: most uncertain edges + class names
        try:
            class_order = json.loads((ROOT / "data" / "processed" / "fusion" / f"{system}_class_order.json").read_text(encoding="utf-8"))
            edges = _top_uncertain_edges(system, int(args.case_topk))
            case = {
                "system": system,
                "topk": int(args.case_topk),
                "edges": [
                    {
                        "i": i,
                        "j": j,
                        "u": u,
                        "class_i": str(class_order[i]),
                        "class_j": str(class_order[j]),
                    }
                    for i, j, u in edges
                ],
                "note": "Edges selected purely by highest normalized uncertainty U(i,j). Use as paper case-study candidates.",
            }
            case_path = out_dir / f"case_study_most_uncertain_{system}_{ts}.json"
            case_path.write_text(json.dumps(case, indent=2, ensure_ascii=False), encoding="utf-8")
        except Exception:
            pass

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    with md_path.open("w", encoding="utf-8") as f:
        f.write("# U Ablation (B): CAC with U vs CAC with U≡0\n\n")
        f.write(f"Generated: {ts}\n\n")
        f.write("| System | cap | BCubedF1 (with U) | BCubedF1 (U≡0) | ΔF1 | MoJoSim (with U) | MoJoSim (U≡0) | ΔMoJo | K(with U) | K(U≡0) |\n")
        f.write("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for r in rows:
            f.write(
                "| {s} | {cap:.2f} | {f1u:.4f} | {f10:.4f} | {df1:+.4f} | {mju:.2f} | {mj0:.2f} | {dmj:+.2f} | {ku} | {k0} |\n".format(
                    s=r["system"],
                    cap=float(r["cap"]),
                    f1u=float(r["with_u_bcubed_f1"]),
                    f10=float(r["no_u_bcubed_f1"]),
                    df1=float(r["delta_bcubed_f1"]),
                    mju=float(r["with_u_mojosim"]),
                    mj0=float(r["no_u_mojosim"]),
                    dmj=float(r["delta_mojosim"]),
                    ku=int(r["with_u_k"]),
                    k0=int(r["no_u_k"]),
                )
            )

        f.write("\n## Notes\n")
        f.write("- This is the hardest ablation: same CAC pipeline, only uncertainty awareness toggled.\n")
        f.write("- Case-study candidates are saved as JSON files: `case_study_most_uncertain_<system>_<ts>.json`.\n")

    return csv_path, md_path, out_dir


def run_daytrader_mu_sweep(args) -> Tuple[Path, Path]:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = ROOT / "results" / "ablation"
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / f"daytrader_mu_sweep_{ts}.csv"
    md_path = out_dir / f"daytrader_mu_sweep_{ts}.md"

    system = "daytrader"

    # cap from best meta
    best_meta = ROOT / "data" / "processed" / "fusion" / f"{system}_cac_best_by_bcubed.json"
    cap = float(args.dpep_cap)
    if best_meta.exists():
        try:
            cap = float(json.loads(best_meta.read_text(encoding="utf-8")).get("best_cap", cap))
        except Exception:
            pass

    mus = [float(x) for x in args.mu_values.split(",") if x.strip()]

    rows: List[Dict[str, Any]] = []
    for mu in mus:
        _phase3_run(
            system,
            cap=cap,
            u_ablation="with_u",
            mode=args.mode,
            alpha=args.alpha,
            merge=args.merge_small_clusters,
            min_cluster=args.min_cluster_size,
            mu_override=float(mu),
        )
        cac_part = ROOT / "data" / "processed" / "fusion" / f"{system}_cac-final_partition.json"
        m = _evaluate(system, cac_part)
        rows.append(
            {
                "mu_override": float(mu),
                "cap": float(cap),
                "bcubed_f1": float(m.get("bcubed_f1", 0.0)),
                "mojosim": float(m.get("mojosim", 0.0)),
                "pred_k": int(m.get("pred_k", 0) or 0),
                "gt_k": int(m.get("gt_k", 0) or 0),
                "k_diff": int(m.get("pred_k", 0) or 0) - int(m.get("gt_k", 0) or 0),
            }
        )

    # sort by bcubed
    rows.sort(key=lambda r: (r["bcubed_f1"], r["mojosim"]), reverse=True)

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    with md_path.open("w", encoding="utf-8") as f:
        f.write("# DayTrader mu_override sweep (surgical weight tuning)\n\n")
        f.write(f"Generated: {ts}\n\n")
        f.write(f"Fixed cap={cap:.2f}, mode={args.mode}, alpha={args.alpha}, merge_small_clusters={args.merge_small_clusters}, min_cluster_size={args.min_cluster_size}\n\n")
        f.write("| mu_override | BCubedF1 | MoJoSim | K | GT_K | K-Diff |\n")
        f.write("|---:|---:|---:|---:|---:|---:|\n")
        for r in rows:
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

        best = rows[0]
        f.write("\n## K-collapse diagnostic\n")
        f.write(
            "If Baseline collapses to very small K (e.g., 2) while GT_K is larger, its score can be misleading. "
            "This sweep reports K for each mu override so CAC can be compared under similar granularity.\n"
        )
        f.write("\n**Best by BCubedF1:**\n\n")
        f.write(json.dumps(best, indent=2, ensure_ascii=False))
        f.write("\n")

    return csv_path, md_path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--systems", type=str, default=",".join(SYSTEMS_DEFAULT))
    ap.add_argument("--mode", type=str, default="sigmoid")
    ap.add_argument("--alpha", type=float, default=15.0)
    ap.add_argument("--dpep_cap", type=float, default=0.18, help="Fallback cap if best-by-bcubed meta is absent.")
    ap.add_argument("--merge_small_clusters", action="store_true", default=True)
    ap.add_argument("--min_cluster_size", type=int, default=3)
    ap.add_argument("--case_topk", type=int, default=20)
    ap.add_argument(
        "--mu_values",
        type=str,
        default="0.10,0.20,0.30,0.40,0.50,0.60,0.70,0.80,0.90",
        help="DayTrader mu sweep values. Here mu_override is structural weight in S=mu*S_struct+(1-mu)*S_sem.",
    )
    args = ap.parse_args()
    args.systems = [s.strip() for s in args.systems.split(",") if s.strip()]

    run_u_ablation_B(args)
    run_daytrader_mu_sweep(args)


if __name__ == "__main__":
    main()
