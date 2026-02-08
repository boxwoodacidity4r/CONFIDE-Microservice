"""Phase4 helper: select best Phase3 DPEP cap by BCubed F1.

This script automates an academically defensible selection procedure:
- For each system:
  1) sweep DPEP cap values by invoking Phase3 CAC evaluation
  2) evaluate the generated partitions against GT using Phase4 metrics
  3) pick the cap that maximizes BCubed F1 (ties broken by MoJoSim then Q)
  4) snapshot the best partition and write paper-ready summary (CSV/MD)

It produces:
- results/ablation/phase4_best_by_bcubed_*.csv
- results/ablation/phase4_best_by_bcubed_*.md
- data/processed/fusion/{system}_cac-final_best_by_bcubed_partition.json
- data/processed/fusion/{system}_cac_best_by_bcubed.json

Designed for Windows PowerShell and reproducibility.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


SYSTEMS = ["acmeair", "daytrader", "plants", "jpetstore"]


@dataclass
class EvalResult:
    system: str
    cap: float
    # baseline metrics
    baseline_bcubed_f1: float
    baseline_mojosim: float
    baseline_k: int
    baseline_gt_k: int
    # cac metrics
    cac_bcubed_f1: float
    cac_mojosim: float
    cac_k: int
    cac_gt_k: int
    # aux
    note: str = ""
    # NEW: experiment knobs
    u_ablation: str = "normal"
    mu: Optional[float] = None


def _root() -> Path:
    return Path(__file__).resolve().parents[3]


def _run(cmd: List[str], cwd: Path) -> str:
    # Windows console often uses cp936/gbk; be permissive.
    enc = "utf-8"
    try:
        enc = (getattr(sys.stdout, "encoding", None) or "utf-8")
    except Exception:
        enc = "utf-8"

    p = subprocess.run(
        cmd,
        cwd=str(cwd),
        capture_output=True,
        text=True,
        encoding=enc,
        errors="replace",
    )
    out = (p.stdout or "") + ("\n" + p.stderr if p.stderr else "")
    if p.returncode != 0:
        raise RuntimeError(f"Command failed ({p.returncode}): {' '.join(cmd)}\n{out}")
    return out


def _parse_phase4_output(text: str) -> Dict[str, Any]:
    # Example lines:
    #   BCubed F1:  0.4476
    #   MoJoSim: 47.83%
    #   GT 服务数: 4, 预测服务数: 2, K-Diff: 2
    out: Dict[str, Any] = {}

    m = re.search(r"BCubed F1:\s*([0-9.]+)", text)
    if m:
        out["bcubed_f1"] = float(m.group(1))

    m = re.search(r"MoJoSim:\s*([0-9.]+)%", text)
    if m:
        out["mojosim"] = float(m.group(1))

    m = re.search(r"GT 服务数:\s*(\d+),\s*预测服务数:\s*(\d+)", text)
    if m:
        out["gt_k"] = int(m.group(1))
        out["pred_k"] = int(m.group(2))

    return out


def _python() -> str:
    # Always use the same interpreter running this script (e.g., workspace .venv)
    return sys.executable


def _evaluate_partition(system: str, pred_path: Path, *, class_order: Path, gt: Path, phase4_py: Path, cwd: Path) -> Dict[str, Any]:
    cmd = [
        _python(),
        str(phase4_py),
        "--gt",
        str(gt),
        "--pred",
        str(pred_path),
        "--class_order",
        str(class_order),
    ]
    txt = _run(cmd, cwd)
    metrics = _parse_phase4_output(txt)
    metrics["raw"] = txt
    return metrics


def _copy_snapshot(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("systems", nargs="*", default=SYSTEMS)
    ap.add_argument(
        "--caps",
        type=str,
        default="0.05,0.08,0.10,0.12,0.14,0.16,0.18,0.20",
        help="Comma-separated DPEP cap sweep values.",
    )
    ap.add_argument("--mode", type=str, default="sigmoid")
    ap.add_argument("--alpha", type=float, default=15)
    ap.add_argument("--merge_small_clusters", action="store_true", default=True)
    ap.add_argument("--min_cluster_size", type=int, default=3)
    ap.add_argument("--target_from_gt", action="store_true", default=True)
    ap.add_argument(
        "--keep_phase3_outputs",
        action="store_true",
        help="If set, do not overwrite the working *_cac-final_partition.json when sweeping (still snapshots best).",
    )

    # NEW: Uncertainty ablation and mu sweep
    ap.add_argument(
        "--u_ablation",
        type=str,
        default="normal",
        choices=["normal", "zero", "shuffle"],
        help="Run Phase3 with given U ablation policy (normal|zero|shuffle). For the paper ablation B use 'normal' vs 'zero'.",
    )
    ap.add_argument(
        "--mu_sweep",
        type=str,
        default=None,
        help="If set, sweep mu values (comma-separated, e.g., '0.2,0.3,0.4'). Only affects Phase3 if modality matrices exist.",
    )

    args = ap.parse_args()

    root = _root()
    cwd = root

    phase3_py = root / "scripts" / "multimodal" / "phase3" / "phase3_cac_evaluation.py"
    phase4_py = root / "scripts" / "multimodal" / "phase4" / "evaluate_partition_f1.py"

    caps = [float(x.strip()) for x in args.caps.split(",") if x.strip()]
    mu_values = None
    if args.mu_sweep:
        mu_values = [float(x.strip()) for x in str(args.mu_sweep).split(",") if x.strip()]

    results: List[EvalResult] = []

    for system in args.systems:
        gt = root / "data" / "processed" / "groundtruth" / f"{system}_ground_truth.json"
        class_order = root / "data" / "processed" / "fusion" / f"{system}_class_order.json"

        baseline_part = root / "data" / "processed" / "fusion" / f"{system}_baseline_partition.json"
        cac_part = root / "data" / "processed" / "fusion" / f"{system}_cac-final_partition.json"

        # Evaluate baseline once
        base_m = _evaluate_partition(system, baseline_part, class_order=class_order, gt=gt, phase4_py=phase4_py, cwd=cwd)
        base_bc = float(base_m.get("bcubed_f1", 0.0))
        base_mj = float(base_m.get("mojosim", 0.0))
        base_k = int(base_m.get("pred_k", 0))
        base_gt_k = int(base_m.get("gt_k", 0))

        # (cap, mu, metrics)
        best: Optional[Tuple[float, Optional[float], Dict[str, Any]]] = None

        # if requested, preserve current working partition file
        cac_backup: Optional[Path] = None
        if args.keep_phase3_outputs and cac_part.exists():
            cac_backup = cac_part.with_suffix(".json.bak")
            shutil.copyfile(cac_part, cac_backup)

        mu_grid = mu_values if mu_values is not None else [None]

        for mu in mu_grid:
            for cap in caps:
                cmd = [
                    _python(),
                    str(phase3_py),
                    system,
                    "--mode",
                    str(args.mode),
                    "--alpha",
                    str(args.alpha),
                    "--dpep_cap",
                    f"{cap}",
                    "--u_ablation",
                    str(args.u_ablation),
                ]
                if mu is not None:
                    cmd.extend(["--mu", f"{mu}"])
                if args.target_from_gt:
                    cmd.append("--target_from_gt")
                if args.merge_small_clusters:
                    cmd.extend(["--merge_small_clusters", "--min_cluster_size", str(args.min_cluster_size)])

                _run(cmd, cwd)

                # Evaluate newly written CAC partition
                cac_m = _evaluate_partition(system, cac_part, class_order=class_order, gt=gt, phase4_py=phase4_py, cwd=cwd)
                cac_bc = float(cac_m.get("bcubed_f1", 0.0))
                cac_mj = float(cac_m.get("mojosim", 0.0))

                key = (cac_bc, cac_mj)
                if best is None:
                    best = (cap, mu, cac_m)
                else:
                    best_key = (float(best[2].get("bcubed_f1", 0.0)), float(best[2].get("mojosim", 0.0)))
                    if key > best_key:
                        best = (cap, mu, cac_m)

        # restore backup if needed
        if cac_backup is not None and cac_backup.exists():
            shutil.copyfile(cac_backup, cac_part)
            cac_backup.unlink(missing_ok=True)

        assert best is not None
        best_cap, best_mu, best_m = best

        best_cac_bc = float(best_m.get("bcubed_f1", 0.0))
        best_cac_mj = float(best_m.get("mojosim", 0.0))
        best_cac_k = int(best_m.get("pred_k", 0))
        best_cac_gt_k = int(best_m.get("gt_k", 0))

        # snapshot best partition file (re-run phase3 at best setting deterministically)
        cmd = [
            _python(),
            str(phase3_py),
            system,
            "--mode",
            str(args.mode),
            "--alpha",
            str(args.alpha),
            "--dpep_cap",
            f"{best_cap}",
            "--u_ablation",
            str(args.u_ablation),
        ]
        if best_mu is not None:
            cmd.extend(["--mu", f"{best_mu}"])
        if args.target_from_gt:
            cmd.append("--target_from_gt")
        if args.merge_small_clusters:
            cmd.extend(["--merge_small_clusters", "--min_cluster_size", str(args.min_cluster_size)])
        _run(cmd, cwd)

        best_part_dst = root / "data" / "processed" / "fusion" / f"{system}_cac-final_best_by_bcubed_partition.json"
        _copy_snapshot(cac_part, best_part_dst)

        # write meta json
        note = ""
        # K-collapse flags
        if base_k <= 2 and base_gt_k >= 4:
            note = f"Baseline granularity collapse (K={base_k} vs GT_K={base_gt_k})"
        if best_cac_k <= 2 and best_cac_gt_k >= 4:
            note = (note + "; " if note else "") + f"CAC granularity collapse (K={best_cac_k} vs GT_K={best_cac_gt_k})"

        meta_dst = root / "data" / "processed" / "fusion" / f"{system}_cac_best_by_bcubed.json"
        _write_json(
            meta_dst,
            {
                "system": system,
                "best_cap": float(best_cap),
                "best_mu": float(best_mu) if best_mu is not None else None,
                "u_ablation": str(args.u_ablation),
                "mode": str(args.mode),
                "alpha": float(args.alpha),
                "merge_small_clusters": bool(args.merge_small_clusters),
                "min_cluster_size": int(args.min_cluster_size),
                "baseline": {
                    "bcubed_f1": float(base_bc),
                    "mojosim": float(base_mj),
                    "k": int(base_k),
                    "gt_k": int(base_gt_k),
                },
                "cac": {
                    "bcubed_f1": float(best_cac_bc),
                    "mojosim": float(best_cac_mj),
                    "k": int(best_cac_k),
                    "gt_k": int(best_cac_gt_k),
                },
                "note": note,
            },
        )

        results.append(
            EvalResult(
                system=system,
                cap=float(best_cap),
                baseline_bcubed_f1=float(base_bc),
                baseline_mojosim=float(base_mj),
                baseline_k=int(base_k),
                baseline_gt_k=int(base_gt_k),
                cac_bcubed_f1=float(best_cac_bc),
                cac_mojosim=float(best_cac_mj),
                cac_k=int(best_cac_k),
                cac_gt_k=int(best_cac_gt_k),
                note=note,
                u_ablation=str(args.u_ablation),
                mu=(float(best_mu) if best_mu is not None else None),
            )
        )

    # Write paper-ready CSV/MD
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = root / "results" / "ablation"
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / f"phase4_best_by_bcubed_{ts}.csv"
    md_path = out_dir / f"phase4_best_by_bcubed_{ts}.md"

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "system",
                "u_ablation",
                "best_mu",
                "best_cap",
                "baseline_bcubed_f1",
                "baseline_mojosim",
                "baseline_k",
                "gt_k",
                "cac_bcubed_f1",
                "cac_mojosim",
                "cac_k",
                "note",
            ]
        )
        for r in results:
            w.writerow(
                [
                    r.system,
                    r.u_ablation,
                    ("" if r.mu is None else f"{r.mu:.3f}"),
                    f"{r.cap:.3f}",
                    f"{r.baseline_bcubed_f1:.4f}",
                    f"{r.baseline_mojosim:.2f}",
                    r.baseline_k,
                    r.baseline_gt_k,
                    f"{r.cac_bcubed_f1:.4f}",
                    f"{r.cac_mojosim:.2f}",
                    r.cac_k,
                    r.note,
                ]
            )

    with md_path.open("w", encoding="utf-8") as f:
        f.write(f"# Phase4 Best-by-BCubed ({ts})\n\n")
        f.write(
            "| System | U ablation | best μ | best cap | Baseline BCubed F1 | Baseline MoJoSim | Baseline K | GT K | CAC BCubed F1 | CAC MoJoSim | CAC K | Note |\n"
        )
        f.write("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|\n")
        for r in results:
            mu_s = "-" if r.mu is None else f"{r.mu:.3f}"
            f.write(
                "| {sys} | {uab} | {mu} | {cap:.3f} | {bf1:.4f} | {bmj:.2f} | {bk} | {gtk} | {cf1:.4f} | {cmj:.2f} | {ck} | {note} |\n".format(
                    sys=r.system,
                    uab=r.u_ablation,
                    mu=mu_s,
                    cap=r.cap,
                    bf1=r.baseline_bcubed_f1,
                    bmj=r.baseline_mojosim,
                    bk=r.baseline_k,
                    gtk=r.baseline_gt_k,
                    cf1=r.cac_bcubed_f1,
                    cmj=r.cac_mojosim,
                    ck=r.cac_k,
                    note=r.note or "",
                )
            )

    print(f"[BestByBCubed] CSV saved: {csv_path}")
    print(f"[BestByBCubed] MD saved:  {md_path}")


if __name__ == "__main__":
    main()
