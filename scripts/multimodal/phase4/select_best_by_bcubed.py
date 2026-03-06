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


# NEW: one row per sweep point (cap, mu) for sensitivity plots
@dataclass
class SweepPoint:
    system: str
    u_ablation: str
    mode: str
    alpha: float
    merge_small_clusters: bool
    min_cluster_size: int
    target_from_gt: bool
    mu: Optional[float]
    cap: float
    baseline_bcubed_f1: float
    baseline_mojosim: float
    baseline_k: int
    baseline_gt_k: int
    cac_bcubed_f1: float
    cac_mojosim: float
    cac_k: int
    cac_gt_k: int


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
    """Parse Phase4 evaluation output robustly.

    The Phase4 script is used both interactively and in batch runs. Its stdout can vary
    (English/Chinese, with/without percentage signs, or printing a JSON dict). This
    parser tries multiple strategies and normalizes to:
      - bcubed_f1 (float)
      - mojosim (float)
      - gt_k (int)
      - pred_k (int)

    It is intentionally best-effort: missing fields are left absent.
    """

    out: Dict[str, Any] = {}

    # (1) Best-effort: recover a printed JSON metrics object.
    # We only accept dict-like JSON that contains at least one target key.
    try:
        for cand in reversed(re.findall(r"\{[\s\S]*?\}", text)):
            try:
                obj = json.loads(cand)
            except Exception:
                continue
            if not isinstance(obj, dict):
                continue

            hit = False
            # normalize common variants
            key_map = {
                "bcubed_f1": "bcubed_f1",
                "BCubedF1": "bcubed_f1",
                "mojosim": "mojosim",
                "MoJoSim": "mojosim",
                "gt_k": "gt_k",
                "GT_K": "gt_k",
                "gtK": "gt_k",
                "pred_k": "pred_k",
                "Pred_K": "pred_k",
                "predK": "pred_k",
                "k": "pred_k",
            }
            for k, v in obj.items():
                if k in key_map and v is not None:
                    out[key_map[k]] = v
                    hit = True
            if hit:
                break
    except Exception:
        pass

    # (2) Regex fallback: BCubed F1
    if "bcubed_f1" not in out:
        m = re.search(
            r"BCubed\s*F\s*1\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)",
            text,
            flags=re.IGNORECASE,
        )
        if m:
            out["bcubed_f1"] = float(m.group(1))

    # (3) Regex fallback: MoJoSim
    if "mojosim" not in out:
        m = re.search(
            r"MoJoSim\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)\s*%?",
            text,
            flags=re.IGNORECASE,
        )
        if m:
            out["mojosim"] = float(m.group(1))

    # (4) Regex fallback: K values
    if "gt_k" not in out or "pred_k" not in out:
        patterns = [
            # English
            r"ground\s*truth\s*(?:service\s*)?count\s*[:=]\s*(\d+)\s*[,:]?\s*(?:pred|predict(?:ed)?)\s*(?:service\s*)?count\s*[:=]\s*(\d+)",
            r"GT\s*K\s*[:=]\s*(\d+)\s*[,:]?\s*(?:Pred|Predict(?:ed)?)\s*K\s*[:=]\s*(\d+)",
            r"GT[_\s]*K\s*[:=]\s*(\d+)\s*[,:]?\s*Pred[_\s]*K\s*[:=]\s*(\d+)",
            # Very permissive: "GT: 5 Pred: 6" on the same line
            r"\bGT\b\s*[:=]\s*(\d+)\s+\bPred\b\s*[:=]\s*(\d+)",
        ]
        for pat in patterns:
            m = re.search(pat, text, flags=re.IGNORECASE)
            if m:
                out.setdefault("gt_k", int(m.group(1)))
                out.setdefault("pred_k", int(m.group(2)))
                break

    # Coerce numeric types
    if "bcubed_f1" in out:
        try:
            out["bcubed_f1"] = float(out["bcubed_f1"])
        except Exception:
            pass
    if "mojosim" in out:
        try:
            out["mojosim"] = float(out["mojosim"])
        except Exception:
            pass
    if "gt_k" in out:
        try:
            out["gt_k"] = int(out["gt_k"])
        except Exception:
            pass
    if "pred_k" in out:
        try:
            out["pred_k"] = int(out["pred_k"])
        except Exception:
            pass

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

    # NEW: persist all sweep points for sensitivity plots
    ap.add_argument(
        "--save_sweep_points",
        action="store_true",
        default=True,
        help="Save all (cap,mu) evaluation points to results/ablation/phase4_cap_sweep_all_scores_*.csv.",
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

    # NEW: map Phase4 script's naming to Phase3 CLI
    # phase3 expects: with_u | no_u | shuffle
    u_map = {
        "normal": "with_u",
        "zero": "no_u",
        "shuffle": "shuffle",
    }
    phase3_u_ablation = u_map.get(str(args.u_ablation).lower().strip(), "with_u")

    results: List[EvalResult] = []
    sweep_points: List[SweepPoint] = []

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
                    str(phase3_u_ablation),
                ]
                if mu is not None:
                    # phase3 uses --mu_override
                    cmd.extend(["--mu_override", f"{mu}"])
                if args.target_from_gt:
                    cmd.append("--target_from_gt")
                if args.merge_small_clusters:
                    cmd.extend(["--merge_small_clusters", "--min_cluster_size", str(args.min_cluster_size)])

                _run(cmd, cwd)

                # Evaluate newly written CAC partition
                cac_m = _evaluate_partition(
                    system,
                    cac_part,
                    class_order=class_order,
                    gt=gt,
                    phase4_py=phase4_py,
                    cwd=cwd,
                )
                cac_bc = float(cac_m.get("bcubed_f1", 0.0))
                cac_mj = float(cac_m.get("mojosim", 0.0))
                cac_k = int(cac_m.get("pred_k", 0))
                cac_gt_k = int(cac_m.get("gt_k", 0))

                # NEW: record every cap/mu point for sensitivity plots
                sweep_points.append(
                    SweepPoint(
                        system=system,
                        u_ablation=str(args.u_ablation),
                        mode=str(args.mode),
                        alpha=float(args.alpha),
                        merge_small_clusters=bool(args.merge_small_clusters),
                        min_cluster_size=int(args.min_cluster_size),
                        target_from_gt=bool(args.target_from_gt),
                        mu=(float(mu) if mu is not None else None),
                        cap=float(cap),
                        baseline_bcubed_f1=float(base_bc),
                        baseline_mojosim=float(base_mj),
                        baseline_k=int(base_k),
                        baseline_gt_k=int(base_gt_k),
                        cac_bcubed_f1=float(cac_bc),
                        cac_mojosim=float(cac_mj),
                        cac_k=int(cac_k),
                        cac_gt_k=int(cac_gt_k),
                    )
                )

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
            str(phase3_u_ablation),
        ]
        if best_mu is not None:
            cmd.extend(["--mu_override", f"{best_mu}"])
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

    # NEW: write all sweep points for cap sensitivity / plotting
    if args.save_sweep_points:
        sweep_csv = out_dir / f"phase4_cap_sweep_all_scores_{ts}.csv"
        with sweep_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(
                [
                    "system",
                    "u_ablation",
                    "mode",
                    "alpha",
                    "merge_small_clusters",
                    "min_cluster_size",
                    "target_from_gt",
                    "mu",
                    "cap",
                    "baseline_bcubed_f1",
                    "baseline_mojosim",
                    "baseline_k",
                    "baseline_gt_k",
                    "cac_bcubed_f1",
                    "cac_mojosim",
                    "cac_k",
                    "cac_gt_k",
                ]
            )
            for p in sweep_points:
                w.writerow(
                    [
                        p.system,
                        p.u_ablation,
                        p.mode,
                        f"{p.alpha:.3f}",
                        int(p.merge_small_clusters),
                        p.min_cluster_size,
                        int(p.target_from_gt),
                        "" if p.mu is None else f"{p.mu:.3f}",
                        f"{p.cap:.3f}",
                        f"{p.baseline_bcubed_f1:.4f}",
                        f"{p.baseline_mojosim:.2f}",
                        p.baseline_k,
                        p.baseline_gt_k,
                        f"{p.cac_bcubed_f1:.4f}",
                        f"{p.cac_mojosim:.2f}",
                        p.cac_k,
                        p.cac_gt_k,
                    ]
                )
        print(f"[BestByBCubed] Sweep points CSV saved: {sweep_csv}")

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
