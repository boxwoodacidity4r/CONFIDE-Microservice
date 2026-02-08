"""Phase4: run mono baselines + simple fusion + ours table (paper-ready).

Rows:
- Pure Semantic (semantic-only clustering using S_sem_dade)
- Pure Structural (structural-only clustering using S_struct)
- Simple Fusion (S_final with U≡0)  [i.e., CAC without uncertainty]
- Ours (CAC with U)

For each row, we optionally K-lock to GT_K by using --target_from_gt in Phase3.

Outputs:
- results/ablation/mono_baselines_vs_ours_<system>_<ts>.csv/.md

Note: This script uses Phase3 for clustering and Phase4 for evaluation.
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

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
PHASE3 = ROOT / "scripts" / "multimodal" / "phase3" / "phase3_cac_evaluation.py"
PHASE4 = ROOT / "scripts" / "multimodal" / "phase4" / "evaluate_partition_f1.py"


def _python() -> str:
    return sys.executable


def _run(cmd: List[str]) -> str:
    p = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True, encoding="utf-8", errors="replace")
    out = (p.stdout or "") + ("\n" + p.stderr if p.stderr else "")
    if p.returncode != 0:
        raise RuntimeError(f"Command failed ({p.returncode}): {' '.join(cmd)}\n{out}")
    return out


def _eval(system: str, pred: Path) -> Dict[str, Any]:
    gt = ROOT / "data" / "processed" / "groundtruth" / f"{system}_ground_truth.json"
    order = ROOT / "data" / "processed" / "fusion" / f"{system}_class_order.json"
    out_json = ROOT / "results" / "ablation" / "_tmp" / f"metrics_{system}.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)

    _run([
        _python(),
        str(PHASE4),
        "--gt",
        str(gt),
        "--pred",
        str(pred),
        "--class_order",
        str(order),
        "--out_json",
        str(out_json),
    ])

    m = json.loads(out_json.read_text(encoding="utf-8"))

    # --- A 补强：K 值兜底（彻底消灭 K=0 / 缺失） ---
    def _count_k(mapping: Dict[str, Any]) -> int:
        if not isinstance(mapping, dict) or not mapping:
            return 0
        try:
            return int(len(set(int(v) for v in mapping.values())))
        except Exception:
            return int(len(set(str(v) for v in mapping.values())))

    # metrics.json 若缺 pred_k / gt_k，则直接从 pred/gt 映射统计
    if not m.get("pred_k"):
        try:
            pred_map = json.loads(Path(pred).read_text(encoding="utf-8"))
            m["pred_k"] = _count_k(pred_map)
        except Exception:
            pass
    if not m.get("gt_k"):
        try:
            gt_map = json.loads(Path(gt).read_text(encoding="utf-8"))
            m["gt_k"] = _count_k(gt_map)
        except Exception:
            pass

    # 最后兜底：保证是 int
    m["pred_k"] = int(m.get("pred_k", 0) or 0)
    m["gt_k"] = int(m.get("gt_k", 0) or 0)
    return m


def _phase3(system: str, *, cap: float, u_ablation: str, mu_override: float | None, target_from_gt: bool) -> Path:
    cmd = [_python(), str(PHASE3), system, "--mode", "sigmoid", "--alpha", "15", "--dpep_cap", str(cap), "--u_ablation", u_ablation]
    if target_from_gt:
        cmd.append("--target_from_gt")
    cmd += ["--merge_small_clusters", "--min_cluster_size", "3"]
    if mu_override is not None:
        cmd += ["--mu_override", str(mu_override)]
    _run(cmd)
    return ROOT / "data" / "processed" / "fusion" / f"{system}_cac-final_partition.json"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("system", type=str, default="daytrader")
    ap.add_argument("--cap", type=float, default=0.16)
    ap.add_argument("--mu_daytrader", type=float, default=0.30, help="Use the best mu_override discovered for DayTrader.")
    ap.add_argument("--k_lock", action="store_true", default=True, help="Use --target_from_gt for all methods.")
    args = ap.parse_args()

    system = args.system.lower().strip()
    cap = float(args.cap)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = ROOT / "results" / "ablation"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []

    # Pure semantic: approximate via mu_override=0 (use S_sem only) but our mu_override mixes struct+sem.
    # For simplicity (and no Phase1 rerun), we simulate:
    #   semantic-only => S = (1-mu)*S_sem + mu*S_struct with mu=0
    #   structural-only => mu=1
    # Simple fusion => use persisted S_final but U≡0 (u_ablation=no_u)
    # Ours => with U; for DayTrader we also set mu_override to robust point.

    configs = [
        ("PureSemantic", "with_u", 0.0),
        ("PureStructural", "with_u", 1.0),
        ("SimpleFusion_noU", "no_u", None),
        ("Ours_CAC_withU", "with_u", (args.mu_daytrader if system == "daytrader" else None)),
    ]

    for name, u_ab, mu in configs:
        pred = _phase3(system, cap=cap, u_ablation=u_ab, mu_override=mu, target_from_gt=bool(args.k_lock))
        m = _eval(system, pred)
        rows.append(
            {
                "system": system,
                "method": name,
                "cap": cap,
                "u_ablation": u_ab,
                "mu_override": ("-" if mu is None else float(mu)),
                "bcubed_f1": float(m.get("bcubed_f1", 0.0)),
                "mojosim": float(m.get("mojosim", 0.0)),
                "pred_k": int(m.get("pred_k", 0)),
                "gt_k": int(m.get("gt_k", 0)),
                "k_diff": int(m.get("pred_k", 0)) - int(m.get("gt_k", 0)),
            }
        )

    csv_path = out_dir / f"mono_baselines_vs_ours_{system}_{ts}.csv"
    md_path = out_dir / f"mono_baselines_vs_ours_{system}_{ts}.md"

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    with md_path.open("w", encoding="utf-8") as f:
        f.write(f"# Mono baselines vs ours ({system})\n\n")
        f.write(f"Generated: {ts}\n\n")
        f.write("| Method | BCubedF1 | MoJoSim | K | GT_K | K-Diff | mu_override | U |\n")
        f.write("|---|---:|---:|---:|---:|---:|---:|---|\n")
        for r in rows:
            f.write(
                "| {m} | {f1:.4f} | {mj:.2f} | {k} | {gtk} | {kd:+d} | {mu} | {u} |\n".format(
                    m=r["method"],
                    f1=float(r["bcubed_f1"]),
                    mj=float(r["mojosim"]),
                    k=int(r["pred_k"]),
                    gtk=int(r["gt_k"]),
                    kd=int(r["k_diff"]),
                    mu=str(r["mu_override"]),
                    u=str(r["u_ablation"]),
                )
            )

        f.write("\n## Notes\n")
        f.write("- PureSemantic/PureStructural are approximated via mu_override=0/1 using existing S_sem and S_struct.\n")
        f.write("- SimpleFusion_noU is CAC with U≡0 (uncertainty disabled) over persisted S_final.\n")
        f.write("- K-lock uses --target_from_gt to keep service granularity comparable.\n")

    print(f"[OK] saved: {csv_path}")
    print(f"[OK] saved: {md_path}")


if __name__ == "__main__":
    main()
