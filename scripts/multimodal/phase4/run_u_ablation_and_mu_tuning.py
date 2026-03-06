"""Run paper-ready experiments:

1) Uncertainty ablation (B): CAC with U vs CAC with U≡0
2) DayTrader mu weight tuning (best-effort) + K-collapse check
3) Case study: dump 'most uncertain edges' evidence for visual inspection

Outputs:
- results/ablation/u_ablation_B_<ts>.csv / .md
- results/ablation/daytrader_mu_sweep_<ts>.csv / .md
- results/ablation/case_study_edges_<system>_<variant>.jsonl

This script does NOT require rerunning Phase1/Phase2.
"""

from __future__ import annotations

import csv
import json
import subprocess
import sys
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[3]
PHASE3 = ROOT / "scripts" / "multimodal" / "phase3" / "phase3_cac_evaluation.py"
PHASE4 = ROOT / "scripts" / "multimodal" / "phase4" / "evaluate_partition_f1.py"

SYSTEMS = ["acmeair", "daytrader", "plants", "jpetstore"]


def _run(cmd: List[str]) -> str:
    enc = getattr(sys.stdout, "encoding", None) or "utf-8"
    p = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True, encoding=enc, errors="replace")
    out = (p.stdout or "") + ("\n" + p.stderr if p.stderr else "")
    if p.returncode != 0:
        raise RuntimeError(f"Command failed ({p.returncode}): {' '.join(cmd)}\n{out}")
    return out


def _eval_phase4(system: str, pred_path: Path) -> Dict[str, Any]:
    gt = ROOT / "data" / "processed" / "groundtruth" / f"{system}_ground_truth.json"
    class_order = ROOT / "data" / "processed" / "fusion" / f"{system}_class_order.json"
    cmd = [sys.executable, str(PHASE4), "--gt", str(gt), "--pred", str(pred_path), "--class_order", str(class_order)]
    txt = _run(cmd)

    def grab_num(prefix: str) -> float:
        # robust to suffix annotations, e.g., "BCubed F1: 0.4476 (xxx)"
        for line in txt.splitlines():
            if line.strip().startswith(prefix):
                m = re.search(r"([0-9]+\.[0-9]+|[0-9]+)", line.split(":", 1)[1])
                if m:
                    return float(m.group(1))
        return 0.0

    bc_f1 = grab_num("BCubed F1")
    mojosim = grab_num("MoJoSim")

    # parse K info (robust regex)
    gt_k = 0
    pred_k = 0
    # Expected line pattern (English output): "#GT services: X, #Pred services: Y, K-Diff: Z"
    # Use a tolerant regex fallback to keep backward compatibility with older localized outputs.
    m = re.search(r"#GT\s*services\s*:\s*(\d+)\s*,\s*#Pred\s*services\s*:\s*(\d+)", txt)
    if not m:
        m = re.search(r"GT\s*services\s*:\s*(\d+)\s*,\s*pred(?:icted)?\s*services\s*:\s*(\d+)", txt, flags=re.IGNORECASE)
    if not m:
        # legacy Chinese output (deprecated)
        m = re.search(r"GT\s*\u670d\u52a1\u6570\s*:\s*(\d+)\s*,\s*\u9884\u6d4b\u670d\u52a1\u6570\s*:\s*(\d+)", txt.replace("\uff0c", ","))
    if m:
        gt_k = int(m.group(1))
        pred_k = int(m.group(2))

    return {"bcubed_f1": bc_f1, "mojosim": mojosim, "gt_k": gt_k, "pred_k": pred_k, "raw": txt}


def _run_phase3(system: str, *, cap: float, u_ablation: str, mu: Optional[float], dump_edges: Optional[Path]) -> None:
    cmd = [
        sys.executable,
        str(PHASE3),
        system,
        "--mode",
        "sigmoid",
        "--alpha",
        "15",
        "--dpep_cap",
        f"{cap}",
        "--target_from_gt",
        "--merge_small_clusters",
        "--min_cluster_size",
        "3",
        "--u_ablation",
        u_ablation,
    ]
    if mu is not None:
        cmd += ["--mu", f"{mu}"]
    if dump_edges is not None:
        cmd += ["--dump_edge_evidence", str(dump_edges), "--edge_evidence_topk", "30"]
    _run(cmd)


def main() -> None:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = ROOT / "results" / "ablation"
    out_dir.mkdir(parents=True, exist_ok=True)

    # (A) Uncertainty ablation B: U normal vs U zero
    caps_default = {"acmeair": 0.14, "daytrader": 0.18, "plants": 0.22, "jpetstore": 0.14}

    ab_rows = []
    for system in SYSTEMS:
        cap = float(caps_default[system])

        for variant in ("normal", "zero"):
            edges_path = out_dir / f"case_study_edges_{system}_{variant}_{ts}.jsonl"
            _run_phase3(system, cap=cap, u_ablation=variant, mu=None, dump_edges=edges_path)
            pred = ROOT / "data" / "processed" / "fusion" / f"{system}_cac-final_partition.json"
            m = _eval_phase4(system, pred)
            ab_rows.append(
                {
                    "system": system,
                    "variant": variant,
                    "cap": cap,
                    "bcubed_f1": m["bcubed_f1"],
                    "mojosim": m["mojosim"],
                    "pred_k": m["pred_k"],
                    "gt_k": m["gt_k"],
                    "k_collapse": int(m["pred_k"] <= 2 and m["gt_k"] >= 4),
                    "edge_evidence": str(edges_path),
                }
            )

    csv_path = out_dir / f"u_ablation_B_{ts}.csv"
    md_path = out_dir / f"u_ablation_B_{ts}.md"

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "system",
                "variant",
                "cap",
                "bcubed_f1",
                "mojosim",
                "pred_k",
                "gt_k",
                "k_collapse",
                "edge_evidence",
            ],
        )
        w.writeheader()
        w.writerows(ab_rows)

    # Markdown table
    with md_path.open("w", encoding="utf-8") as f:
        f.write(f"# Uncertainty Ablation (B): CAC with U vs CAC with U≡0 ({ts})\n\n")
        f.write("| System | Variant | cap | BCubed F1 | MoJoSim | pred K | GT K | K-collapse | Edge evidence |\n")
        f.write("|---|---|---:|---:|---:|---:|---:|---:|---|\n")
        for r in ab_rows:
            f.write(
                "| {system} | {variant} | {cap:.2f} | {bcubed_f1:.4f} | {mojosim:.2f} | {pred_k} | {gt_k} | {k_collapse} | `{edge_evidence}` |\n".format(
                    **r
                )
            )

        f.write("\n## How to use edge evidence (case study)\n")
        f.write(
            "Each JSONL record contains a class pair (i,j) with high uncertainty U, plus S/U/W. "
            "Compare `variant=normal` vs `variant=zero`: with U enabled, W should be reduced for conflicting signals, "
            "preventing semantic-noise-induced merges.\n"
        )

    print(f"[AblationB] CSV saved: {csv_path}")
    print(f"[AblationB] MD saved:  {md_path}")

    # (B) DayTrader mu sweep (best-effort; only works if modality matrices exist). Still useful for K-collapse diagnosis.
    mu_grid = [0.10, 0.20, 0.30, 0.40, 0.50]
    cap = float(caps_default["daytrader"])

    mu_rows = []
    for mu in mu_grid:
        _run_phase3("daytrader", cap=cap, u_ablation="normal", mu=float(mu), dump_edges=None)
        pred = ROOT / "data" / "processed" / "fusion" / "daytrader_cac-final_partition.json"
        m = _eval_phase4("daytrader", pred)
        mu_rows.append(
            {
                "mu": mu,
                "cap": cap,
                "bcubed_f1": m["bcubed_f1"],
                "mojosim": m["mojosim"],
                "pred_k": m["pred_k"],
                "gt_k": m["gt_k"],
                "k_collapse": int(m["pred_k"] <= 2 and m["gt_k"] >= 4),
            }
        )

    mu_csv = out_dir / f"daytrader_mu_sweep_{ts}.csv"
    mu_md = out_dir / f"daytrader_mu_sweep_{ts}.md"

    with mu_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(mu_rows[0].keys()))
        w.writeheader()
        w.writerows(mu_rows)

    with mu_md.open("w", encoding="utf-8") as f:
        f.write(f"# DayTrader μ Sweep ({ts})\n\n")
        f.write("| μ (semantic weight) | cap | BCubed F1 | MoJoSim | pred K | GT K | K-collapse |\n")
        f.write("|---:|---:|---:|---:|---:|---:|---:|\n")
        for r in mu_rows:
            f.write(
                "| {mu:.2f} | {cap:.2f} | {bcubed_f1:.4f} | {mojosim:.2f} | {pred_k} | {gt_k} | {k_collapse} |\n".format(
                    **r
                )
            )

    print(f"[MuSweep] CSV saved: {mu_csv}")
    print(f"[MuSweep] MD saved:  {mu_md}")


if __name__ == "__main__":
    main()
