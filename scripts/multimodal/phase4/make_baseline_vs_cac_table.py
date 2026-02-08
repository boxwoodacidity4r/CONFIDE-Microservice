"""Build a baseline-vs-CAC comparison table for Phase4.

This script runs evaluate_partition_f1.py for both baseline and cac-final partitions
and produces a single CSV/Markdown table for paper writing.

Usage:
  python scripts/multimodal/phase4/make_baseline_vs_cac_table.py

Outputs:
  results/phase4_baseline_vs_cac.csv
  results/phase4_baseline_vs_cac.md
"""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path


SYSTEMS = ["acmeair", "daytrader", "jpetstore", "plants"]

ROOT = Path(__file__).resolve().parents[3]
EVAL = ROOT / "scripts" / "multimodal" / "phase4" / "evaluate_partition_f1.py"

OUT_CSV = ROOT / "results" / "phase4_baseline_vs_cac.csv"
OUT_MD = ROOT / "results" / "phase4_baseline_vs_cac.md"


def run_eval(gt: Path, pred: Path, out_json: Path, dep: Path | None = None):
    cmd = [sys.executable, str(EVAL), "--gt", str(gt), "--pred", str(pred), "--out_json", str(out_json)]
    if dep is not None and dep.exists():
        cmd.extend(["--dep", str(dep)])
    subprocess.run(cmd, check=True)


def main():
    tmp = ROOT / "results" / "_tmp_phase4"
    tmp.mkdir(parents=True, exist_ok=True)

    rows = []

    for sys_name in SYSTEMS:
        gt = ROOT / "data" / "processed" / "groundtruth" / f"{sys_name}_ground_truth.json"
        base_pred = ROOT / "data" / "processed" / "fusion" / f"{sys_name}_baseline_partition.json"
        cac_pred = ROOT / "data" / "processed" / "fusion" / f"{sys_name}_cac-final_partition.json"
        dep = ROOT / "data" / "processed" / "dependency" / f"{sys_name}_dependency_matrix.json"

        out_base = tmp / f"{sys_name}_baseline.json"
        out_cac = tmp / f"{sys_name}_cac-final.json"

        run_eval(gt, base_pred, out_base, dep=dep)
        run_eval(gt, cac_pred, out_cac, dep=dep)

        mb = json.loads(out_base.read_text(encoding="utf-8"))
        mc = json.loads(out_cac.read_text(encoding="utf-8"))

        rows.append({
            "system": sys_name,
            "baseline_f1": mb.get("f1"),
            "cac_f1": mc.get("f1"),
            "delta_f1": (mc.get("f1") - mb.get("f1")) if (mb.get("f1") is not None and mc.get("f1") is not None) else None,
            "baseline_mojosim": mb.get("mojosim"),
            "cac_mojosim": mc.get("mojosim"),
            "delta_mojosim": (mc.get("mojosim") - mb.get("mojosim")) if (mb.get("mojosim") is not None and mc.get("mojosim") is not None) else None,
            "gt_k": mc.get("gt_k"),
            "baseline_k": mb.get("pred_k"),
            "cac_k": mc.get("pred_k"),
            "has_dep": bool(dep.exists()),
        })

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # Markdown table
    def fmt(x, nd=4):
        if x is None:
            return "-"
        if isinstance(x, bool):
            return "yes" if x else "no"
        if isinstance(x, int):
            return str(x)
        try:
            return f"{float(x):.{nd}f}"
        except Exception:
            return str(x)

    md_lines = []
    hdr = [
        "System",
        "Baseline F1",
        "CAC F1",
        "ΔF1",
        "Baseline MoJoSim",
        "CAC MoJoSim",
        "ΔMoJoSim",
        "GT k",
        "Baseline k",
        "CAC k",
        "Dep?",
    ]
    md_lines.append("| " + " | ".join(hdr) + " |")
    md_lines.append("|" + "|".join(["---"] * len(hdr)) + "|")
    for r in rows:
        md_lines.append(
            "| "
            + " | ".join(
                [
                    r["system"],
                    fmt(r["baseline_f1"]),
                    fmt(r["cac_f1"]),
                    fmt(r["delta_f1"]),
                    fmt(r["baseline_mojosim"], nd=2),
                    fmt(r["cac_mojosim"], nd=2),
                    fmt(r["delta_mojosim"], nd=2),
                    fmt(r["gt_k"], nd=0),
                    fmt(r["baseline_k"], nd=0),
                    fmt(r["cac_k"], nd=0),
                    fmt(r["has_dep"]),
                ]
            )
            + " |"
        )

    OUT_MD.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    print(f"[OK] Written: {OUT_CSV}")
    print(f"[OK] Written: {OUT_MD}")


if __name__ == "__main__":
    main()
