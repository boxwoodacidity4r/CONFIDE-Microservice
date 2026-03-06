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
ABL = ROOT / "results" / "ablation" / "baseline"


def _pick_latest_csv(system: str) -> Path:
    # Prefer stable filenames (no timestamp) produced by run_mono_baselines_and_ours_table.py
    stable = ABL / f"mono_baselines_vs_ours_{system}.csv"
    if stable.exists():
        return stable

    # Backward compatibility: older runs with timestamps in filename
    pats = sorted(ABL.glob(f"mono_baselines_vs_ours_{system}_*.csv"))
    if not pats:
        raise FileNotFoundError(f"No CSV found for system={system}. Run run_mono_baselines_and_ours_table.py first.")
    return pats[-1]


def _to_float(x: object) -> float | None:
    if x is None:
        return None
    s = str(x).strip()
    if s == "" or s == "-":
        return None
    try:
        return float(s)
    except Exception:
        return None


def _mean(xs: List[float]) -> float | None:
    if not xs:
        return None
    return sum(xs) / float(len(xs))


def _std(xs: List[float]) -> float | None:
    if len(xs) < 2:
        return 0.0 if xs else None
    m = _mean(xs)
    if m is None:
        return None
    var = sum((v - m) ** 2 for v in xs) / float(len(xs) - 1)
    return var ** 0.5


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

    # Compute per-method summary across systems
    methods = sorted({r.get("method", "") for r in rows if r.get("method")})
    metric_keys = [
        "bcubed_f1",
        "mojosim",
        "ifn",
        "ned",
        "sm",
        "icp",
        "pred_k",
        "gt_k",
        "k_diff",
    ]

    summary_mean: Dict[str, Dict[str, float | None]] = {}
    summary_std: Dict[str, Dict[str, float | None]] = {}
    for m in methods:
        sub = [r for r in rows if r.get("method") == m]
        mm: Dict[str, float | None] = {}
        ss: Dict[str, float | None] = {}
        for k in metric_keys:
            vals = [_to_float(r.get(k)) for r in sub]
            vals2 = [v for v in vals if v is not None]
            mm[k] = _mean(vals2)
            ss[k] = _std(vals2)
        summary_mean[m] = mm
        summary_std[m] = ss

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    # NOTE: stable filenames (no timestamp) to keep artifact folder tidy
    out_csv = ABL / "mono_baselines_vs_ours_ALL.csv"
    out_md = ABL / "mono_baselines_vs_ours_ALL.md"

    # Robust fieldnames: union across all rows (older CSVs may miss newer columns like pred_path)
    field_set = set()
    for r in rows:
        field_set.update(r.keys())

    # Prefer a stable column order (if present)
    preferred = [
        "system",
        "method",
        "cap",
        "u_ablation",
        "mu_override",
        "pred_path",
        "bcubed_f1",
        "mojosim",
        "pred_k",
        "gt_k",
        "k_diff",
        "ifn",
        "ned",
        "sm",
        "icp",
    ]
    fieldnames = [k for k in preferred if k in field_set] + sorted([k for k in field_set if k not in preferred])

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    # markdown
    with out_md.open("w", encoding="utf-8") as f:
        f.write("# Mono baselines vs ours (ALL systems)\n\n")
        f.write(f"Generated: (no timestamp; deterministic overwrite)\n\n")
        f.write("| System | Method | BCubedF1 | MoJoSim | IFN | NED | SM | ICP | K | GT_K | K-Diff | mu_override | U | cap |\n")
        f.write("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|\n")

        for r in rows:
            def _f(key: str, nd: int = 4) -> str:
                v = r.get(key)
                if v is None or v == "":
                    return "-"
                try:
                    if key in ("pred_k", "gt_k", "k_diff"):
                        return str(int(float(v)))
                    if key in ("mojosim", "ifn"):
                        return f"{float(v):.2f}"
                    return f"{float(v):.{nd}f}"
                except Exception:
                    return str(v)

            f.write(
                "| {sys} | {m} | {f1} | {mj} | {ifn} | {ned} | {sm} | {icp} | {k} | {gtk} | {kd} | {mu} | {u} | {cap} |\n".format(
                    sys=r.get("system"),
                    m=r.get("method"),
                    f1=_f("bcubed_f1", nd=4),
                    mj=_f("mojosim", nd=2),
                    ifn=_f("ifn", nd=2),
                    ned=_f("ned", nd=4),
                    sm=_f("sm", nd=4),
                    icp=_f("icp", nd=4),
                    k=_f("pred_k"),
                    gtk=_f("gt_k"),
                    kd=_f("k_diff"),
                    mu=r.get("mu_override", "-"),
                    u=r.get("u_ablation", ""),
                    cap=_f("cap", nd=2),
                )
            )

        # Reviewer-facing summary section
        f.write("\n## All-Systems Summary (mean \u00b1 std)\n\n")
        f.write(
            "| Method | BCubedF1 | IFN | ICP | K | K-Diff |\n"
        )
        f.write("|---|---:|---:|---:|---:|---:|\n")

        def _fmt_pm(mu: float | None, sd: float | None, nd: int = 4) -> str:
            if mu is None:
                return "-"
            if sd is None:
                return f"{mu:.{nd}f}"
            return f"{mu:.{nd}f} \u00b1 {sd:.{nd}f}"

        def _fmt_pm_int(mu: float | None, sd: float | None) -> str:
            if mu is None:
                return "-"
            if sd is None:
                return str(int(round(mu)))
            return f"{int(round(mu))} \u00b1 {sd:.2f}"

        for m in methods:
            mm = summary_mean.get(m, {})
            ss = summary_std.get(m, {})
            f.write(
                "| {m} | {f1} | {ifn} | {icp} | {k} | {kd} |\n".format(
                    m=m,
                    f1=_fmt_pm(mm.get("bcubed_f1"), ss.get("bcubed_f1"), nd=4),
                    ifn=_fmt_pm(mm.get("ifn"), ss.get("ifn"), nd=2),
                    icp=_fmt_pm(mm.get("icp"), ss.get("icp"), nd=4),
                    k=_fmt_pm_int(mm.get("pred_k"), ss.get("pred_k")),
                    kd=_fmt_pm_int(mm.get("k_diff"), ss.get("k_diff")),
                )
            )

        f.write("\n**Note**: This summary supports reviewer-facing comparisons of overall advantage and stability (mean\u00b1std).\n")
        f.write("We recommend reporting: per-system results under K-lock + all-systems mean\u00b1std.\n")

    print(f"[OK] saved: {out_csv}")
    print(f"[OK] saved: {out_md}")


if __name__ == "__main__":
    main()
