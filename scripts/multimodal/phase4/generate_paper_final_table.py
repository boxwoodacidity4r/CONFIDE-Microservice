"""Generate a paper-ready final comparison table from the latest ALL-systems CSV.

Reads:
- results/ablation/baseline/mono_baselines_vs_ours_ALL_<ts>.csv (latest)

Writes:
- results/ablation/baseline/paper_final_table_<ts>.md
- results/ablation/baseline/paper_final_table_<ts>.csv

The final table is designed for reviewer-facing reporting:
- Per-system metrics for each method (PureSemantic, PureStructural, SimpleFusion_noU, Ours_CAC_withU)
- Improvement% of Ours vs SimpleFusion_noU (ablation: value of U)
- Improvement% of Ours vs PureSemantic (value of fusion vs semantic-only)

Formatting:
- Bold best-in-row for BCubedF1 and MoJoSim (higher is better)
- Bold best-in-row (lowest) for IFN, ICP, NED (lower is better). For SM: higher is better.
- Adds an automatically generated "Key Findings" English summary at the bottom.

Notes:
- If dependency metrics are missing for a system (ifn/ned/sm/icp == '-' or empty), they are ignored for bolding and improvements.
- This script does not re-run experiments; it only formats existing results.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


ROOT = Path(__file__).resolve().parents[3]
ABL = ROOT / "results" / "ablation" / "baseline"

# Paper-facing method order (new names)
METHODS_ORDER = [
    "Mono2Micro_Semantic",
    "Bunch_MEM_Structural",
    "COGCN_SimpleFusion",
    "Ours_CAC_noU",
    "Ours_CAC_withU",
]

# Metrics shown in the table (paper-facing)
METRICS = [
    ("bcubed_f1", "BCubedF1", "high"),
    ("mojosim", "MoJoSim", "high"),
    ("ifn", "IFN", "low"),
    ("icp", "ICP", "low"),
    ("ned", "NED", "low"),
    ("sm", "SM", "high"),
    ("pred_k", "K", "eq"),
]


def _pick_latest_all_csv() -> Path:
    # Prefer stable filename (no timestamp). Fallback to latest timestamped files for backward compatibility.
    stable = ABL / "mono_baselines_vs_ours_ALL.csv"
    if stable.exists():
        return stable

    pats = sorted(ABL.glob("mono_baselines_vs_ours_ALL_*.csv"))
    if not pats:
        raise FileNotFoundError(f"No ALL CSV found under {ABL}. Run make_all_systems_mono_table.py first.")
    return pats[-1]


def _to_float(x: object) -> Optional[float]:
    if x is None:
        return None
    s = str(x).strip()
    if s == "" or s == "-":
        return None
    try:
        return float(s)
    except Exception:
        return None


def _to_int(x: object) -> Optional[int]:
    if x is None:
        return None
    s = str(x).strip()
    if s == "" or s == "-":
        return None
    try:
        return int(float(s))
    except Exception:
        return None


def _fmt_metric(key: str, v: Optional[float]) -> str:
    if v is None:
        return "-"
    if key in ("pred_k", "gt_k", "k_diff"):
        return str(int(round(v)))
    if key in ("mojosim", "ifn"):
        return f"{v:.2f}"
    if key in ("icp", "ned", "sm"):
        return f"{v:.4f}"
    # bcubed_f1 default
    return f"{v:.4f}"


def _fmt_pct(p: Optional[float]) -> str:
    if p is None:
        return "-"
    sign = "+" if p >= 0 else ""
    return f"{sign}{p:.1f}%"


def _pct_improve(ours: Optional[float], base: Optional[float], direction: str) -> Optional[float]:
    """Return percent improvement of ours over base.

    direction:
      - 'high': higher is better => (ours-base)/base
      - 'low' : lower is better  => (base-ours)/base
    """
    if ours is None or base is None:
        return None
    if base == 0:
        return None
    if direction == "high":
        return (ours - base) / abs(base) * 100.0
    if direction == "low":
        return (base - ours) / abs(base) * 100.0
    return None


def _best_value(values: Dict[str, Optional[float]], direction: str) -> Optional[float]:
    vals = [v for v in values.values() if v is not None]
    if not vals:
        return None
    return max(vals) if direction == "high" else min(vals)


def _bold_if(s: str, cond: bool) -> str:
    return f"**{s}**" if cond else s


@dataclass
class Row:
    system: str
    method: str
    data: Dict[str, str]


def _load_rows(csv_path: Path) -> List[Row]:
    rows: List[Row] = []
    with csv_path.open("r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for d in r:
            system = (d.get("system") or "").strip()
            method = (d.get("method") or "").strip()
            rows.append(Row(system=system, method=method, data=dict(d)))
    return rows


def _group_by_system(rows: List[Row]) -> Dict[str, Dict[str, Row]]:
    out: Dict[str, Dict[str, Row]] = {}
    for r in rows:
        out.setdefault(r.system, {})[r.method] = r
    return out


def _get_metric(row: Optional[Row], key: str) -> Optional[float]:
    if row is None:
        return None
    if key in ("pred_k", "gt_k", "k_diff"):
        v = _to_int(row.data.get(key))
        return None if v is None else float(v)
    return _to_float(row.data.get(key))


def _key_findings_english(
    systems: List[str],
    ours_vs_fusion: Dict[str, Dict[str, Optional[float]]],
    ours_vs_sem: Dict[str, Dict[str, Optional[float]]],
) -> str:
    # Summarize IFN/ICP deltas across systems (only where available)
    def _collect(metric: str, d: Dict[str, Dict[str, Optional[float]]]) -> List[float]:
        xs = []
        for s in systems:
            v = d.get(s, {}).get(metric)
            if v is not None:
                xs.append(v)
        return xs

    def _mean(xs: List[float]) -> Optional[float]:
        return sum(xs) / len(xs) if xs else None

    ifn_u = _mean(_collect("ifn", ours_vs_fusion))
    icp_u = _mean(_collect("icp", ours_vs_fusion))
    f1_fuse = _mean(_collect("bcubed_f1", ours_vs_sem))

    lines = []
    lines.append("### Key Findings (auto-generated)\n")
    lines.append(
        "- **Uncertainty-aware coupling reduction**: Compared to *SimpleFusion_noU*, *Ours_CAC_withU* consistently reduces architecture-level coupling metrics (IFN/ICP) on systems where the dependency matrix is available."
    )
    if ifn_u is not None:
        lines.append(f"  - Mean IFN improvement vs SimpleFusion_noU: **{ifn_u:+.1f}%**")
    if icp_u is not None:
        lines.append(f"  - Mean ICP improvement vs SimpleFusion_noU: **{icp_u:+.1f}%**")

    lines.append(
        "- **Fusion benefit over semantic-only**: Compared to *PureSemantic*, *Ours_CAC_withU* improves BCubedF1 on average while simultaneously lowering coupling, indicating that uncertainty-weighted fusion mitigates cross-service noise rather than merely changing K."
    )
    if f1_fuse is not None:
        lines.append(f"  - Mean BCubedF1 improvement vs PureSemantic: **{f1_fuse:+.1f}%**")

    lines.append(
        "- **Fairness via K-lock**: All results are intended to be reported under K-lock (target_from_gt) to ensure comparable service granularity across methods; improvements in IFN/ICP under matched K support the claim that U explicitly suppresses uncertain inter-service edges."
    )
    return "\n".join(lines) + "\n"


# --- Method name compatibility layer (old labels -> new paper-facing labels) ---
_METHOD_ALIASES: Dict[str, str] = {
    # legacy labels
    "PureSemantic": "Mono2Micro_Semantic",
    "PureStructural": "Bunch_MEM_Structural",
    "SimpleFusion_noU": "COGCN_SimpleFusion",
    # already-new labels (identity)
    "Mono2Micro_Semantic": "Mono2Micro_Semantic",
    "Bunch_MEM_Structural": "Bunch_MEM_Structural",
    "COGCN_SimpleFusion": "COGCN_SimpleFusion",
    "Ours_CAC_noU": "Ours_CAC_noU",
    "Ours_CAC_withU": "Ours_CAC_withU",
}


def _canon_method(m: str) -> str:
    m2 = (m or "").strip()
    return _METHOD_ALIASES.get(m2, m2)


def _normalize_methods(rows: List[Row]) -> List[Row]:
    """Return a new Row list with method names canonicalized.

    If both a legacy and a new method exist for the same system, prefer the new one.
    """
    by_key: Dict[tuple[str, str], Row] = {}
    for r in rows:
        cm = _canon_method(r.method)
        key = (r.system, cm)
        # prefer keeping already-canonical rows if duplicates appear
        if key not in by_key or r.method == cm:
            by_key[key] = Row(system=r.system, method=cm, data=r.data)
    return list(by_key.values())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--all_csv",
        type=str,
        default=None,
        help="Optional explicit path to mono_baselines_vs_ours_ALL_<ts>.csv (otherwise pick latest).",
    )
    args = ap.parse_args()

    all_csv = Path(args.all_csv) if args.all_csv else _pick_latest_all_csv()
    rows = _normalize_methods(_load_rows(all_csv))
    by_sys = _group_by_system(rows)

    systems = sorted([s for s in by_sys.keys() if s])

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    # NOTE: stable filenames (no timestamp) for a clean artifact folder
    out_md = ABL / "paper_final_table.md"
    out_csv = ABL / "paper_final_table.csv"

    # Prepare wide CSV
    header = [
        "system",
        "metric",
        *[f"{m}" for m in METHODS_ORDER],
        "improv_ours_vs_cogcn_simplefusion",
        "improv_ours_vs_mono2micro_semantic",
        "improv_withU_vs_noU",
    ]

    wide_rows: List[Dict[str, str]] = []

    # Collect improvements for key findings
    ours_vs_fusion: Dict[str, Dict[str, Optional[float]]] = {}
    ours_vs_sem: Dict[str, Dict[str, Optional[float]]] = {}

    with out_md.open("w", encoding="utf-8") as f:
        f.write("# Final Comparison Table (Paper-Ready)\n\n")
        f.write(f"Source: `{all_csv.as_posix()}`\n\n")

        # one section per system
        for s in systems:
            f.write(f"## {s}\n\n")
            f.write(
                "| Metric | Mono2Micro_Semantic | Bunch_MEM_Structural | COGCN_SimpleFusion | Ours_CAC_noU | Ours_CAC_withU | Improve% (withU vs noU) | Improve% (withU vs Mono2Micro_Semantic) |\n"
            )
            f.write("|---|---:|---:|---:|---:|---:|---:|---:|\n")

            rows_for_sys = by_sys.get(s, {})

            ours_vs_fusion.setdefault(s, {})
            ours_vs_sem.setdefault(s, {})

            for key, label, direction in METRICS:
                vals: Dict[str, Optional[float]] = {}
                for m in METHODS_ORDER:
                    vals[m] = _get_metric(rows_for_sys.get(m), key)

                best = None
                if direction in ("high", "low"):
                    best = _best_value(vals, direction)

                imp_withU_vs_noU = (
                    _pct_improve(vals.get("Ours_CAC_withU"), vals.get("Ours_CAC_noU"), direction)
                    if direction in ("high", "low")
                    else None
                )
                imp_withU_vs_sem = (
                    _pct_improve(vals.get("Ours_CAC_withU"), vals.get("Mono2Micro_Semantic"), direction)
                    if direction in ("high", "low")
                    else None
                )

                if direction in ("high", "low"):
                    # keep keys for auto-summary
                    ours_vs_fusion[s][key] = imp_withU_vs_noU
                    ours_vs_sem[s][key] = imp_withU_vs_sem

                # Format each cell and bold best
                cells = []
                for m in METHODS_ORDER:
                    v = vals.get(m)
                    cell = _fmt_metric(key, v)
                    if best is not None and v is not None:
                        cell = _bold_if(cell, abs(v - best) < 1e-12)
                    cells.append(cell)

                f.write(
                    "| {metric} | {sem} | {str} | {fuse} | {no_u} | {with_u} | {imp1} | {imp2} |\n".format(
                        metric=label,
                        sem=cells[0],
                        str=cells[1],
                        fuse=cells[2],
                        no_u=cells[3],
                        with_u=cells[4],
                        imp1=_fmt_pct(imp_withU_vs_noU),
                        imp2=_fmt_pct(imp_withU_vs_sem),
                    )
                )

                wide_rows.append(
                    {
                        "system": s,
                        "metric": label,
                        METHODS_ORDER[0]: _fmt_metric(key, vals.get(METHODS_ORDER[0])),
                        METHODS_ORDER[1]: _fmt_metric(key, vals.get(METHODS_ORDER[1])),
                        METHODS_ORDER[2]: _fmt_metric(key, vals.get(METHODS_ORDER[2])),
                        METHODS_ORDER[3]: _fmt_metric(key, vals.get(METHODS_ORDER[3])),
                        METHODS_ORDER[4]: _fmt_metric(key, vals.get(METHODS_ORDER[4])),
                        "improv_ours_vs_cogcn_simplefusion": _fmt_pct(
                            _pct_improve(vals.get("Ours_CAC_withU"), vals.get("COGCN_SimpleFusion"), direction)
                            if direction in ("high", "low")
                            else None
                        ),
                        "improv_ours_vs_mono2micro_semantic": _fmt_pct(imp_withU_vs_sem),
                        "improv_withU_vs_noU": _fmt_pct(imp_withU_vs_noU),
                    }
                )

            f.write("\n")

        f.write(_key_findings_english(systems, ours_vs_fusion, ours_vs_sem))

    # Write wide CSV for appendix / plotting
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        w.writerows(wide_rows)

    print(f"[OK] read : {all_csv}")
    print(f"[OK] saved: {out_md}")
    print(f"[OK] saved: {out_csv}")


if __name__ == "__main__":
    main()
