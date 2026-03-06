"""Plot mu_override sweeps for any system (paper-ready).

Input (auto-picks latest if not provided):
- results/ablation/<system>_mu_sweep_*.csv

Output:
- results/plots/<system>_mu_sweep_<ts>.png

This mirrors the DayTrader-only plotting script but works for all systems.

Usage:
  python scripts/multimodal/phase4/plot_mu_sweep_all_systems.py --system acmeair
  python scripts/multimodal/phase4/plot_mu_sweep_all_systems.py --system plants --csv_path results/ablation/plants_mu_sweep_....csv
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[3]
ABL = ROOT / "results" / "ablation"
OUT = ROOT / "results" / "plots"


def _pick_latest(system: str) -> Path:
    cands = sorted(ABL.glob(f"{system}_mu_sweep_*.csv"))
    if not cands:
        raise FileNotFoundError(
            f"No {system}_mu_sweep_*.csv found. Run run_mu_sweep_all_systems.py first (or provide --csv_path)."
        )
    return cands[-1]


def _load_rows(p: Path) -> List[Dict[str, str]]:
    with p.open("r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        return [dict(row) for row in r]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", type=str, required=True)
    ap.add_argument("--csv_path", type=str, default=None)
    ap.add_argument("--title", type=str, default=None)
    args = ap.parse_args()

    system = args.system.strip().lower()
    csv_path = Path(args.csv_path) if args.csv_path else _pick_latest(system)
    rows = _load_rows(csv_path)
    if not rows:
        raise RuntimeError(f"Empty CSV: {csv_path}")

    def f(x: str) -> float:
        try:
            return float(x)
        except Exception:
            return 0.0

    xs = [f(r.get("mu_override", "0")) for r in rows]
    f1 = [f(r.get("bcubed_f1", "0")) for r in rows]
    mj = [f(r.get("mojosim", "0")) for r in rows]
    k = [int(float(r.get("pred_k", 0) or 0)) for r in rows]
    gt_k = int(float(rows[0].get("gt_k", 0) or 0))

    # ensure sorted by mu (CSV should already be sorted, but enforce)
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    xs = [xs[i] for i in order]
    f1 = [f1[i] for i in order]
    mj = [mj[i] for i in order]
    k = [k[i] for i in order]

    best_i = max(range(len(xs)), key=lambda i: f1[i])

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax1 = plt.subplots(figsize=(8.8, 4.6), dpi=220)

    col_blue = "#1f77b4"
    col_gray = "#999999"

    ax1.plot(xs, f1, marker="o", linewidth=2.2, color=col_blue, label="BCubed F1")
    ax1.set_xlabel(r"$\mu$ (structural weight)")
    ax1.set_ylabel("BCubed F1", color=col_blue)
    ax1.tick_params(axis="y", labelcolor=col_blue)
    ax1.set_ylim(0, max(0.55, max(f1) * 1.15))

    ax2 = ax1.twinx()
    ax2.plot(xs, mj, marker="s", linewidth=2.0, linestyle="--", color=col_gray, label="MoJoSim")
    ax2.set_ylabel("MoJoSim", color=col_gray)
    ax2.tick_params(axis="y", labelcolor=col_gray)
    ax2.set_ylim(0, max(60, max(mj) * 1.15))

    ax1.scatter([xs[best_i]], [f1[best_i]], s=90, color=col_blue, zorder=5)
    ax1.annotate(
        f"best F1={f1[best_i]:.3f} at $\\mu$={xs[best_i]:.2f}\\nK={k[best_i]} (GT_K={gt_k})",
        xy=(xs[best_i], f1[best_i]),
        xytext=(10, 18),
        textcoords="offset points",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#888", alpha=0.95),
        arrowprops=dict(arrowstyle="->", color="#666"),
    )

    title = args.title or f"{system} μ sensitivity (structural weight)"
    ax1.set_title(title)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="lower right", frameon=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    OUT.mkdir(parents=True, exist_ok=True)
    out_path = OUT / f"{system}_mu_sweep_{ts}.png"
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)

    print(f"[OK] input: {csv_path}")
    print(f"[OK] saved: {out_path}")


if __name__ == "__main__":
    main()
