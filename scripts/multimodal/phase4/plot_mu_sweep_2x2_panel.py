"""Plot mu_override sweeps for all four systems as a single 1×4 compact panel figure.

Inputs (auto-pick latest per system):
- results/ablation/<system>_mu_sweep_*.csv

Output:
- results/plots/mu_sweep_1x4_<ts>.png

Design goals (reviewer-facing):
- One consolidated figure (1×4), comparable axes.
- BCubed F1 on left y-axis (blue), MoJoSim on right y-axis (gray).
- Annotate best BCubed F1 point with μ and K/GT_K.

Usage:
  python scripts/multimodal/phase4/plot_mu_sweep_1x4_panel.py
  python scripts/multimodal/phase4/plot_mu_sweep_1x4_panel.py --systems acmeair,daytrader,plants,jpetstore
  python scripts/multimodal/phase4/plot_mu_sweep_1x4_panel.py --ts 20260220_191513

If you pass --ts, it will prefer files matching that timestamp; otherwise it
picks the latest CSV for each system.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt

# Optional: better label placement to reduce overlaps
try:
    from adjustText import adjust_text  # type: ignore
except Exception:  # pragma: no cover
    adjust_text = None


ROOT = Path(__file__).resolve().parents[3]
ABL = ROOT / "results" / "ablation"
OUT = ROOT / "results" / "plots"

DEFAULT_SYSTEMS = ["acmeair", "daytrader", "plants", "jpetstore"]

# Distinct, colorblind-friendly palette + markers for 4 systems
SYS_STYLES = {
    "acmeair": {"color": "#1f77b4", "marker": "o"},
    "daytrader": {"color": "#ff7f0e", "marker": "s"},
    "plants": {"color": "#2ca02c", "marker": "^"},
    "jpetstore": {"color": "#d62728", "marker": "D"},
}


def _load_rows(p: Path) -> List[Dict[str, str]]:
    with p.open("r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        return [dict(row) for row in r]


def _pick_csv(system: str, ts: Optional[str]) -> Path:
    if ts:
        cands = sorted(ABL.glob(f"{system}_mu_sweep_{ts}.csv"))
        if cands:
            return cands[-1]
    cands = sorted(ABL.glob(f"{system}_mu_sweep_*.csv"))
    if not cands:
        raise FileNotFoundError(f"No {system}_mu_sweep_*.csv found under {ABL}")
    return cands[-1]


def _to_series(rows: List[Dict[str, str]]) -> Tuple[List[float], List[float], List[float], List[int], int]:
    def f(x: str) -> float:
        try:
            return float(x)
        except Exception:
            return 0.0

    xs = [f(r.get("mu_override", "0")) for r in rows]
    f1 = [f(r.get("bcubed_f1", "0")) for r in rows]
    mj = [f(r.get("mojosim", "0")) for r in rows]
    k = [int(float(r.get("pred_k", 0) or 0)) for r in rows]
    gt_k = int(float(rows[0].get("gt_k", 0) or 0)) if rows else 0

    order = sorted(range(len(xs)), key=lambda i: xs[i])
    xs = [xs[i] for i in order]
    f1 = [f1[i] for i in order]
    mj = [mj[i] for i in order]
    k = [k[i] for i in order]
    return xs, f1, mj, k, gt_k


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--systems", type=str, default=",".join(DEFAULT_SYSTEMS))
    ap.add_argument("--ts", type=str, default=None, help="Preferred timestamp, e.g. 20260220_191513")
    ap.add_argument("--title", type=str, default="μ sensitivity (structural weight) — 1×4 compact panel")
    # NEW: all-in-one option
    ap.add_argument(
        "--combined",
        action="store_true",
        default=False,
        help="Plot all systems on one axes using BCubedF1 only (1 figure, 4 lines).",
    )
    ap.add_argument(
        "--skip_missing",
        action="store_true",
        default=True,
        help="Skip systems with missing <system>_mu_sweep_*.csv instead of failing.",
    )
    args = ap.parse_args()

    systems = [s.strip().lower() for s in args.systems.split(",") if s.strip()]

    # Load data for each system
    data = []
    for s in systems:
        try:
            csv_path = _pick_csv(s, args.ts)
        except FileNotFoundError:
            if args.skip_missing or args.combined:
                print(f"[WARN] skip missing mu_sweep CSV for system: {s}")
                continue
            raise
        rows = _load_rows(csv_path)
        if not rows:
            if args.skip_missing or args.combined:
                print(f"[WARN] skip empty CSV for system: {s}: {csv_path}")
                continue
            raise RuntimeError(f"Empty CSV: {csv_path}")
        xs, f1, mj, k, gt_k = _to_series(rows)
        data.append((s, csv_path, xs, f1, mj, k, gt_k))

    if args.combined and len(data) < 2:
        raise RuntimeError(
            "Need at least 2 systems with available mu_sweep CSVs for --combined. "
            "Tip: run scripts/multimodal/phase4/run_mu_sweep_all_systems.py first, or pass --systems daytrader only."
        )

    plt.style.use("seaborn-v0_8-whitegrid")

    if args.combined:
        # --- NEW: single-axes combined BCubedF1 plot ---
        # Narrower paper figure + black frame (spines)
        fig, ax = plt.subplots(figsize=(9.2, 5.6), dpi=220)

        # Paper-friendly font sizes
        plt.rcParams.update(
            {
                "axes.titlesize": 17,
                # Keep general label size, but bump axis-labels explicitly below for μ / BCubed F1
                "axes.labelsize": 15,
                "xtick.labelsize": 13,
                "ytick.labelsize": 13,
                "legend.fontsize": 13,
            }
        )

        # Black border/frame
        for side in ("top", "right", "bottom", "left"):
            ax.spines[side].set_color("black")
            ax.spines[side].set_linewidth(1.2)

        ax.tick_params(axis="both", colors="black")

        all_f1 = [v for (_, _, _, f1, _, _, _) in data for v in f1]
        f1_max = max(0.55, max(all_f1) * 1.15) if all_f1 else 1.0

        texts = []

        for (system, csv_path, xs, f1, _mj, k, gt_k) in data:
            st = SYS_STYLES.get(system, {"color": "#333333", "marker": "o"})
            ax.plot(
                xs,
                f1,
                marker=st["marker"],
                markersize=4.6,
                linewidth=2.0,
                color=st["color"],
                label=system,
            )

            # annotate best point per system (keep label away from marker)
            best_i = max(range(len(xs)), key=lambda i: f1[i]) if xs else 0
            if xs:
                ax.scatter(
                    [xs[best_i]],
                    [f1[best_i]],
                    s=75,
                    facecolors="white",
                    edgecolors=st["color"],
                    linewidths=2.0,
                    zorder=6,
                )

                # Put some labels below to prevent collisions
                label_offsets = {
                    "acmeair": (-8, -58),
                    "jpetstore": (0, -52),
                }
                dx, dy = label_offsets.get(system, (18, 22))

                # larger offset + arrow, so label won't be covered by points
                txt = ax.annotate(
                    f"{system}: μ={xs[best_i]:.2f}, F1={f1[best_i]:.3f}",
                    xy=(xs[best_i], f1[best_i]),
                    xytext=(dx, dy),
                    textcoords="offset points",
                    fontsize=12,
                    color=st["color"],
                    bbox=dict(boxstyle="round,pad=0.22", fc="white", ec="#bbb", alpha=0.95),
                    arrowprops=dict(arrowstyle="->", color=st["color"], lw=0.8, alpha=0.9),
                )
                texts.append(txt)

        # If available, automatically repel annotations from each other and from the data
        if adjust_text is not None and texts:
            adjust_text(
                texts,
                ax=ax,
                only_move={"points": "y", "text": "xy"},
                expand_points=(1.2, 1.35),
                expand_text=(1.15, 1.25),
                force_points=0.15,
                force_text=0.20,
                arrowprops=dict(arrowstyle="->", lw=0.8, color="#666", alpha=0.65),
            )

        ax.set_title(args.title if args.title else "μ sensitivity (BCubedF1 only)")
        ax.set_xlabel(r"$\mu$ (structural weight)", fontsize=17)
        ax.set_ylabel("BCubed F1", fontsize=17)
        ax.set_ylim(0, f1_max)
        ax.grid(True, alpha=0.25)
        ax.legend(ncol=2, frameon=True, loc="lower right")

        ts_out = datetime.now().strftime("%Y%m%d_%H%M%S")
        OUT.mkdir(parents=True, exist_ok=True)
        out_path = OUT / f"mu_sweep_bcubed_combined_{ts_out}.png"
        fig.tight_layout()
        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)

        print("[OK] combined BCubedF1 figure saved:", out_path)
        for system, csv_path, *_ in data:
            print(f"  - {system}: {csv_path}")
        return

    # --- existing 1x4 panel code path (unchanged) ---
    if len(systems) != 4:
        raise ValueError("This panel plot expects exactly 4 systems (for a 1×4 grid).")

    # Global y-limits for comparability
    all_f1 = [v for (_, _, _, f1, _, _, _) in data for v in f1]
    all_mj = [v for (_, _, _, _, mj, _, _) in data for v in mj]
    f1_max = max(0.55, max(all_f1) * 1.15)
    mj_max = max(60, max(all_mj) * 1.15)

    # tight horizontal strip
    fig, axes = plt.subplots(1, 4, figsize=(16.0, 3.6), dpi=220, sharex=True)

    col_blue = "#1f77b4"
    col_gray = "#999999"

    # store twin axes only for setting labels on the last panel
    twin_axes = []

    for idx, (system, csv_path, xs, f1, mj, k, gt_k) in enumerate(data):
        ax1 = axes[idx]
        ax2 = ax1.twinx()
        twin_axes.append(ax2)

        ax1.plot(xs, f1, marker="o", markersize=3.8, linewidth=1.8, color=col_blue, label="BCubed F1")
        ax2.plot(xs, mj, marker="s", markersize=3.6, linewidth=1.4, linestyle="--", color=col_gray, label="MoJoSim")

        ax1.set_title(system)
        ax1.set_ylim(0, f1_max)
        ax2.set_ylim(0, mj_max)

        best_i = max(range(len(xs)), key=lambda i: f1[i])
        ax1.scatter([xs[best_i]], [f1[best_i]], s=40, color=col_blue, zorder=5)
        ax1.annotate(
            f"F1={f1[best_i]:.3f}\nμ={xs[best_i]:.2f}\nK={k[best_i]} (GT={gt_k})",
            xy=(xs[best_i], f1[best_i]),
            xytext=(8, 10),
            textcoords="offset points",
            fontsize=7,
            bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="#888", alpha=0.92),
            arrowprops=dict(arrowstyle="->", color="#666", lw=0.7),
        )

        # x label only once to keep compact
        ax1.set_xlabel(r"$\mu$")

        # left y-axis label only on first panel
        if idx == 0:
            ax1.set_ylabel("BCubed F1", color=col_blue)
            ax1.tick_params(axis="y", labelcolor=col_blue)
        else:
            ax1.set_ylabel("")
            ax1.tick_params(axis="y", labelleft=False)

        # right y-axis label only on last panel
        if idx == 3:
            ax2.set_ylabel("MoJoSim", color=col_gray)
            ax2.tick_params(axis="y", labelcolor=col_gray)
        else:
            ax2.set_ylabel("")
            ax2.tick_params(axis="y", labelright=False)

    # One shared legend (compact)
    from matplotlib.lines import Line2D

    proxies = [
        Line2D([0], [0], color=col_blue, marker="o", linewidth=1.8, label="BCubed F1"),
        Line2D([0], [0], color=col_gray, marker="s", linewidth=1.4, linestyle="--", label="MoJoSim"),
    ]
    fig.legend(handles=proxies, loc="lower center", ncol=2, frameon=True, bbox_to_anchor=(0.5, -0.02))

    fig.suptitle(args.title, y=1.02)
    fig.tight_layout(rect=[0, 0.08, 1, 0.95])

    ts_out = datetime.now().strftime("%Y%m%d_%H%M%S")
    OUT.mkdir(parents=True, exist_ok=True)
    out_path = OUT / f"mu_sweep_1x4_{ts_out}.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)

    print("[OK] 1x4 compact panel saved:", out_path)
    for system, csv_path, *_ in data:
        print(f"  - {system}: {csv_path}")


if __name__ == "__main__":
    main()
