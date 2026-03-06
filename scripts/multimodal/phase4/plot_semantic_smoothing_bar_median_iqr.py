"""Plot semantic smoothing as a compact bar chart: median + IQR.

This figure is designed for the *Introduction* (self-explanatory and reviewer-friendly).
It summarizes the off-diagonal cosine similarity distribution of the semantic
similarity matrix (raw or DADE) using:
- bar height: median
- error bar: IQR (p25..p75)

It optionally filters out GT=-1 classes (default: enabled) to avoid infrastructure
or utility classes dominating the statistics.

Inputs (per system) under --input_root:
- data/processed/fusion/{system}_S_sem.npy                 (raw)
- data/processed/fusion/{system}_S_sem_dade_base.npy       (dade)
- data/processed/groundtruth/{system}_ground_truth.json    (for filtering)
- data/processed/fusion/{system}_class_order.json          (for alignment)

Outputs (paper_mode):
- results/paper/semantic_smoothing_bar_<matrix>.png
- results/paper/semantic_smoothing_bar_<matrix>.manifest.json

Example (PowerShell):
  python scripts/multimodal/phase4/plot_semantic_smoothing_bar_median_iqr.py `
    --paper_mode `
    --input_root results/paper_snapshot/paper_v1 `
    --matrix raw

"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for headless/CI/PowerShell
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[3]


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_gt_labels(gt_path: Path, class_order_path: Path) -> List[int]:
    gt: Dict[str, int] = json.loads(gt_path.read_text(encoding="utf-8"))
    order: List[str] = json.loads(class_order_path.read_text(encoding="utf-8"))
    labels: List[int] = []
    for name in order:
        key = str(name).replace(".java", "").strip()
        labels.append(int(gt.get(key, -1)))
    return labels


def _offdiag_values(S: np.ndarray, labels: List[int] | None = None) -> np.ndarray:
    n = int(S.shape[0])
    vals: List[float] = []
    for i in range(n):
        for j in range(i + 1, n):
            if labels is not None:
                if i >= len(labels) or j >= len(labels):
                    continue
                if int(labels[i]) == -1 or int(labels[j]) == -1:
                    continue
            vals.append(float(S[i, j]))
    return np.asarray(vals, dtype=float)


def _summary(vals: np.ndarray) -> Dict[str, float]:
    if vals.size == 0:
        return {
            "n": 0,
            "median": float("nan"),
            "p25": float("nan"),
            "p75": float("nan"),
            "mean": float("nan"),
        }
    p25 = float(np.percentile(vals, 25))
    med = float(np.median(vals))
    p75 = float(np.percentile(vals, 75))
    return {
        "n": int(vals.size),
        "median": med,
        "p25": p25,
        "p75": p75,
        "mean": float(np.mean(vals)),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--systems", type=str, default="acmeair,daytrader,plants,jpetstore")
    ap.add_argument(
        "--input_root",
        type=str,
        default=str(ROOT),
        help="Root folder that contains data/processed/... (project root or a snapshot root).",
    )
    ap.add_argument("--paper_mode", action="store_true", default=False)
    ap.add_argument("--out_dir", type=str, default=None)
    ap.add_argument("--matrix", type=str, default="raw", choices=["raw", "dade"])
    ap.add_argument(
        "--no_gt_filter",
        action="store_true",
        default=False,
        help="If set, do not filter out GT=-1 classes.",
    )
    ap.add_argument("--title", type=str, default=None)

    # Styling knobs (paper defaults)
    ap.add_argument(
        "--font_family",
        type=str,
        default="Times New Roman",
        help="Font family (override to e.g., 'Inter'/'Helvetica' if available).",
    )
    ap.add_argument("--label_size", type=int, default=18, help="Axis/tick label font size.")
    ap.add_argument("--annot_size", type=int, default=17, help="Median annotation font size (bold).")
    ap.add_argument("--title_size", type=int, default=18, help="Title font size.")
    ap.add_argument("--subtitle_size", type=int, default=14, help="Subtitle font size.")
    ap.add_argument("--bar_width", type=float, default=0.46, help="Bar width.")
    ap.add_argument(
        "--palette",
        type=str,
        default="paper_pastel_pinkblue",
        choices=["paper_pastel_pinkblue", "morandi_bluegray", "grey_with_highlight", "muted"],
        help="Color palette preset.",
    )
    ap.add_argument(
        "--highlight",
        type=str,
        default=None,
        help="Optional system name to highlight (e.g., 'jpetstore'). Only used for grey_with_highlight palette.",
    )

    ap.add_argument("--fig_w", type=float, default=8.0)
    ap.add_argument("--fig_h", type=float, default=5.0)

    args = ap.parse_args()

    systems = [s.strip().lower() for s in args.systems.split(",") if s.strip()]
    input_root = Path(args.input_root).resolve()
    data_root = input_root / "data" / "processed"

    out_dir = Path(args.out_dir).resolve() if args.out_dir else (ROOT / "results" / "plots")
    if args.paper_mode:
        out_dir = Path(args.out_dir).resolve() if args.out_dir else (ROOT / "results" / "paper")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"[START] semantic_smoothing_bar (matrix={args.matrix}, systems={systems})\n"
        f"        input_root={str(input_root)}\n"
        f"        out_dir={str(out_dir)}\n"
        f"        gt_filter={not bool(args.no_gt_filter)}",
        flush=True,
    )

    # --- style ---
    # Palette presets:
    # - morandi_bluegray: coordinated low-saturation blue/gray tones
    # - grey_with_highlight: all light grey except one highlighted system
    # - muted: seaborn muted
    if args.palette == "paper_pastel_pinkblue":
        # User-provided pastel palette (low saturation):
        # f59790  f5dbb6  cde2e8  c8d4e9
        bar_colors = ["#f59790", "#f5dbb6", "#cde2e8", "#c8d4e9"]
    elif args.palette == "morandi_bluegray":
        bar_colors = ["#2F4858", "#466B7A", "#6E8F9B", "#A9BBC3"]
    elif args.palette == "grey_with_highlight":
        base = "#C7CCD1"
        hi = "#2F6F9F"
        bar_colors = [base for _ in systems]
        if args.highlight:
            try:
                idx = systems.index(str(args.highlight).strip().lower())
                bar_colors[idx] = hi
            except ValueError:
                pass
    else:
        bar_colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B2"]
        try:
            import seaborn as sns  # optional

            sns.set_theme(style="white", font="sans-serif")
            sns.set_palette("muted")
            bar_colors = sns.color_palette("muted", n_colors=max(4, len(systems)))
        except Exception:
            pass

    plt.rcParams.update(
        {
            "font.family": "sans-serif" if str(args.font_family).lower() not in ["times new roman", "times"] else "serif",
            "font.serif": [str(args.font_family), "Times", "DejaVu Serif"],
            "font.sans-serif": [str(args.font_family), "Inter", "Helvetica", "Arial", "DejaVu Sans"],
            "axes.labelsize": int(args.label_size),
            "axes.titlesize": int(args.title_size),
            "xtick.labelsize": int(args.label_size),
            "ytick.labelsize": int(args.label_size),
        }
    )

    stats_by_system: Dict[str, Dict[str, float]] = {}
    manifest_files = []

    for system in systems:
        print(f"[LOAD] {system}...", flush=True)
        sem_raw = data_root / "fusion" / f"{system}_S_sem.npy"
        sem_dade = data_root / "fusion" / f"{system}_S_sem_dade_base.npy"
        mat_path = sem_raw if args.matrix == "raw" else sem_dade

        if not mat_path.exists():
            raise FileNotFoundError(f"Missing semantic matrix: {mat_path}")

        labels = None
        gt_path = data_root / "groundtruth" / f"{system}_ground_truth.json"
        order_path = data_root / "fusion" / f"{system}_class_order.json"
        if not args.no_gt_filter:
            if not gt_path.exists() or not order_path.exists():
                raise FileNotFoundError(f"Missing GT/order for filtering: {gt_path} / {order_path}")
            labels = _load_gt_labels(gt_path, order_path)

        S = np.load(str(mat_path))
        vals = _offdiag_values(S, labels=labels)
        stats_by_system[system] = _summary(vals)
        print(
            f"[STAT] {system}: n={stats_by_system[system]['n']} "
            f"median={stats_by_system[system]['median']:.4f} "
            f"p25={stats_by_system[system]['p25']:.4f} "
            f"p75={stats_by_system[system]['p75']:.4f}",
            flush=True,
        )

        # manifest input tracking
        for p in [mat_path] + ([] if args.no_gt_filter else [gt_path, order_path]):
            st = p.stat()
            manifest_files.append(
                {
                    "system": system,
                    "rel": str(p.relative_to(input_root)).replace("\\", "/"),
                    "size": int(st.st_size),
                    "mtime": int(st.st_mtime),
                    "sha256": _sha256(p),
                }
            )

    print("[PLOT] rendering figure...", flush=True)

    # --- plot ---
    fig, ax = plt.subplots(figsize=(float(args.fig_w), float(args.fig_h)), dpi=300)

    x = np.arange(len(systems))
    medians = np.array([stats_by_system[s]["median"] for s in systems], dtype=float)
    p25 = np.array([stats_by_system[s]["p25"] for s in systems], dtype=float)
    p75 = np.array([stats_by_system[s]["p75"] for s in systems], dtype=float)

    # yerr: distance to p25/p75
    yerr = np.vstack([
        np.maximum(0.0, medians - p25),
        np.maximum(0.0, p75 - medians),
    ])

    # Baseline line
    ax.axhline(
        0.5,
        color="#B5BBC1",
        linestyle="--",
        linewidth=1.2,
        alpha=0.65,
        zorder=0,
    )
    # Put baseline label directly on the line (avoid legend)
    ax.text(
        0.99,
        0.5,
        "Ideal Baseline",
        transform=ax.get_yaxis_transform(),
        ha="right",
        va="bottom",
        fontsize=int(args.subtitle_size),
        color="#8A9096",
    )

    bar_facecolors = [bar_colors[i % len(bar_colors)] for i in range(len(systems))]

    bars = ax.bar(
        x,
        medians,
        width=float(args.bar_width),
        color=bar_facecolors,
        alpha=0.95,
        edgecolor=bar_facecolors,  # match edge to fill
        linewidth=0.9,
        zorder=2,
    )

    ax.errorbar(
        x,
        medians,
        yerr=yerr,
        fmt="none",
        ecolor="#2F2F2F",
        elinewidth=1.3,
        capsize=4,
        capthick=1.3,
        alpha=0.65,
        zorder=5,
    )

    ax.set_xticks(x)
    ax.set_xticklabels([s for s in systems])
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Cosine similarity")

    # Title + subtitle (left aligned)
    title = args.title
    if not title:
        title = "Semantic smoothing"
    ax.set_title(title, loc="left", fontweight="bold", pad=22)
    ax.text(
        0.0,
        1.01,
        "Bars: median; error bars: IQR (25th–75th), off-diagonal pairs",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=int(args.subtitle_size),
        color="#6F767D",
    )

    # annotate bars with median value (slightly inside bar top)
    for rect, v in zip(bars, medians):
        y = float(rect.get_height())
        ax.text(
            rect.get_x() + rect.get_width() / 2,
            max(0.01, y - 0.03),
            f"{float(v):.3f}",
            ha="center",
            va="top",
            fontsize=int(args.annot_size),
            fontweight="bold",
            color="#111111",
        )

    # Clean spines: keep left & bottom only
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Reduce grid visual noise: only major y ticks
    ax.grid(False)
    ax.set_yticks([0.0, 0.5, 1.0])
    ax.grid(True, axis="y", which="major", alpha=0.18, linewidth=0.6)

    # Remove legend (baseline is labeled on the line)
    if ax.get_legend() is not None:
        ax.get_legend().remove()

    fig.tight_layout()
    # More headroom for enlarged title/subtitle.
    fig.subplots_adjust(top=0.82)

    out_path = out_dir / f"semantic_smoothing_bar_{args.matrix}.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)

    print("[OK] saved:", out_path, flush=True)

    if args.paper_mode:
        manifest = {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "input_root": str(input_root).replace("\\", "/"),
            "systems": systems,
            "matrix": str(args.matrix),
            "gt_filter": (not bool(args.no_gt_filter)),
            "summaries": stats_by_system,
            "files": manifest_files,
        }
        mpath = out_dir / f"semantic_smoothing_bar_{args.matrix}.manifest.json"
        mpath.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print("[OK] manifest:", mpath, flush=True)


if __name__ == "__main__":
    main()
