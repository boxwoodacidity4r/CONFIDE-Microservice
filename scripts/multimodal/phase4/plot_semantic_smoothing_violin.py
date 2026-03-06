"""Plot semantic smoothing as violin plots (distribution + median/IQR).

Goal
----
Make the "semantic smoothing" phenomenon visually intuitive: most off-diagonal
pairwise cosine similarities crowd near 0.9~1.0.

This script mirrors `plot_semantic_smoothing_bar_median_iqr.py`'s inputs and
(optional) GT=-1 filtering, but renders a violin plot per system.

Outputs (paper_mode):
- results/paper/semantic_smoothing_violin_<matrix>.png
- results/paper/semantic_smoothing_violin_<matrix>.manifest.json

Example (PowerShell):
  python scripts/multimodal/phase4/plot_semantic_smoothing_violin.py --paper_mode --matrix raw

"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")  # headless
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
        return {"n": 0, "median": float("nan"), "p25": float("nan"), "p75": float("nan")}
    return {
        "n": int(vals.size),
        "median": float(np.median(vals)),
        "p25": float(np.percentile(vals, 25)),
        "p75": float(np.percentile(vals, 75)),
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
    ap.add_argument("--font_family", type=str, default="Times New Roman")
    ap.add_argument("--label_size", type=int, default=18)
    ap.add_argument("--annot_size", type=int, default=16)
    ap.add_argument("--title_size", type=int, default=18)
    ap.add_argument("--subtitle_size", type=int, default=14)
    ap.add_argument(
        "--palette",
        type=str,
        default="cool_greys",
        choices=["cool_greys", "paper_pastel_pinkblue", "muted"],
    )
    ap.add_argument("--fig_w", type=float, default=8.0)
    ap.add_argument("--fig_h", type=float, default=5.2)
    ap.add_argument("--baseline", type=float, default=0.5)
    ap.add_argument("--max_points", type=int, default=120000, help="Cap samples per system for plotting speed.")
    ap.add_argument("--seed", type=int, default=42)

    # NEW: y-axis lower bound (paper default: start at baseline=0.5)
    ap.add_argument("--ymin", type=float, default=0.5, help="Y-axis lower bound.")

    args = ap.parse_args()

    systems = [s.strip().lower() for s in args.systems.split(",") if s.strip()]
    input_root = Path(args.input_root).resolve()
    data_root = input_root / "data" / "processed"

    out_dir = Path(args.out_dir).resolve() if args.out_dir else (ROOT / "results" / "plots")
    if args.paper_mode:
        out_dir = Path(args.out_dir).resolve() if args.out_dir else (ROOT / "results" / "paper")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"[START] semantic_smoothing_violin (matrix={args.matrix}, systems={systems})\n"
        f"        input_root={str(input_root)}\n"
        f"        out_dir={str(out_dir)}\n"
        f"        gt_filter={not bool(args.no_gt_filter)}",
        flush=True,
    )

    rng = np.random.default_rng(int(args.seed))

    # --- style ---
    if args.palette == "cool_greys":
        face_colors = ["#C7CCD1", "#BFC7CF", "#B7C2CD", "#B0BDCA"]
        edge_color = "#47505A"
        median_color = "#111111"
    elif args.palette == "paper_pastel_pinkblue":
        face_colors = ["#f59790", "#f5dbb6", "#cde2e8", "#c8d4e9"]
        edge_color = "#2F2F2F"
        median_color = "#111111"
    else:
        face_colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B2"]
        edge_color = "#2F2F2F"
        median_color = "#111111"

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

    datasets: List[np.ndarray] = []
    stats_by_system: Dict[str, Dict[str, float]] = {}
    manifest_files = []

    for i, system in enumerate(systems):
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
        if vals.size > int(args.max_points):
            idx = rng.choice(vals.size, size=int(args.max_points), replace=False)
            vals = vals[idx]

        datasets.append(vals)
        stats_by_system[system] = _summary(vals)
        print(
            f"[STAT] {system}: n={stats_by_system[system]['n']} "
            f"median={stats_by_system[system]['median']:.4f} "
            f"p25={stats_by_system[system]['p25']:.4f} "
            f"p75={stats_by_system[system]['p75']:.4f}",
            flush=True,
        )

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

    fig, ax = plt.subplots(figsize=(float(args.fig_w), float(args.fig_h)), dpi=300)

    positions = np.arange(1, len(systems) + 1)
    parts = ax.violinplot(
        datasets,
        positions=positions,
        widths=0.78,
        showmeans=False,
        showmedians=False,  # draw our own median (more control)
        showextrema=False,
    )

    for i, body in enumerate(parts["bodies"]):
        body.set_facecolor(face_colors[i % len(face_colors)])
        body.set_edgecolor(edge_color)
        body.set_alpha(0.85)
        body.set_linewidth(0.9)

    # overlay IQR box
    for pos, system, vals in zip(positions, systems, datasets):
        st = stats_by_system[system]
        med, q1, q3 = float(st["median"]), float(st["p25"]), float(st["p75"])

        # IQR box
        ax.add_patch(
            plt.Rectangle(
                (pos - 0.08, q1),
                0.16,
                max(0.0, q3 - q1),
                facecolor="white",
                edgecolor=edge_color,
                linewidth=1.0,
                alpha=0.85,
                zorder=6,
            )
        )
        # median line
        ax.plot([pos - 0.10, pos + 0.10], [med, med], color=median_color, linewidth=2.2, zorder=7)

        # median text: place slightly BELOW the median line to avoid overlapping the subtitle/top area
        y_text = max(float(args.ymin) + 0.015, med - 0.028)
        ax.text(
            pos,
            y_text,
            f"{med:.3f}",
            ha="center",
            va="top",
            fontsize=int(args.annot_size),
            fontweight="bold",
            color="#111111",
            zorder=8,
        )

    # baseline line
    ax.axhline(float(args.baseline), color="#9AA3AB", linestyle="--", linewidth=1.2, alpha=0.7, zorder=1)
    ax.text(
        0.99,
        float(args.baseline),
        "Ideal Baseline (0.5)",
        transform=ax.get_yaxis_transform(),
        ha="right",
        va="bottom",
        fontsize=int(args.subtitle_size),
        color="#7D858D",
    )

    ax.set_xticks(positions)
    ax.set_xticklabels(systems)

    # Y range: start from ymin (default 0.5) to remove empty bottom
    ax.set_ylim(float(args.ymin), 1.0)

    ax.set_ylabel("Cosine similarity")

    title = args.title or "Semantic smoothing"
    ax.set_title(title, loc="left", fontweight="bold", pad=18)
    ax.text(
        0.0,
        1.01,
        "Violin width shows probability density of off-diagonal pairs; box shows IQR (25th–75th)",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=int(args.subtitle_size),
        color="#6F767D",
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, axis="y", which="major", alpha=0.16, linewidth=0.6)
    ax.set_yticks([float(args.ymin), 0.75, 1.0] if float(args.ymin) >= 0.49 else [0.0, 0.5, 1.0])

    fig.tight_layout()
    fig.subplots_adjust(top=0.84)

    out_path = out_dir / f"semantic_smoothing_violin_{args.matrix}.png"
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
            "max_points": int(args.max_points),
            "summaries": stats_by_system,
            "files": manifest_files,
        }
        mpath = out_dir / f"semantic_smoothing_violin_{args.matrix}.manifest.json"
        mpath.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print("[OK] manifest:", mpath, flush=True)


if __name__ == "__main__":
    main()
