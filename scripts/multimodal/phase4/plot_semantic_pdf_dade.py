"""Plot semantic similarity PDF before/after DADE.

This figure supports the paper claim that DADE performs a 'contrast stretching'
on semantic similarities by reducing global similarity collapse and increasing
intra-vs-inter separability.

Inputs (for each system):
- raw semantic matrix (pre-DADE)
- DADE semantic matrix (post-DADE)
- ground truth mapping (for intra/inter split)
- class order (to align GT with matrix indices)

By default, it reads these repo-standard paths:
- raw:  data/processed/fusion/{system}_S_sem.npy
- dade: data/processed/fusion/{system}_S_sem_dade_base.npy
- gt:   data/processed/groundtruth/{system}_ground_truth.json
- order:data/processed/fusion/{system}_class_order.json

Outputs:
- results/plots/semantic_pdf_dade_all.png (stable filename)
- results/plots/semantic_pdf_dade_<system>.png (stable filenames)

Notes:
- No timestamps in filenames (artifact-ready).
- Density computed via numpy histogram (avoids extra deps).
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt


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
    labels = []
    for name in order:
        # tolerate missing keys by assigning -1 (will be treated as inter)
        key = str(name).replace(".java", "").strip()
        labels.append(int(gt.get(key, -1)))
    return labels


def _pair_values(S: np.ndarray, labels: List[int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (all, intra, inter) similarity values from upper triangle."""
    n = int(S.shape[0])
    vals_all = []
    vals_intra = []
    vals_inter = []
    for i in range(n):
        for j in range(i + 1, n):
            if labels[i] == -1 or labels[j] == -1:
                continue
            v = float(S[i, j])
            vals_all.append(v)
            if labels[i] == labels[j]:
                vals_intra.append(v)
            else:
                vals_inter.append(v)
    return np.asarray(vals_all, dtype=float), np.asarray(vals_intra, dtype=float), np.asarray(vals_inter, dtype=float)


def _density(vals: np.ndarray, bins: int = 60, range_: Tuple[float, float] = (0.0, 1.0)) -> Tuple[np.ndarray, np.ndarray]:
    hist, edges = np.histogram(vals, bins=bins, range=range_, density=True)
    x = 0.5 * (edges[:-1] + edges[1:])
    return x, hist


def _plot_one(system: str, raw: np.ndarray, dade: np.ndarray, labels: List[int], out_path: Path, *, bins: int = 60, only_all: bool = False):
    raw_all, raw_intra, raw_inter = _pair_values(raw, labels)
    d_all, d_intra, d_inter = _pair_values(dade, labels)

    fig, ax = plt.subplots(figsize=(7.2, 4.2))

    if only_all:
        series = [
            ("Raw (all)", raw_all, "#999999", "-"),
            ("Semantic Refiner (all)", d_all, "#1f77b4", "-"),
        ]
    else:
        series = [
            ("Raw (all)", raw_all, "#999999", "-"),
            ("Semantic Refiner (all)", d_all, "#1f77b4", "-"),
            ("Raw (intra)", raw_intra, "#999999", "--"),
            ("Semantic Refiner (intra)", d_intra, "#1f77b4", "--"),
            ("Raw (inter)", raw_inter, "#999999", ":"),
            ("Semantic Refiner (inter)", d_inter, "#1f77b4", ":"),
        ]

    for name, vals, color, ls in series:
        if vals.size == 0:
            continue
        x, y = _density(vals, bins=bins)
        ax.plot(x, y, label=name, color=color, linestyle=ls, linewidth=2)

    ax.set_title(f"Semantic similarity PDF (pre/post Semantic Refiner) - {system}")
    ax.set_xlabel("Cosine similarity")
    ax.set_ylabel("Probability density")
    ax.set_xlim(0, 1)
    ax.grid(True, alpha=0.25)
    ax.legend(ncol=2, fontsize=9)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


# --- NEW: combined figure helper (semantic_pdf_dade_all.png) ---
def _plot_all_2x2_panel(
    systems: List[str],
    per_system_pngs: Dict[str, Path],
    out_path: Path,
    *,
    suptitle: str = "Semantic similarity PDF (pre/post DADE) — all systems",
) -> None:
    """Combine the 4 per-system PNGs into one 2×2 panel."""
    use = systems[:4]
    if len(use) != 4:
        raise ValueError("semantic_pdf_dade_all.png expects exactly 4 systems for a 2×2 panel")

    import matplotlib.image as mpimg

    fig, axes = plt.subplots(2, 2, figsize=(11.2, 6.2), dpi=220)
    axes = axes.flatten()

    for i, sys in enumerate(use):
        ax = axes[i]
        p = per_system_pngs.get(sys)
        if not p:
            raise FileNotFoundError(f"Missing per-system png for {sys}")
        img = mpimg.imread(p)
        ax.imshow(img)
        ax.set_axis_off()
        # No extra per-panel title here; each per-system PNG already contains
        # its own title (e.g., "... - plants"), and adding another title can clip.

    # No suptitle: the paper will provide the caption/title.
    fig.subplots_adjust(left=0.02, right=0.985, bottom=0.04, top=0.98, wspace=0.06, hspace=0.14)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def _offdiag_values(S: np.ndarray, labels: List[int] | None = None) -> np.ndarray:
    """Collect upper-triangular off-diagonal similarities.

    If labels is provided, pairs where either endpoint is -1 will be skipped
    (mirrors the GT filtering used elsewhere).
    """
    n = int(S.shape[0])
    vals = []
    for i in range(n):
        for j in range(i + 1, n):
            if labels is not None:
                if i >= len(labels) or j >= len(labels):
                    continue
                if int(labels[i]) == -1 or int(labels[j]) == -1:
                    continue
            vals.append(float(S[i, j]))
    return np.asarray(vals, dtype=float)


def _plot_smoothing_panel(
    ax: plt.Axes,
    system: str,
    vals: np.ndarray,
    *,
    bins: int,
    xlim: Tuple[float, float] = (0.0, 1.0),
) -> Dict[str, float]:
    """Plot one histogram panel and return summary stats."""
    if vals.size == 0:
        ax.set_title(system)
        ax.text(0.5, 0.5, "No pairs", ha="center", va="center")
        ax.set_axis_off()
        return {"mean": float("nan"), "median": float("nan"), "n": 0}

    mean = float(np.mean(vals))
    median = float(np.median(vals))

    ax.hist(vals, bins=bins, range=xlim, color="#8FDEE3", edgecolor="#333333", linewidth=0.6)
    ax.axvline(mean, color="#1f77b4", linewidth=2.0, label=f"mean={mean:.3f}")
    ax.axvline(median, color="#999999", linewidth=2.0, linestyle="--", label=f"median={median:.3f}")

    ax.set_title(system)
    ax.set_xlim(*xlim)
    ax.grid(True, alpha=0.25)
    ax.set_xlabel("Cosine similarity")
    ax.set_ylabel("Count")

    # Small, unobtrusive legend per panel
    ax.legend(loc="upper left", fontsize=8, frameon=True)

    return {"mean": mean, "median": median, "n": int(vals.size)}


def _plot_semantic_smoothing_2x2(
    systems: List[str],
    matrices: Dict[str, np.ndarray],
    labels_by_system: Dict[str, List[int] | None],
    out_path: Path,
    *,
    bins: int,
) -> Dict[str, Dict[str, float]]:
    """2×2 histogram figure for semantic smoothing claim.

    We compute cosine similarities from the *semantic similarity matrix* (S_sem_
    or S_sem_dade depending on caller), using off-diagonal upper-triangle pairs.
    """
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.2), dpi=220)
    axes = axes.flatten()

    summaries: Dict[str, Dict[str, float]] = {}
    for k, system in enumerate(systems[:4]):
        ax = axes[k]
        S = matrices[system]
        labels = labels_by_system.get(system)
        vals = _offdiag_values(S, labels=labels)
        summaries[system] = _plot_smoothing_panel(ax, system, vals, bins=bins)

    # hide any unused panels (if systems < 4)
    for j in range(len(systems), 4):
        axes[j].set_axis_off()

    fig.suptitle("Semantic smoothing: off-diagonal cosine similarity distribution", y=0.98)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    return summaries


def _plot_semantic_smoothing_1x4(
    systems: List[str],
    matrices: Dict[str, np.ndarray],
    labels_by_system: Dict[str, List[int] | None],
    out_path: Path,
    *,
    bins: int,
) -> Dict[str, Dict[str, float]]:
    """1×4 compact histogram figure for semantic smoothing claim."""
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 4, figsize=(16.0, 3.6), dpi=220)

    summaries: Dict[str, Dict[str, float]] = {}
    for k, system in enumerate(systems[:4]):
        ax = axes[k]
        S = matrices[system]
        labels = labels_by_system.get(system)
        vals = _offdiag_values(S, labels=labels)
        summaries[system] = _plot_smoothing_panel(ax, system, vals, bins=bins)

        # reduce label clutter for compact layout
        if k != 0:
            ax.set_ylabel("")
            ax.tick_params(axis="y", labelleft=False)

    fig.suptitle("Semantic smoothing: off-diagonal cosine similarity distribution", y=1.02)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return summaries


# --- NEW: combined figure helper (semantic_pdf_dade_all.png) ---
def _plot_all_2x2_panel(
    systems: List[str],
    per_system_pngs: Dict[str, Path],
    out_path: Path,
    *,
    suptitle: str = "Semantic similarity PDF (pre/post DADE) — all systems",
) -> None:
    """Combine the 4 per-system PNGs into one 2×2 panel."""
    use = systems[:4]
    if len(use) != 4:
        raise ValueError("semantic_pdf_dade_all.png expects exactly 4 systems for a 2×2 panel")

    import matplotlib.image as mpimg

    fig, axes = plt.subplots(2, 2, figsize=(11.2, 6.2), dpi=220)
    axes = axes.flatten()

    for i, sys in enumerate(use):
        ax = axes[i]
        p = per_system_pngs.get(sys)
        if not p:
            raise FileNotFoundError(f"Missing per-system png for {sys}")
        img = mpimg.imread(p)
        ax.imshow(img)
        ax.set_axis_off()
        # No extra per-panel title here; each per-system PNG already contains
        # its own title (e.g., "... - plants"), and adding another title can clip.

    # No suptitle: the paper will provide the caption/title.
    fig.subplots_adjust(left=0.02, right=0.985, bottom=0.04, top=0.98, wspace=0.06, hspace=0.14)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def _plot_one_allpairs_legacy(
    system: str,
    raw: np.ndarray,
    dade: np.ndarray,
    labels: List[int],
    out_path: Path,
    *,
    bins: int = 60,
):
    """Legacy style: only plot Raw(all) vs DADE(all), per-panel autoscale.

    This matches the older paper figure style: two lines and title suffix
    '(all pairs)'.
    """
    raw_all, _, _ = _pair_values(raw, labels)
    d_all, _, _ = _pair_values(dade, labels)

    fig, ax = plt.subplots(figsize=(7.2, 4.2))

    for name, vals, color in [
        ("Raw", raw_all, "#999999"),
        ("DADE", d_all, "#1f77b4"),
    ]:
        if vals.size == 0:
            continue
        x, y = _density(vals, bins=bins)
        ax.plot(x, y, label=name, color=color, linewidth=2)

    ax.set_title(f"Semantic similarity PDF: Raw vs DADE (all pairs) — {system}")
    ax.set_xlabel("Cosine similarity")
    ax.set_ylabel("Probability density")
    ax.set_xlim(0, 1)
    ax.grid(True, alpha=0.25)
    ax.legend(ncol=2, fontsize=9)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--systems", type=str, default="acmeair,daytrader,plants,jpetstore")
    ap.add_argument(
        "--input_root",
        type=str,
        default=str(ROOT),
        help=(
            "Root folder that contains data/processed/... (e.g., project root or a snapshot root like "
            "results/paper_snapshot/paper_v1)."
        ),
    )
    ap.add_argument("--paper_mode", action="store_true", default=False, help="Write stable filenames + manifest.")
    ap.add_argument("--out_dir", type=str, default=None, help="Override output directory.")
    ap.add_argument("--bins", type=int, default=60)

    # NEW: allow generating only the smoothing histogram without requiring DADE inputs
    ap.add_argument(
        "--skip_pdf",
        action="store_true",
        default=False,
        help=(
            "Skip generating the semantic PDF (Raw vs DADE) plots. Useful when you only want the --smoothing figure "
            "and the snapshot does not include DADE matrices."
        ),
    )

    # NEW: lock semantic DADE matrix source to a paper base folder
    ap.add_argument(
        "--paper_base_tag",
        type=str,
        default=None,
        help=(
            "If set, read DADE semantic matrices from data/processed/fusion/paper_<tag>_base/ "
            "(expects files: {system}_S_sem_dade_base.npy)."
        ),
    )

    # NEW: control how GT=-1 filtering is handled for the PDF figure
    ap.add_argument(
        "--no_gt_filter",
        action="store_true",
        default=False,
        help=(
            "Do not filter out GT=-1 classes when collecting pair similarities for the PDF plots. "
            "(Keeps all pairs; may change the shape vs older runs.)"
        ),
    )

    # NEW: semantic smoothing 2×2 histogram (paper claim support)
    ap.add_argument(
        "--smoothing",
        action="store_true",
        default=False,
        help="Also generate semantic_smoothing_2x2.png (off-diagonal similarity histogram).",
    )
    ap.add_argument(
        "--smoothing_matrix",
        type=str,
        default="raw",
        choices=["raw", "dade"],
        help="Which matrix to use for smoothing histogram: 'raw' uses *_S_sem.npy; 'dade' uses *_S_sem_dade_base.npy.",
    )

    # NEW: control layout of the semantic smoothing figure
    ap.add_argument(
        "--smoothing_layout",
        type=str,
        default="2x2",
        choices=["2x2", "1x4"],
        help="Layout for the --smoothing figure: '2x2' (default) or '1x4' compact horizontal strip.",
    )

    # NEW: also write combined 2×2 panel (semantic_pdf_dade_all.png)
    ap.add_argument(
        "--write_all",
        action="store_true",
        default=True,
        help="Write the combined 2×2 panel figure semantic_pdf_dade_all.png.",
    )
    ap.add_argument(
        "--mode",
        type=str,
        default="full",
        choices=["full", "legacy_allpairs"],
        help=(
            "Plot style. 'full' draws Raw/DADE for all/intra/inter (6 lines). "
            "'legacy_allpairs' draws only Raw(all) vs DADE(all) (2 lines), matching the older figure."
        ),
    )
    ap.add_argument(
        "--only_all",
        action="store_true",
        default=False,
        help="Plot only Raw(all) vs DADE(all) (2 lines) for each system.",
    )

    args = ap.parse_args()

    systems = [s.strip().lower() for s in args.systems.split(",") if s.strip()]

    input_root = Path(args.input_root).resolve()
    data_root = input_root / "data" / "processed"

    out_dir = Path(args.out_dir).resolve() if args.out_dir else (ROOT / "results" / "plots")
    if args.paper_mode:
        out_dir = Path(args.out_dir).resolve() if args.out_dir else (ROOT / "results" / "paper")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Collect inputs for manifest
    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "input_root": str(input_root).replace("\\", "/"),
        "systems": systems,
        "bins": int(args.bins),
        "mode": str(args.mode),
        "skip_pdf": bool(args.skip_pdf),
        "smoothing": bool(args.smoothing),
        "smoothing_matrix": str(args.smoothing_matrix),
        "files": [],
    }

    # --- Existing per-system DADE PDF plot (kept) ---
    # Plot per-system and combined
    per_paths = {}
    if not args.skip_pdf:
        for system in systems:
            raw_path = data_root / "fusion" / f"{system}_S_sem.npy"
            if args.paper_base_tag:
                dade_path = data_root / "fusion" / f"paper_{args.paper_base_tag}_base" / f"{system}_S_sem_dade_base.npy"
            else:
                # Backwards-compatible default
                dade_path = data_root / "fusion" / f"{system}_S_sem_dade_base.npy"
            gt_path = data_root / "groundtruth" / f"{system}_ground_truth.json"
            order_path = data_root / "fusion" / f"{system}_class_order.json"

            for p in [raw_path, dade_path, gt_path, order_path]:
                if not p.exists():
                    raise FileNotFoundError(f"Missing required input: {p}")
                st = p.stat()
                manifest["files"].append(
                    {
                        "system": system,
                        "rel": str(p.relative_to(input_root)).replace("\\", "/"),
                        "size": int(st.st_size),
                        "mtime": int(st.st_mtime),
                        "sha256": _sha256(p),
                    }
                )

            raw = np.load(str(raw_path))
            dade = np.load(str(dade_path))
            labels = _load_gt_labels(gt_path, order_path)

            # Optionally keep GT=-1 pairs (to match older figures if they didn’t filter)
            if args.no_gt_filter:
                labels = [0] * len(labels)

            out_path = out_dir / f"semantic_pdf_dade_{system}.png"
            if args.mode == "legacy_allpairs":
                _plot_one_allpairs_legacy(system, raw, dade, labels, out_path, bins=int(args.bins))
            else:
                _plot_one(system, raw, dade, labels, out_path, bins=int(args.bins), only_all=bool(args.only_all))
            per_paths[system] = out_path

        # --- NEW: combined 2×2 panel figure (semantic_pdf_dade_all.png) ---
        if args.write_all:
            all_out = out_dir / "semantic_pdf_dade_all.png"
            _plot_all_2x2_panel(systems, per_paths, all_out)
            if args.paper_mode:
                all_manifest = {
                    "created_at": datetime.now().isoformat(timespec="seconds"),
                    "input_root": str(input_root).replace("\\", "/"),
                    "systems": systems[:4],
                    "bins": int(args.bins),
                    "gt_filter": (not bool(args.no_gt_filter)),
                    "sources": {k: str(v).replace("\\", "/") for k, v in per_paths.items()},
                }
                (out_dir / "semantic_pdf_dade_all.panel.manifest.json").write_text(
                    json.dumps(all_manifest, indent=2), encoding="utf-8"
                )
            print(f"[OK] combined figure: {all_out}")

    # --- NEW: semantic smoothing 2×2 histogram (paper claim support) ---
    if args.smoothing:
        matrices: Dict[str, np.ndarray] = {}
        labels_by_system: Dict[str, List[int] | None] = {}
        for system in systems[:4]:
            raw_path = data_root / "fusion" / f"{system}_S_sem.npy"
            if args.paper_base_tag:
                dade_path = data_root / "fusion" / f"paper_{args.paper_base_tag}_base" / f"{system}_S_sem_dade_base.npy"
            else:
                dade_path = data_root / "fusion" / f"{system}_S_sem_dade_base.npy"
            gt_path = data_root / "groundtruth" / f"{system}_ground_truth.json"
            order_path = data_root / "fusion" / f"{system}_class_order.json"

            mat_path = raw_path if args.smoothing_matrix == "raw" else dade_path
            if not mat_path.exists():
                raise FileNotFoundError(
                    f"Missing required input for smoothing_matrix='{args.smoothing_matrix}': {mat_path}\n"
                    "Tip: if you are using a snapshot input_root that doesn't include DADE matrices, run with --smoothing_matrix raw."
                )
            matrices[system] = np.load(str(mat_path))

            if args.no_gt_filter:
                labels_by_system[system] = None
            else:
                # still require GT+order for filtering, but give a clearer error
                for p in [gt_path, order_path]:
                    if not p.exists():
                        raise FileNotFoundError(f"Missing required input for GT filtering: {p}")
                labels_by_system[system] = _load_gt_labels(gt_path, order_path)

        if args.smoothing_layout == "1x4":
            out_path = out_dir / "semantic_smoothing_1x4.png"
            summaries = _plot_semantic_smoothing_1x4(
                systems=systems[:4],
                matrices=matrices,
                labels_by_system=labels_by_system,
                out_path=out_path,
                bins=int(args.bins),
            )
        else:
            out_path = out_dir / "semantic_smoothing_2x2.png"
            summaries = _plot_semantic_smoothing_2x2(
                systems=systems[:4],
                matrices=matrices,
                labels_by_system=labels_by_system,
                out_path=out_path,
                bins=int(args.bins),
            )

        if args.paper_mode:
            smoothing_manifest = {
                "created_at": datetime.now().isoformat(timespec="seconds"),
                "input_root": str(input_root).replace("\\", "/"),
                "systems": systems[:4],
                "bins": int(args.bins),
                "matrix": args.smoothing_matrix,
                "layout": str(args.smoothing_layout),
                "gt_filter": (not bool(args.no_gt_filter)),
                "summaries": summaries,
            }
            (out_dir / f"semantic_smoothing_{args.smoothing_layout}.manifest.json").write_text(
                json.dumps(smoothing_manifest, indent=2),
                encoding="utf-8",
            )
            print(f"[OK] semantic smoothing figure: {out_path}")

    # Write manifest in paper mode
    if args.paper_mode:
        manifest_path = out_dir / "semantic_pdf_dade_all.manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(f"[OK] manifest: {manifest_path}")


if __name__ == "__main__":
    main()
