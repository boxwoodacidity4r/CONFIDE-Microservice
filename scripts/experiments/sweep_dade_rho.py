import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import matplotlib.pyplot as plt

# -------------------- Config --------------------
SYSTEMS = ["acmeair", "daytrader", "jpetstore", "plants"]

# Default sweep range (can be overridden by CLI)
# Paper-friendly grid: coarse-to-fine search with denser sampling in the typical
# stability region. You can always override via --rhos.
DEFAULT_RHOS = [
    0.03,
    0.04,
    0.05,
    0.06,
    0.07,
    0.08,
    0.09,
    0.10,
    0.11,
    0.12,
    0.13,
    0.14,
    0.15,
]

OUTPUT_CSV = Path("results/dade_sensitivity_results.csv")
OUTPUT_PLOT = Path("results/plots/dade_sensitivity_analysis.png")

# Which matrix ratio to record from diagnose output
TARGET_MATRIX = "final"  # typically most convincing for papers

# Default location of DADE matrix produced by rescale_semantic_dade.py
DADE_MATRIX_PATH_TMPL = "data/processed/fusion/{system}_S_sem_dade.npy"

# -------------------- Helpers --------------------

def run_cmd(cmd: List[str], cwd: str | None = None) -> str:
    """Run a command and return stdout (raises on failure).

    Notes:
      - Uses text mode utf-8.
      - Prints stderr to help debugging upstream scripts.
    """
    proc = subprocess.run(
        cmd,
        cwd=cwd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=True,
    )
    # keep stderr in case upstream scripts log there
    if proc.stderr and proc.stderr.strip():
        print(proc.stderr)
    return proc.stdout


def parse_ratios_from_diagnose(
    output: str,
    systems: List[str],
    matrix: str = TARGET_MATRIX,
) -> Dict[str, float]:
    """Parse per-system ratio from diagnose output.

    We search for a row like:
      <system> | <matrix> | <S_intra> | <S_inter> | <Ratio> |

    This implementation is tolerant to:
      - extra whitespace
      - scientific notation
      - minor column width variations
    """
    ratios: Dict[str, float] = {}

    # number pattern: 1, 1.0, .5, 1e-3, 1.2E+02 etc.
    num = r"[+-]?(?:\d+\.\d+|\d+|\.\d+)(?:[eE][+-]?\d+)?"

    # Example line:
    # daytrader  | final     |  0.190 |  0.104 |  1.82 | 67/123
    for sys_name in systems:
        # capture the Ratio column (5th numeric column in that row)
        pattern = (
            rf"^\s*{re.escape(sys_name)}\s*\|\s*{re.escape(matrix)}\s*\|\s*"
            rf"{num}\s*\|\s*{num}\s*\|\s*({num})\s*\|"
        )
        m = re.search(pattern, output, flags=re.MULTILINE)
        if m:
            ratios[sys_name] = float(m.group(1))

    return ratios


def configure_plot_style_for_bw() -> Tuple[List[str], List[str]]:
    """Return (linestyles, markers) suitable for B/W printing."""
    linestyles = ["-", "--", "-.", ":"]
    markers = ["o", "s", "^", "D"]
    return linestyles, markers


def compute_dade_density(dade_npy_path: Path) -> Dict[str, float]:
    """Compute simple graph-like connectivity metrics for DADE matrix.

    Metrics:
      - nnz_ratio: non-zero ratio of the matrix
      - avg_degree: average number of non-zeros per row

    If file missing or loading fails, returns empty dict.
    """
    if not dade_npy_path.exists():
        return {}

    try:
        import numpy as np

        mat = np.load(dade_npy_path)
        if mat.ndim != 2:
            return {}

        # treat exact zeros as absence (DADE top-k creates hard zeros)
        nnz = float((mat != 0).sum())
        rows, cols = mat.shape
        total = float(rows * cols) if rows * cols else 0.0
        nnz_ratio = (nnz / total) if total else 0.0
        avg_degree = (nnz / rows) if rows else 0.0
        return {"dade_nnz_ratio": nnz_ratio, "dade_avg_degree": avg_degree}
    except Exception:
        return {}


def sweep(
    rhos: List[float],
    systems: List[str],
    workspace_root: Path,
    output_csv: Path,
    output_plot: Path,
    matrix: str,
    python_exe: str,
    collect_density: bool,
) -> pd.DataFrame:
    results = []

    # Ensure output dirs exist
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_plot.parent.mkdir(parents=True, exist_ok=True)

    # Use your existing scripts; build consumes DADE file from fusion/.
    dade_script = workspace_root / "scripts" / "multimodal" / "phase1" / "rescale_semantic_dade.py"
    build_script = workspace_root / "scripts" / "multimodal" / "phase1" / "build_multimodal_matrices.py"
    # NOTE: diagnose script lives under phase4 in this repo.
    diag_script = workspace_root / "scripts" / "multimodal" / "phase4" / "diagnose_intra_inter_similarity.py"

    for rho in rhos:
        print(f"\n=== Sweeping rho={rho:.3f} ===")

        # 1) regenerate DADE for each system with current rho (topk-ratio)
        for sys_name in systems:
            print(f"[DADE] {sys_name} rho={rho:.3f}")
            run_cmd(
                [
                    python_exe,
                    str(dade_script),
                    "--system",
                    sys_name,
                    "--topk-ratio",
                    f"{rho}",
                ],
                cwd=str(workspace_root),
            )

        # 2) rebuild matrices so final uses updated DADE
        for sys_name in systems:
            print(f"[BUILD] {sys_name}")
            run_cmd(
                [
                    python_exe,
                    str(build_script),
                    "--system",
                    sys_name,
                ],
                cwd=str(workspace_root),
            )

        # 3) batch diagnose
        diag_output = run_cmd(
            [
                python_exe,
                str(diag_script),
                "--batch",
            ],
            cwd=str(workspace_root),
        )

        ratios = parse_ratios_from_diagnose(diag_output, systems=systems, matrix=matrix)

        for sys_name in systems:
            if sys_name not in ratios:
                print(
                    f"[WARN] Missing ratio parsed for {sys_name} at rho={rho:.3f} (matrix={matrix}). "
                    "Check diagnose output formatting or system names."
                )
                continue

            row = {"rho": rho, "system": sys_name, f"{matrix}_ratio": ratios[sys_name]}

            if collect_density:
                dade_path = workspace_root / DADE_MATRIX_PATH_TMPL.format(system=sys_name)
                row.update(compute_dade_density(dade_path))

            results.append(row)

    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False, encoding="utf-8")
    print(f"\n[OK] Saved CSV -> {output_csv}")

    return df


def plot(df: pd.DataFrame, output_plot: Path, matrix: str):
    if df.empty:
        raise RuntimeError("No results to plot (empty DataFrame).")

    ycol = f"{matrix}_ratio"

    plt.figure(figsize=(8, 5))
    linestyles, markers = configure_plot_style_for_bw()

    # Per-system curves
    for i, sys_name in enumerate(SYSTEMS):
        sys_data = df[df["system"] == sys_name].sort_values("rho")
        if sys_data.empty:
            continue
        plt.plot(
            sys_data["rho"],
            sys_data[ycol],
            linestyle=linestyles[i % len(linestyles)],
            marker=markers[i % len(markers)],
            linewidth=1.8,
            markersize=5,
            label=sys_name,
            color=str(0.2 + 0.15 * i),  # grayscale
        )

    # Average line (most convincing)
    avg = df.groupby("rho")[ycol].mean().reset_index().sort_values("rho")
    plt.plot(
        avg["rho"],
        avg[ycol],
        linestyle="-",
        color="black",
        linewidth=2.5,
        label="Average",
    )

    plt.title("Sensitivity of DADE Top-k Ratio (ρ)")
    plt.xlabel("Top-k Ratio (ρ)")
    plt.ylabel("Final Intra/Inter Ratio" if matrix == "final" else f"{matrix} Intra/Inter Ratio")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(frameon=False)
    plt.tight_layout()

    output_plot.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_plot, dpi=300)
    print(f"[OK] Saved plot -> {output_plot}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Sweep DADE top-k ratio (rho) and plot sensitivity.")
    parser.add_argument(
        "--matrix",
        default=TARGET_MATRIX,
        choices=["sem_dade", "final", "struct", "temp", "sem_raw"],
        help="Which matrix row to parse from diagnose output.",
    )
    parser.add_argument(
        "--systems",
        default=",".join(SYSTEMS),
        help="Comma-separated systems to include, e.g. acmeair,daytrader,plants",
    )
    parser.add_argument(
        "--rhos",
        default=",".join(str(r) for r in DEFAULT_RHOS),
        help="Comma-separated rho values, e.g. 0.05,0.08,0.10,0.12",
    )
    parser.add_argument(
        "--output-csv",
        default=str(OUTPUT_CSV),
        help="Output CSV path",
    )
    parser.add_argument(
        "--output-plot",
        default=str(OUTPUT_PLOT),
        help="Output plot path",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable used to run sub-scripts (default: current interpreter).",
    )
    parser.add_argument(
        "--collect-density",
        action="store_true",
        help="Also compute DADE density metrics (nnz ratio / avg degree) and store into CSV.",
    )
    args = parser.parse_args()

    rhos = [float(x.strip()) for x in args.rhos.split(",") if x.strip()]
    selected_systems = [s.strip() for s in args.systems.split(",") if s.strip()]

    unknown = sorted(set(selected_systems) - set(SYSTEMS))
    if unknown:
        raise SystemExit(f"Unknown systems: {unknown}. Allowed: {SYSTEMS}")

    workspace_root = Path(__file__).resolve().parents[2]  # scripts/experiments/..

    df = sweep(
        rhos=rhos,
        systems=selected_systems,
        workspace_root=workspace_root,
        output_csv=Path(args.output_csv),
        output_plot=Path(args.output_plot),
        matrix=args.matrix,
        python_exe=args.python,
        collect_density=args.collect_density,
    )

    # Plot still uses the global SYSTEMS list for consistent legend ordering,
    # but it will simply skip systems absent from the DataFrame.
    plot(df, output_plot=Path(args.output_plot), matrix=args.matrix)


if __name__ == "__main__":
    main()

