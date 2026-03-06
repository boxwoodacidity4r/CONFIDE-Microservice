"""Parameter sensitivity (reviewer-facing): inference-side evidence_smooth sweep.

Goal
----
Demonstrate that small changes in the EDL inference smoothing factor
(--evidence_smooth in Phase2 `edl_infer.py`) do NOT cause dramatic changes in
final decomposition quality.

Key constraint: do NOT change existing core data.
- We keep the trained EDL model checkpoints fixed.
- We only re-run *inference* with different evidence_smooth values.
- Then we re-run Phase3 evaluation using the generated uncertainty matrices
  via `phase3_cac_evaluation.py --u_path ...`.

Outputs
-------
- results/ablation/edl_smooth_sensitivity/edl_smooth_sensitivity.csv
- results/plots/edl_smooth_sensitivity_<ts>.png

Usage (example)
--------------
python scripts/multimodal/phase4/run_edl_infer_smooth_sensitivity.py \
  --systems acmeair,daytrader,plants,jpetstore \
  --smooth_list 0.0,0.05,0.1,0.2,0.3 \
  --models data/processed/edl/edl_model_acmeair.pt,data/processed/edl/edl_model_daytrader.pt,data/processed/edl/edl_model_plants.pt,data/processed/edl/edl_model_jpetstore.pt \
  --scalers data/processed/edl/acmeair_scaler.pkl,data/processed/edl/daytrader_scaler.pkl,data/processed/edl/plants_scaler.pkl,data/processed/edl/jpetstore_scaler.pkl

Notes
-----
- This script assumes Phase2 inference features exist: data/processed/edl/<sys>_all_X.npy
- This script uses the same Phase3 "golden" settings as `phase3/run_final_eval.py`.
"""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[3]
PHASE2_INFER = ROOT / "scripts" / "multimodal" / "phase2" / "edl_infer.py"
PHASE3_EVAL = ROOT / "scripts" / "multimodal" / "phase3" / "phase3_cac_evaluation.py"

# Keep consistent with scripts/multimodal/phase3/run_final_eval.py
GOLDEN = {
    "acmeair": (5, 5, 0.14),
    "daytrader": (5, 8, 0.18),
    "plants": (3, 6, 0.22),
    "jpetstore": (4, 8, 0.14),
}
COMMON = ["--mode", "sigmoid", "--alpha", "15", "--merge_small_clusters", "--min_cluster_size", "3"]


@dataclass
class Row:
    system: str
    evidence_smooth: float
    u_path: str
    cac_q: float | None
    cac_ifn_ratio: float | None
    cac_services: int | None
    baseline_q: float | None
    baseline_ifn_ratio: float | None
    baseline_services: int | None


def _parse_metrics(stdout: str) -> dict:
    """Parse Phase3 output.

    Supported formats (current repo):
      - [ResolutionBest] <sys> | CAC-Final: ... services=<K> Q=<q> IFN_Ratio=<r>
    Legacy formats may print table rows; we try best-effort parsing.

    Returns: {'Baseline': {...}?, 'CAC-Final': {...}?}
    """
    out: dict = {}
    for line in (stdout or "").splitlines():
        s = (line or "").strip()
        if not s:
            continue

        # Current stable format
        if s.startswith("[ResolutionBest]") and "CAC-Final" in s and "Q=" in s and "IFN_Ratio=" in s:
            # Example:
            # [ResolutionBest] acmeair | CAC-Final: best_res=0.8500  services=5 Q=0.085476 IFN_Ratio=0.369748
            try:
                # system
                sys_part = s.split("[ResolutionBest]", 1)[1].strip()
                system = sys_part.split("|", 1)[0].strip()

                # numbers
                def _grab(key: str) -> float:
                    return float(s.split(key, 1)[1].split()[0])

                services = int(_grab("services="))
                q = float(_grab("Q="))
                ifn_ratio = float(_grab("IFN_Ratio="))

                out["CAC-Final"] = {"Q": q, "IFN_Ratio": ifn_ratio, "Services": services}
                out.setdefault("system", system)
            except Exception:
                pass

        # Optional baseline summary if it exists in some runs (best-effort)
        if s.startswith("[ResolutionBest]") and "Baseline" in s and "Q=" in s and "IFN_Ratio=" in s:
            try:
                def _grab(key: str) -> float:
                    return float(s.split(key, 1)[1].split()[0])

                services = int(_grab("services="))
                q = float(_grab("Q="))
                ifn_ratio = float(_grab("IFN_Ratio="))
                out["Baseline"] = {"Q": q, "IFN_Ratio": ifn_ratio, "Services": services}
            except Exception:
                pass

    return out


def _run(cmd: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)


def infer_u(system: str, *, smooth: float, model: str, scaler: str, out_path: Path) -> None:
    x_path = ROOT / "data" / "processed" / "edl" / f"{system}_all_X.npy"
    class_order = ROOT / "data" / "processed" / "fusion" / f"{system}_class_order.json"

    cmd = [
        sys.executable,
        str(PHASE2_INFER),
        "--X",
        str(x_path),
        "--model",
        str(Path(model)),
        "--class_order",
        str(class_order),
        "--scaler",
        str(Path(scaler)),
        "--out",
        str(out_path),
        "--system",
        system,
        "--evidence_smooth",
        str(float(smooth)),
        # Keep evidence_scale fixed to 1.0 to avoid changing core behavior
        "--evidence_scale",
        "1.0",
        # Keep numeric matrix unchanged (no normalization at Phase2 stage)
        "--u_norm",
        "none",
    ]

    p = _run(cmd)
    if p.returncode != 0:
        raise RuntimeError(
            "EDL infer failed for system={} smooth={}\nstdout:\n{}\nstderr:\n{}".format(
                system, smooth, p.stdout[-2000:], p.stderr[-2000:]
            )
        )


def eval_phase3(system: str, *, u_path: Path) -> dict:
    tmin, tmax, cap = GOLDEN[system]
    cmd = [
        sys.executable,
        str(PHASE3_EVAL),
        system,
        "--dpep_cap",
        str(float(cap)),
        "--target_min",
        str(int(tmin)),
        "--target_max",
        str(int(tmax)),
        "--u_path",
        str(u_path),
        *COMMON,
    ]

    p = _run(cmd)
    if p.returncode != 0:
        raise RuntimeError(
            "Phase3 eval failed for system={} u_path={}\nstdout:\n{}\nstderr:\n{}".format(
                system, u_path, p.stdout[-2000:], p.stderr[-2000:]
            )
        )
    return _parse_metrics((p.stdout or "") + "\n" + (p.stderr or ""))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--systems", type=str, default="acmeair,daytrader,plants,jpetstore")
    ap.add_argument(
        "--smooth_list",
        type=str,
        default="0.0,0.05,0.1,0.2,0.3",
        help="Comma-separated evidence_smooth values to sweep.",
    )
    ap.add_argument(
        "--models",
        type=str,
        default=None,
        help="Comma-separated model checkpoint paths aligned with systems. If omitted, use data/processed/edl/edl_model_<sys>.pt",
    )
    ap.add_argument(
        "--scalers",
        type=str,
        default=None,
        help="Comma-separated scaler paths aligned with systems. If omitted, use data/processed/edl/<sys>_scaler.pkl",
    )
    args = ap.parse_args()

    systems = [s.strip().lower() for s in args.systems.split(",") if s.strip()]
    smooth_list = [float(x.strip()) for x in args.smooth_list.split(",") if x.strip()]

    # Resolve model/scaler paths
    if args.models:
        models = [s.strip() for s in args.models.split(",") if s.strip()]
    else:
        models = [str(ROOT / "data" / "processed" / "edl" / f"edl_model_{sys}.pt") for sys in systems]

    if args.scalers:
        scalers = [s.strip() for s in args.scalers.split(",") if s.strip()]
    else:
        scalers = [str(ROOT / "data" / "processed" / "edl" / f"{sys}_scaler.pkl") for sys in systems]

    if len(models) != len(systems) or len(scalers) != len(systems):
        raise ValueError("--models/--scalers must have the same length as --systems")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = ROOT / "results" / "ablation" / "edl_smooth_sensitivity"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Store inferred U matrices under data/processed to keep Phase3 path conventions clear
    u_out_dir = ROOT / "data" / "processed" / "edl" / "sensitivity" / f"smooth_{ts}"
    u_out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[Row] = []

    for smooth in smooth_list:
        print(f"[SmoothSweep] evidence_smooth={smooth}", flush=True)
        for sys_name, model, scaler in zip(systems, models, scalers):
            u_path = u_out_dir / f"{sys_name}_edl_uncertainty_smooth{smooth}.npy"
            print(f"  - infer {sys_name} -> {u_path.name}", flush=True)
            infer_u(sys_name, smooth=smooth, model=model, scaler=scaler, out_path=u_path)

            print(f"  - eval phase3 {sys_name}", flush=True)
            metrics = eval_phase3(sys_name, u_path=u_path)
            b = metrics.get("Baseline") or {}
            c = metrics.get("CAC-Final") or {}
            rows.append(
                Row(
                    system=sys_name,
                    evidence_smooth=float(smooth),
                    u_path=str(u_path),
                    cac_q=c.get("Q"),
                    cac_ifn_ratio=c.get("IFN_Ratio"),
                    cac_services=c.get("Services"),
                    baseline_q=b.get("Q"),
                    baseline_ifn_ratio=b.get("IFN_Ratio"),
                    baseline_services=b.get("Services"),
                )
            )

    # Write CSV
    csv_path = out_dir / "edl_smooth_sensitivity.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "system",
                "evidence_smooth",
                "baseline_services",
                "baseline_q",
                "baseline_ifn_ratio",
                "cac_services",
                "cac_q",
                "cac_ifn_ratio",
                "u_path",
            ]
        )
        for r in rows:
            w.writerow(
                [
                    r.system,
                    r.evidence_smooth,
                    r.baseline_services,
                    r.baseline_q,
                    r.baseline_ifn_ratio,
                    r.cac_services,
                    r.cac_q,
                    r.cac_ifn_ratio,
                    r.u_path,
                ]
            )

    # Plot: IFN ratio vs smooth
    fig, ax = plt.subplots(figsize=(7.5, 4.5), dpi=200)
    for sys_name in systems:
        pts = [r for r in rows if r.system == sys_name]
        pts.sort(key=lambda r: r.evidence_smooth)
        xs = [r.evidence_smooth for r in pts]
        ys = [float(r.cac_ifn_ratio) if r.cac_ifn_ratio is not None else float("nan") for r in pts]
        ax.plot(xs, ys, marker="o", linewidth=1.8, label=sys_name)

    ax.set_xlabel("Inference smoothing (evidence_smooth)")
    ax.set_ylabel("CAC IFN_Ratio (lower is better)")
    ax.set_title("Parameter sensitivity: inference-side evidence_smooth")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(ncol=2, fontsize=9)

    png_path = ROOT / "results" / "plots" / f"edl_smooth_sensitivity_{ts}.png"
    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(png_path, bbox_inches="tight")
    plt.close(fig)

    print(f"[OK] CSV saved: {csv_path}")
    print(f"[OK] Plot saved: {png_path}")
    print(f"[INFO] U matrices saved under: {u_out_dir}")


if __name__ == "__main__":
    main()
