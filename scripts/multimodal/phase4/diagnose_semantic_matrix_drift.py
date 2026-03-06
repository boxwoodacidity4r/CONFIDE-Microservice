"""Diagnose drift in semantic matrices by summarizing off-diagonal distributions.

This is a read-only diagnostic utility.

Inputs:
  data/processed/fusion/{system}_S_sem.npy
  data/processed/fusion/{system}_S_sem_dade_base.npy

Output:
  Prints per-system summary stats + quantiles.

Usage (PowerShell):
  python scripts/multimodal/phase4/diagnose_semantic_matrix_drift.py
  python scripts/multimodal/phase4/diagnose_semantic_matrix_drift.py --systems acmeair,daytrader
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]


def _offdiag_vals(S: np.ndarray) -> np.ndarray:
    n = int(S.shape[0])
    iu = np.triu_indices(n, k=1)
    v = S[iu].astype(float)
    v = v[np.isfinite(v)]
    return v


def _summarize(v: np.ndarray) -> dict:
    if v.size == 0:
        return {}
    qs = np.quantile(v, [0, 0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99, 1.0])
    return {
        "n": int(v.size),
        "min": float(qs[0]),
        "q01": float(qs[1]),
        "q05": float(qs[2]),
        "q10": float(qs[3]),
        "q25": float(qs[4]),
        "median": float(qs[5]),
        "q75": float(qs[6]),
        "q90": float(qs[7]),
        "q95": float(qs[8]),
        "q99": float(qs[9]),
        "max": float(qs[10]),
        "mean": float(np.mean(v)),
        "std": float(np.std(v)),
        "p>=0.90": float(np.mean(v >= 0.90)),
        "p>=0.95": float(np.mean(v >= 0.95)),
        "p>=0.99": float(np.mean(v >= 0.99)),
        "p<=0.10": float(np.mean(v <= 0.10)),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--systems", type=str, default="acmeair,daytrader,plants,jpetstore")
    args = ap.parse_args()

    systems = [s.strip().lower() for s in args.systems.split(",") if s.strip()]
    fusion = ROOT / "data" / "processed" / "fusion"

    print("Fusion semantic matrix distribution summary (upper triangle off-diagonal)")
    for sys in systems:
        raw_p = fusion / f"{sys}_S_sem.npy"
        dade_p = fusion / f"{sys}_S_sem_dade_base.npy"

        raw = np.load(raw_p)
        dade = np.load(dade_p)

        vr = _offdiag_vals(raw)
        vd = _offdiag_vals(dade)

        sr = _summarize(vr)
        sd = _summarize(vd)

        print(f"\n=== {sys} ===")
        print(f"raw shape: {tuple(raw.shape)}; dade shape: {tuple(dade.shape)}")

        keys = [
            "n",
            "min",
            "q05",
            "q25",
            "median",
            "q75",
            "q95",
            "q99",
            "max",
            "mean",
            "std",
            "p>=0.95",
            "p>=0.99",
            "p<=0.10",
        ]

        def fmt(s: dict, k: str) -> str:
            if not s or k not in s:
                return f"{k}=NA"
            v = s[k]
            return f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"

        print("RAW : " + ", ".join(fmt(sr, k) for k in keys))
        print("DADE: " + ", ".join(fmt(sd, k) for k in keys))


if __name__ == "__main__":
    main()
