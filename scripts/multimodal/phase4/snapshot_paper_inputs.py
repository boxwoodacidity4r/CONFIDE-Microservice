"""Snapshot paper-critical inputs and write a manifest with hashes.

Why
---
Your repo has many generated intermediate artifacts that can be overwritten.
To make figures/tables *stable* for paper writing, we snapshot the exact
inputs used by Phase4 figures/tables into a dedicated directory and record
SHA256 hashes.

This script copies a minimal, auditable set of files:
- Semantic similarity matrices pre/post DADE: data/processed/fusion/*_S_sem*.npy
- Structural/final matrices used by Phase3/4 (optional but helpful): *_S_final.npy, *_S_struct.npy
- Uncertainty matrices: data/processed/edl/*_edl_uncertainty.npy
- Alignment files: class_order.json, ground_truth.json

Outputs
-------
- results/paper_snapshot/<timestamp_or_tag>/... (mirrors relative paths)
- results/paper_snapshot/<tag>/manifest.json

Usage (PowerShell)
------------------
python scripts\multimodal\phase4\snapshot_paper_inputs.py --tag paper_v1
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List


ROOT = Path(__file__).resolve().parents[3]


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _candidate_files(systems: List[str]) -> List[Path]:
    out: List[Path] = []
    # alignment
    for sys in systems:
        out.append(ROOT / "data" / "processed" / "fusion" / f"{sys}_class_order.json")
        out.append(ROOT / "data" / "processed" / "groundtruth" / f"{sys}_ground_truth.json")
        out.append(ROOT / "data" / "processed" / "edl" / f"{sys}_edl_uncertainty.npy")
        # semantic matrices
        out.append(ROOT / "data" / "processed" / "fusion" / f"{sys}_S_sem.npy")
        # NEW naming: base (paper-safe)
        out.append(ROOT / "data" / "processed" / "fusion" / f"{sys}_S_sem_dade_base.npy")
        # Optional: include all rho variants if they exist (ablation artifacts)
        for p in sorted((ROOT / "data" / "processed" / "fusion").glob(f"{sys}_S_sem_dade_rho_*.npy")):
            out.append(p)
        # optional but useful
        out.append(ROOT / "data" / "processed" / "fusion" / f"{sys}_S_struct.npy")
        out.append(ROOT / "data" / "processed" / "fusion" / f"{sys}_S_final.npy")
    # keep only existing
    return [p for p in out if p.exists()]


def _copy_preserve_relative(src: Path, dst_root: Path) -> Path:
    rel = src.relative_to(ROOT)
    dst = dst_root / rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return dst


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--systems", type=str, default="acmeair,daytrader,plants,jpetstore")
    ap.add_argument("--tag", type=str, default=None, help="Snapshot tag (used as folder name).")
    args = ap.parse_args()

    systems = [s.strip().lower() for s in args.systems.split(",") if s.strip()]
    tag = (args.tag or datetime.now().strftime("%Y%m%d_%H%M%S")).strip()

    snap_root = ROOT / "results" / "paper_snapshot" / tag
    snap_root.mkdir(parents=True, exist_ok=True)

    files = _candidate_files(systems)
    if not files:
        raise SystemExit("No candidate input files found. Check data/processed paths.")

    manifest: Dict[str, object] = {
        "tag": tag,
        "systems": systems,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "files": [],
    }

    for src in sorted(files):
        dst = _copy_preserve_relative(src, snap_root)
        st = src.stat()
        manifest["files"].append(
            {
                "rel_path": str(src.relative_to(ROOT)).replace("\\", "/"),
                "size": int(st.st_size),
                "mtime": int(st.st_mtime),
                "sha256": _sha256(src),
            }
        )

    (snap_root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[OK] snapshot: {snap_root}")
    print(f"[OK] manifest: {snap_root / 'manifest.json'}")


if __name__ == "__main__":
    main()
