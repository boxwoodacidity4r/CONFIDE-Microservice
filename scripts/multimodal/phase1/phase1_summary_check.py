import os
import json
from pathlib import Path
import argparse
from typing import Optional

import numpy as np


# Keep ROOT path logic stable: go up 3 levels to project root
ROOT = Path(__file__).resolve().parents[3]


def load_matrix(path: Path) -> Optional[np.ndarray]:
    if not path.exists():
        print(f"[MISSING] {path}")
        return None
    mat = np.load(path)
    nnz = int(np.count_nonzero(mat))
    diag = np.diag(mat) if mat.ndim == 2 else np.array([])
    diag_mean = float(diag.mean()) if diag.size > 0 else float("nan")
    print(
        f"[OK] {path.name}: shape={mat.shape}, nnz={nnz}, "
        f"diag_mean={diag_mean:.4f}"
    )
    return mat


def check_alignment(struct_mat: np.ndarray, sem_mat: np.ndarray, temp_mat: np.ndarray) -> None:
    if struct_mat is None or sem_mat is None or temp_mat is None:
        print("[WARN] Skipping alignment check due to missing matrix.")
        return

    if not (struct_mat.shape == sem_mat.shape == temp_mat.shape):
        print("[FAIL] Shape mismatch among modalities:")
        print(f"   S_struct: {struct_mat.shape}")
        print(f"   S_sem:    {sem_mat.shape}")
        print(f"   S_temp:   {temp_mat.shape}")
        return

    n, m = struct_mat.shape
    if n != m:
        print(f"[FAIL] Matrices are not square: {struct_mat.shape}")
        return

    print("[OK] All matrices share the same square shape.")

    # Simple cross-modality conflict rate: off-diagonal only
    eye = np.eye(n, dtype=bool)
    struct_nz = (struct_mat != 0) & ~eye
    temp_nz = (temp_mat != 0) & ~eye

    total_pairs = int(np.count_nonzero(struct_nz | temp_nz))
    if total_pairs == 0:
        print("[WARN] No off-diagonal edges in either S_struct or S_temp; conflict rate undefined.")
        return

    # Conflict: an edge appears in only one modality
    conflict = struct_nz ^ temp_nz
    conflict_count = int(np.count_nonzero(conflict))
    conflict_rate = conflict_count / total_pairs

    print(
        f"[CONFLICT] struct vs temp: conflict_rate={conflict_rate:.4f} "
        f"({conflict_count}/{total_pairs} off-diagonal positions)"
    )


def main(system_name: str = "acmeair") -> None:
    fusion_dir = ROOT / "data" / "processed" / "fusion"
    temporal_dir = ROOT / "data" / "processed" / "temporal"

    print(f"=== Phase 1 Matrix Summary ({system_name}) ===")

    s_struct_path = fusion_dir / f"{system_name}_S_struct.npy"
    s_sem_path = fusion_dir / f"{system_name}_S_sem_dade_base.npy"

    s_struct = load_matrix(s_struct_path)
    s_sem = load_matrix(s_sem_path)

    # Temporal matrix candidates:
    # - prefer hybrid, then fusion naming, then raw temporal S_temp.npy
    s_temp_candidates = [
        temporal_dir / f"{system_name}_S_temp_hybrid.npy",
        temporal_dir / f"{system_name}_S_temp_jtl_session.npy",
        temporal_dir / f"{system_name}_S_temp_basic.npy",
        fusion_dir / f"{system_name}_S_temp.npy",
        temporal_dir / f"{system_name}_S_temp.npy",  # ensure we can detect a direct S_temp output
    ]

    s_temp = None
    for p in s_temp_candidates:
        if p.exists():
            print(f"[INFO] Using temporal matrix: {p}")
            s_temp = load_matrix(p)
            break

    if s_temp is None:
        print("[FAIL] No temporal matrix found. Checked:")
        for p in s_temp_candidates:
            print(f"   - {p}")
        return
    else:
        print(f"[STEMP] shape={s_temp.shape}, nnz={np.count_nonzero(s_temp)}, max={s_temp.max():.4f}, min={s_temp.min():.4f}, diag_mean={np.diag(s_temp).mean():.4f}")

    # 2) Shape/alignment + simple conflict-rate check
    print("\n=== Alignment & Conflict Check ===")
    check_alignment(s_struct, s_sem, s_temp)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 1 multimodal matrix summary and alignment check.")
    parser.add_argument("--system", type=str, default="acmeair", help="System name (e.g. acmeair, daytrader, jpetstore, plants)")
    args = parser.parse_args()
    main(args.system)
