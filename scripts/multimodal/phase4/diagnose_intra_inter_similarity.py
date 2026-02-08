import numpy as np
import json
import sys
import os
from typing import Dict, List, Tuple
from pathlib import Path
import hashlib

# Usage:
#   Single: python diagnose_intra_inter_similarity.py <system> <s_matrix_path> <gt_path> <class_order_path>
#   Batch : python diagnose_intra_inter_similarity.py --batch [--verify]
#   Optional: --merge-labels "1,3->1;2->2" (applied in-memory to GT labels)
#            --plants-intent-matrices (also report temp_j1/temp_j2 if present)


def _load_labels(gt_path: str, class_order_path: str) -> Tuple[List[int], int]:
    with open(gt_path, 'r', encoding='utf-8') as f:
        gt_map = json.load(f)
    with open(class_order_path, 'r', encoding='utf-8') as f:
        class_order = json.load(f)
    labels = [gt_map.get(cls, -1) for cls in class_order]
    return labels, len(class_order)


def _apply_merge_labels(labels: List[int], merge_spec: str) -> List[int]:
    """Apply label merging spec to a list of int labels.

    Spec format (order-independent, ';' separated):
      "1,3->1; 0->0; 2->2"
      "1,3->1"  (only merges specified labels; others unchanged)

    Notes:
    - -1 is never rewritten unless explicitly included.
    - RHS must be an int.
    """
    spec = (merge_spec or '').strip()
    if not spec:
        return labels

    mapping: Dict[int, int] = {}
    parts = [p.strip() for p in spec.split(';') if p.strip()]
    for p in parts:
        if '->' not in p:
            raise ValueError(f"Bad --merge-labels token (missing '->'): {p}")
        left, right = [x.strip() for x in p.split('->', 1)]
        if right == '':
            raise ValueError(f"Bad --merge-labels token (empty RHS): {p}")
        tgt = int(right)
        if left == '':
            raise ValueError(f"Bad --merge-labels token (empty LHS): {p}")
        for tok in [x.strip() for x in left.split(',') if x.strip()]:
            mapping[int(tok)] = tgt

    return [mapping.get(int(l), int(l)) for l in labels]


def _safe_ratio(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Return safe ratio, handling inf and NaN."""
    if denominator == 0:
        return float('inf') if numerator > 0 else default
    return numerator / denominator


def _intra_inter_stats(S: np.ndarray, labels: List[int]) -> Tuple[float, float, float, int, int]:
    n = len(labels)
    intra = []
    inter = []
    skipped = 0
    for i in range(n):
        for j in range(i + 1, n):
            if labels[i] == -1 or labels[j] == -1:
                skipped += 1
                continue
            if labels[i] == labels[j]:
                intra.append(float(S[i, j]))
            else:
                inter.append(float(S[i, j]))

    intra_avg = float(np.nanmean(intra)) if intra else 0.0
    inter_avg = float(np.nanmean(inter)) if inter else 0.0
    ratio = _safe_ratio(intra_avg, inter_avg)
    return intra_avg, inter_avg, ratio, len(intra), len(inter)


def _intra_inter_stats_nz(S: np.ndarray, labels: List[int]) -> Tuple[float, float, float, int, int]:
    """Nonzero-only intra/inter stats.

    For sparse modalities (like temporal), averaging over all class pairs can be
    dominated by zeros. This reports the mean strength conditioned on having an edge.

    Returns (intra_avg_nz, inter_avg_nz, nz_ratio, nz_intra_cnt, nz_inter_cnt)
    where counts are number of nonzero pairs in each group.
    """
    n = len(labels)
    intra = []
    inter = []
    for i in range(n):
        for j in range(i + 1, n):
            if labels[i] == -1 or labels[j] == -1:
                continue
            v = float(S[i, j])
            if v <= 0:
                continue
            if labels[i] == labels[j]:
                intra.append(v)
            else:
                inter.append(v)

    intra_avg = float(np.nanmean(intra)) if intra else 0.0
    inter_avg = float(np.nanmean(inter)) if inter else 0.0
    ratio = _safe_ratio(intra_avg, inter_avg)
    return intra_avg, inter_avg, ratio, len(intra), len(inter)


def _try_stat(p: str) -> Tuple[str, str, str]:
    """Return (exists, size, mtime_iso)."""
    path = Path(p)
    if not path.exists():
        return "no", "-", "-"
    st = path.stat()
    # ISO-like without importing datetime; good enough for quick comparisons
    return "yes", str(st.st_size), str(int(st.st_mtime))


def _sha256_file(path: Path, max_bytes: int | None = None) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        if max_bytes is None:
            for chunk in iter(lambda: f.read(1024 * 1024), b''):
                h.update(chunk)
        else:
            remaining = max_bytes
            while remaining > 0:
                chunk = f.read(min(1024 * 1024, remaining))
                if not chunk:
                    break
                h.update(chunk)
                remaining -= len(chunk)
    return h.hexdigest()


def _fingerprint_npy(npy_path: str) -> str:
    """Stable fingerprint for a matrix file: SHA256 over shape+dtype+stats+sampled bytes.

    - Includes numeric stats so it changes even if file header changes.
    - Samples raw file bytes (first 2MB) for extra confidence.
    """
    p = Path(npy_path)
    if not p.exists():
        return "-"
    try:
        S = np.load(str(p))
        stats = (
            str(S.shape),
            str(S.dtype),
            f"mean={float(np.mean(S)):.6f}",
            f"std={float(np.std(S)):.6f}",
            f"min={float(np.min(S)):.6f}",
            f"max={float(np.max(S)):.6f}",
        )
        h = hashlib.sha256()
        h.update("|".join(stats).encode('utf-8'))
        # also mix in a small sample of the raw .npy file bytes
        h.update(_sha256_file(p, max_bytes=2 * 1024 * 1024).encode('utf-8'))
        return h.hexdigest()[:16]
    except Exception:
        # fallback: file hash sample only
        return _sha256_file(p, max_bytes=2 * 1024 * 1024)[:16]


def _default_paths(system: str) -> Dict[str, str]:
    # matrix paths
    paths = {
        # IMPORTANT: do NOT silently fall back here; batch mode should show sem_raw and sem_dade separately.
        'sem_dade': f"data/processed/fusion/{system}_S_sem_dade.npy",
        'sem_raw': f"data/processed/fusion/{system}_S_sem.npy",
        'struct': f"data/processed/fusion/{system}_S_struct.npy",
        'temp': f"data/processed/temporal/{system}_S_temp.npy",
        'final': f"data/processed/fusion/{system}_S_final.npy",
        # embeddings (semantic) - preferred
        'emb_pt': f"data/processed/embedding/{system}.pt",
        'emb_class_json': f"data/processed/embedding/{system}_class_embeddings.json",
        # embeddings (semantic) - legacy
        'emb_pt_legacy': f"data/processed/embedding/{system}_embeddings.pt",
        'emb_class_json_legacy': f"data/processed/embedding/{system}_embeddings_class_embeddings.json",
    }
    paths['gt'] = f"data/processed/groundtruth/{system}_ground_truth.json"
    paths['order'] = f"data/processed/fusion/{system}_class_order.json"
    return paths


def diagnose_system(system_name, s_matrix_path, gt_path, class_order_path, merge_labels: str = ""):
    S = np.load(s_matrix_path)
    labels, N = _load_labels(gt_path, class_order_path)
    labels = _apply_merge_labels(labels, merge_labels)
    assert S.shape[0] == S.shape[1] == N, f"S shape {S.shape} != class_order {N}"

    intra_avg, inter_avg, ratio, n_intra, n_inter = _intra_inter_stats(S, labels)
    suffix = f" | merge={merge_labels}" if merge_labels else ""
    print(f"[{system_name}] 簇内均值: {intra_avg:.4f}, 簇间均值: {inter_avg:.4f}, 对比度: {ratio:.2f} | pairs(intra/inter)={n_intra}/{n_inter}{suffix}")
    return intra_avg, inter_avg, ratio


def batch_diagnose(systems: List[str], verify: bool = False, merge_labels: str = "", plants_intent_matrices: bool = False) -> int:
    rows = []
    verify_rows = []

    for sys_name in systems:
        p = _default_paths(sys_name)
        if not (os.path.exists(p['gt']) and os.path.exists(p['order'])):
            print(f"[WARN] Missing GT or class_order for {sys_name}: {p['gt']} / {p['order']}")
            continue
        labels, N = _load_labels(p['gt'], p['order'])
        labels = _apply_merge_labels(labels, merge_labels)

        mat_keys = ['sem_raw', 'sem_dade', 'struct', 'temp', 'final']

        # NOTE: Previously we optionally reported Plants temp_j1/temp_j2 for evidence chains.
        # The matrices are still written to disk by the temporal builder, but the batch summary
        # table intentionally keeps only the main 'temp' row.
        # (Evidence chain can be produced separately via results/plants_evidence_chain.csv.)

        for mat_key in mat_keys:
            mat_path = p.get(mat_key, "")
            if not mat_path or not os.path.exists(mat_path):
                rows.append((sys_name, mat_key, None, None, None, 0, 0, None, None, None, 0, 0, mat_path))
                continue
            S = np.load(mat_path)
            if S.shape[0] != S.shape[1] or S.shape[0] != N:
                rows.append((sys_name, mat_key, None, None, None, 0, 0, None, None, None, 0, 0, f"shape mismatch {S.shape} vs N={N}"))
                continue
            intra_avg, inter_avg, ratio, n_intra, n_inter = _intra_inter_stats(S, labels)
            intra_nz, inter_nz, nz_ratio, nz_intra, nz_inter = _intra_inter_stats_nz(S, labels)
            rows.append((sys_name, mat_key, intra_avg, inter_avg, ratio, n_intra, n_inter, intra_nz, inter_nz, nz_ratio, nz_intra, nz_inter, mat_path))

        if verify:
            # Matrix fingerprints + embedding file stats to confirm changes
            sem_dade_fp = _fingerprint_npy(p['sem_dade'])
            sem_raw_fp = _fingerprint_npy(p['sem_raw'])
            semdade_exists, semdade_size, semdade_mtime = _try_stat(p['sem_dade'])
            semraw_exists, semraw_size, semraw_mtime = _try_stat(p['sem_raw'])

            pt_exists, pt_size, pt_mtime = _try_stat(p['emb_pt'])
            cls_exists, cls_size, cls_mtime = _try_stat(p['emb_class_json'])

            ptL_exists, ptL_size, ptL_mtime = _try_stat(p['emb_pt_legacy'])
            clsL_exists, clsL_size, clsL_mtime = _try_stat(p['emb_class_json_legacy'])

            verify_rows.append((
                sys_name,
                semdade_exists, semdade_size, semdade_mtime, sem_dade_fp,
                semraw_exists, semraw_size, semraw_mtime, sem_raw_fp,
                pt_exists, pt_size, pt_mtime,
                cls_exists, cls_size, cls_mtime,
                ptL_exists, ptL_size, ptL_mtime,
                clsL_exists, clsL_size, clsL_mtime,
                p['sem_dade'], p['emb_pt']
            ))

    # Pretty print primary table
    print("System     | Matrix    | S_intra | S_inter | Ratio | #intra/#inter | S_intra_nz | S_inter_nz | nz_ratio | #nz_intra/#nz_inter")
    print("-" * 120)
    for (sys_name, mat_key, intra_avg, inter_avg, ratio, n_intra, n_inter, intra_nz, inter_nz, nz_ratio, nz_intra, nz_inter, extra) in rows:
        if intra_avg is None:
            print(f"{sys_name:<10} | {mat_key:<9} | {'NA':>6} | {'NA':>6} | {'NA':>5} | {n_intra}/{n_inter:<11} | {'NA':>10} | {'NA':>10} | {'NA':>7} | {nz_intra}/{nz_inter}  ({extra})")
        else:
            print(
                f"{sys_name:<10} | {mat_key:<9} | {intra_avg:>6.3f} | {inter_avg:>6.3f} | {ratio:>5.2f} | {n_intra}/{n_inter:<11} | "
                f"{intra_nz:>10.3f} | {inter_nz:>10.3f} | {nz_ratio:>7.2f} | {nz_intra}/{nz_inter}"
            )

    if merge_labels:
        print(f"\n[INFO] merge-labels applied: {merge_labels}")

    if verify:
        print("\n[VERIFY] Semantic artifacts (exists/size/mtime) + fingerprints")
        print("System     | sem_dade(exists,size,mtime,fp16) | sem_raw(exists,size,mtime,fp16)")
        print("-----------|----------------------------------|--------------------------------")
        for (
            sys_name,
            semdade_exists, semdade_size, semdade_mtime, semdade_fp,
            semraw_exists, semraw_size, semraw_mtime, semraw_fp,
            pt_exists, pt_size, pt_mtime,
            cls_exists, cls_size, cls_mtime,
            ptL_exists, ptL_size, ptL_mtime,
            clsL_exists, clsL_size, clsL_mtime,
            sem_path, pt_path,
        ) in verify_rows:
            print(
                f"{sys_name:<10} | "
                f"{semdade_exists},{semdade_size},{semdade_mtime},{semdade_fp:<16} | "
                f"{semraw_exists},{semraw_size},{semraw_mtime},{semraw_fp:<16}"
            )

        print("\n[VERIFY] Embedding artifacts (exists/size/mtime)")
        print("System     | preferred.pt(exists,size,mtime) | preferred.class_json(exists,size,mtime)")
        print("-----------|----------------------------------|----------------------------------------")
        for (
            sys_name,
            _sem_exists, _sem_size, _sem_mtime, _sem_fp,
            _semraw_exists, _semraw_size, _semraw_mtime, _semraw_fp,
            pt_exists, pt_size, pt_mtime,
            cls_exists, cls_size, cls_mtime,
            _ptL_exists, _ptL_size, _ptL_mtime,
            _clsL_exists, _clsL_size, _clsL_mtime,
            _sem_path, _pt_path,
        ) in verify_rows:
            print(f"{sys_name:<10} | {pt_exists},{pt_size},{pt_mtime:<10} | {cls_exists},{cls_size},{cls_mtime}")

        print("\n[VERIFY] Embedding artifacts (legacy names) (exists/size/mtime)")
        print("System     | legacy.pt(exists,size,mtime)    | legacy.class_json(exists,size,mtime)")
        print("-----------|----------------------------------|----------------------------------------")
        for (
            sys_name,
            _sem_exists, _sem_size, _sem_mtime, _sem_fp,
            _semraw_exists, _semraw_size, _semraw_mtime, _semraw_fp,
            _pt_exists, _pt_size, _pt_mtime,
            _cls_exists, _cls_size, _cls_mtime,
            ptL_exists, ptL_size, ptL_mtime,
            clsL_exists, clsL_size, clsL_mtime,
            _sem_path, _pt_path,
        ) in verify_rows:
            print(f"{sys_name:<10} | {ptL_exists},{ptL_size},{ptL_mtime:<10} | {clsL_exists},{clsL_size},{clsL_mtime}")

        print("\n[VERIFY] Tips:")
        print("- batch now reports sem_raw and sem_dade as separate rows.")
        print("- build_multimodal_matrices will typically use DADE for fusion when available/up-to-date; this script lets you quantify the gain directly.")

    return 0


if __name__ == "__main__":
    # Minimal argv parsing (keep script dependency-free)
    merge_labels = ""
    plants_intent_matrices = False

    if '--merge-labels' in sys.argv:
        i = sys.argv.index('--merge-labels')
        if i + 1 >= len(sys.argv):
            raise SystemExit("--merge-labels requires a value, e.g. \"1,3->1\"")
        merge_labels = sys.argv[i + 1]
        # remove from argv to keep old positional logic
        del sys.argv[i:i+2]

    if '--plants-intent-matrices' in sys.argv:
        plants_intent_matrices = True
        sys.argv.remove('--plants-intent-matrices')

    if len(sys.argv) >= 2 and sys.argv[1] == '--batch':
        verify = '--verify' in sys.argv[2:]
        if verify:
            sys.argv.remove('--verify')
        raise SystemExit(batch_diagnose(['acmeair', 'daytrader', 'jpetstore', 'plants'], verify=verify, merge_labels=merge_labels, plants_intent_matrices=plants_intent_matrices))

    if len(sys.argv) != 5:
        print("Usage: python diagnose_intra_inter_similarity.py <system> <s_matrix_path> <gt_path> <class_order_path>")
        print("   or: python diagnose_intra_inter_similarity.py --batch [--verify]")
        print("Options:")
        print("  --merge-labels \"1,3->1\"          Merge GT labels in-memory for GT-Intent evaluation")
        print("  --plants-intent-matrices           In batch mode, also report plants temp_j1/temp_j2 if present")
        sys.exit(1)

    diagnose_system(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], merge_labels=merge_labels)
