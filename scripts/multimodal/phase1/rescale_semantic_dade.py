import os
import json
import argparse
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

# 修正路径逻辑，向上跳三级到达项目根目录，再进入 data/processed
current_dir = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.abspath(os.path.join(current_dir, '../../../data/processed'))
FUSION_DIR = os.path.join(DATA_ROOT, 'fusion')

SYSTEM_CONFIG = {
    'acmeair': {},
    'daytrader': {},
    'jpetstore': {},
    'plants': {},
}


def safe_normalize(M: np.ndarray) -> np.ndarray:
    M = np.array(M, dtype=np.float32)
    if M.size == 0:
        return M
    minv, maxv = M.min(), M.max()
    if maxv - minv < 1e-9:
        return np.zeros_like(M)
    return (M - minv) / (maxv - minv)


def _apply_topk_sparsify(S: np.ndarray, top_k: int, keep_self: bool = True) -> np.ndarray:
    """Row-wise keep only top-k values (off-diagonal), zero the rest.

    Rationale: DADE aims to build high-contrast 'business fingerprints'. For systems like
    acmeair where many classes remain moderately similar (semantic smoothing), a kNN-style
    sparsification reduces spurious inter edges and amplifies truly strongest links.

    Notes:
    - This is applied AFTER rescaling to [0,1] so top-k is meaningful.
    - Symmetry will be re-enforced later by (S+S.T)/2.
    """
    if top_k <= 0:
        return S

    S = np.array(S, dtype=np.float32)
    n = S.shape[0]
    if n == 0:
        return S

    # Ensure k does not exceed available off-diagonal entries
    k = min(int(top_k), max(0, n - 1))
    if k <= 0:
        return S

    out = np.zeros_like(S, dtype=np.float32)
    for i in range(n):
        row = S[i].copy()
        if keep_self:
            row[i] = -1.0  # exclude diagonal from top-k selection
        # argpartition for efficiency
        idx = np.argpartition(row, -k)[-k:]
        out[i, idx] = S[i, idx]
        if keep_self:
            out[i, i] = S[i, i]
    return out


def dade_rescale(
    S: np.ndarray,
    target_mean: float = 0.5,
    top_k: int = 0,
    keep_self: bool = True,
) -> np.ndarray:
    """A DADE-style rescaling heuristic for semantic similarity matrices.

    Pipeline:
    1) Clip to [0, 1].
    2) Row-wise z-score (excluding diagonal) + sigmoid to enhance contrast.
    3) Global normalize to [0, 1].
    4) Optional mean adjustment towards target_mean.
    5) (NEW) Optional top-k sparsification (business fingerprint distillation).
    6) Enforce symmetry + unit diagonal.

    The top-k step is crucial for systems like acmeair to reduce semantic smoothing.
    """
    S = np.array(S, dtype=np.float32)
    n = S.shape[0]
    if S.shape[0] != S.shape[1]:
        raise ValueError(f"Semantic matrix must be square, got {S.shape}")

    # 1) Clip to [0, 1]
    S = np.clip(S, 0.0, 1.0)

    # 2) Row-wise z-score on off-diagonal entries, then sigmoid
    S_dade = np.zeros_like(S)
    for i in range(n):
        row = S[i].copy()
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        off_diag = row[mask]
        if off_diag.size == 0:
            continue
        mu = off_diag.mean()
        sigma = off_diag.std() + 1e-6
        z = (row - mu) / sigma
        # sigmoid to [0,1]
        row_d = 1.0 / (1.0 + np.exp(-z))
        S_dade[i] = row_d

    # 3) Global normalization to [0, 1]
    S_dade = safe_normalize(S_dade)

    # 4) Adjust global mean towards target_mean (simple linear blending)
    current_mean = float(S_dade.mean()) if S_dade.size > 0 else 0.0
    if current_mean > 1e-6:
        alpha = 0.5  # blend strength
        scale = target_mean / current_mean
        S_adj = S_dade * scale
        S_adj = np.clip(S_adj, 0.0, 1.0)
        S_final = alpha * S_adj + (1 - alpha) * S_dade
        S_final = np.clip(S_final, 0.0, 1.0)
    else:
        S_final = S_dade

    # 5) Optional top-k sparsification
    if top_k and top_k > 0:
        before_nnz = int(np.count_nonzero(S_final))
        S_final = _apply_topk_sparsify(S_final, top_k=top_k, keep_self=keep_self)
        after_nnz = int(np.count_nonzero(S_final))
        logging.info("[DADE] Applied top-k sparsify: k=%d, nnz %d -> %d", int(top_k), before_nnz, after_nnz)

    # enforce symmetry and unit diagonal
    S_sym = 0.5 * (S_final + S_final.T)
    np.fill_diagonal(S_sym, 1.0)
    return S_sym


def process_system(system: str, target_mean: float = 0.5, top_k: int = 0, topk_ratio: float = 0.0, keep_self: bool = True):
    if system not in SYSTEM_CONFIG:
        raise ValueError(f"Unknown system: {system}")

    in_path = os.path.join(FUSION_DIR, f"{system}_S_sem.npy")
    if not os.path.isfile(in_path):
        raise FileNotFoundError(f"Input semantic matrix not found: {in_path}. Please run build_multimodal_matrices.py first.")

    S = np.load(in_path)
    n = int(S.shape[0])

    # resolve k from ratio if provided (takes precedence if >0)
    k = int(top_k)
    if topk_ratio and topk_ratio > 0:
        k = max(1, int(round(float(topk_ratio) * (n - 1))))

    logging.info(
        "[%s] Loaded S_sem from %s (shape=%s, mean=%.4f). DADE params: target_mean=%.3f, top_k=%d (ratio=%.4f)",
        system, in_path, S.shape, float(S.mean()), float(target_mean), int(k), float(topk_ratio)
    )

    S_dade = dade_rescale(S, target_mean=target_mean, top_k=k, keep_self=keep_self)

    out_path = os.path.join(FUSION_DIR, f"{system}_S_sem_dade.npy")
    np.save(out_path, S_dade)
    logging.info("[%s] Saved DADE-rescaled semantic matrix to %s (mean=%.4f)", system, out_path, float(S_dade.mean()))


def main():
    parser = argparse.ArgumentParser(description='DADE-style rescaling for semantic similarity matrices.')
    parser.add_argument('--system', choices=sorted(SYSTEM_CONFIG.keys()), required=True, help='Target system name.')
    parser.add_argument('--target-mean', type=float, default=0.5, help='Target global mean of rescaled S_sem.')
    parser.add_argument('--top-k', type=int, default=0, help='Row-wise keep only top-k semantic links (kNN-style). 0 disables.')
    parser.add_argument('--topk-ratio', type=float, default=0.0, help='Alternative to --top-k: k = round(ratio*(N-1)). Takes precedence if >0.')
    parser.add_argument('--no-keep-self', action='store_true', default=False, help='Do not force keeping diagonal during top-k selection.')
    args = parser.parse_args()

    process_system(
        args.system,
        target_mean=args.target_mean,
        top_k=args.top_k,
        topk_ratio=args.topk_ratio,
        keep_self=(not args.no_keep_self),
    )


if __name__ == '__main__':
    main()
