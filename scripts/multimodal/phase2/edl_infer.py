import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import argparse
import matplotlib.pyplot as plt
import os
import json
import joblib
from sklearn.preprocessing import StandardScaler
import time

def build_all_pairs_features(emb, s_struct, s_sem, s_temp):
    n = emb.shape[0]
    s_static = (s_struct + s_sem) / 2
    all_feats = []
    for i in range(n):
        for j in range(i+1, n):
            struct_score = max(s_struct[i, j], s_struct[j, i])
            sim_static = s_static[i, j]
            sim_temp = s_temp[i, j]
            D = abs(sim_static - sim_temp)  # 跨模态冲突特征
            feat = np.hstack([emb[i], emb[j], [struct_score, s_sem[i, j], s_temp[i, j], D]])
            all_feats.append(feat)
    return np.array(all_feats, dtype=np.float32)

class EDLNet(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 256), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 2)
        )
    def forward(self, x):
        evidence = F.softplus(self.mlp(x))
        alpha = evidence + 1
        return alpha

def infer_uncertainty(model, X, num_classes=2, batch_size=2048, evidence_smooth=0.1, evidence_scale=1.0):
    model.eval()
    u_list = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            batch_x = torch.FloatTensor(X[i : i + batch_size])
            alpha = model(batch_x)

            # 【新增】证据缩放：放大/缩小 (alpha-1)
            if evidence_scale != 1.0:
                evidence = (alpha - 1.0) * evidence_scale
                alpha = evidence + 1.0

            # 证据平滑
            alpha = alpha + evidence_smooth
            S = torch.sum(alpha, dim=1, keepdim=True)
            u = num_classes / S
            u_list.append(u.squeeze().cpu().numpy())
    return np.concatenate(u_list)

def debug_u_matrix(u_matrix, system_name):
    import matplotlib.pyplot as plt
    plt.hist(u_matrix.flatten(), bins=50)
    plt.title(f"Uncertainty Distribution for {system_name}")
    plt.xlabel("u value")
    plt.ylabel("Frequency")
    plt.show()
    print(f"Mean U: {np.mean(u_matrix):.4f}, Std U: {np.std(u_matrix):.4f}")

def _normalize_u_matrix(U: np.ndarray, mode: str, *, q_low: float = 0.05, q_high: float = 0.95):
    """Normalize U into [0,1] with robust options.

    - none   : keep as-is
    - minmax : (U-min)/(max-min)
    - robust : clip to [q_low,q_high] quantiles then minmax
    - quantile : map to empirical CDF (rank-based in [0,1])

    Returns: (U_norm, meta_dict)
    """

    mode = (mode or "none").lower().strip()
    meta = {"mode": mode}

    if mode == "none":
        meta.update({"u_min": float(U.min()), "u_max": float(U.max())})
        return U, meta

    if mode == "minmax":
        u_min, u_max = float(U.min()), float(U.max())
        meta.update({"u_min": u_min, "u_max": u_max})
        if u_max > u_min:
            return (U - u_min) / (u_max - u_min), meta
        return U, meta

    if mode == "robust":
        lo = float(np.quantile(U, q_low))
        hi = float(np.quantile(U, q_high))
        meta.update({"q_low": float(q_low), "q_high": float(q_high), "clip_lo": lo, "clip_hi": hi})
        Uc = np.clip(U, lo, hi)
        u_min, u_max = float(Uc.min()), float(Uc.max())
        meta.update({"u_min": u_min, "u_max": u_max})
        if u_max > u_min:
            return (Uc - u_min) / (u_max - u_min), meta
        return Uc, meta

    if mode == "quantile":
        flat = U.flatten()
        order = np.argsort(flat)
        ranks = np.empty_like(order, dtype=np.float64)
        ranks[order] = np.linspace(0.0, 1.0, num=len(flat), endpoint=True)
        Uq = ranks.reshape(U.shape)
        meta.update({"u_min": float(Uq.min()), "u_max": float(Uq.max())})
        return Uq, meta

    raise ValueError(f"Unknown --u_norm mode: {mode}. Expected one of: none|minmax|robust|quantile")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--X', type=str, default=None, help='拼接好的特征npy，优先使用')
    parser.add_argument('--emb', type=str, default=None)
    parser.add_argument('--s_struct', type=str, default=None)
    parser.add_argument('--s_sem', type=str, default=None)
    parser.add_argument('--s_temp', type=str, default=None)
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--class_order', type=str, required=True)
    parser.add_argument('--out', type=str, default=None)
    parser.add_argument('--scaler', type=str, default=None, help='scaler.pkl 路径')

    # 【新增】实验保存：run 目录与 tag（与 edl_train.py 对齐）
    parser.add_argument('--run_tag', type=str, default=None, help='If set, outputs are saved under data/processed/edl/runs/<run_tag>/infer_outputs/')
    parser.add_argument('--run_dir', type=str, default=None, help='Override run directory. If set, infer outputs are placed under <run_dir>/infer_outputs/')
    parser.add_argument('--system', type=str, default=None, help='Optional system name for naming outputs (e.g., acmeair)')
    parser.add_argument('--note', type=str, default='', help='Optional free-text note saved into infer_config.json')

    # 【新增】不确定性释放相关参数
    parser.add_argument('--evidence_smooth', type=float, default=0.1, help='Additive smoothing to alpha before computing U')
    parser.add_argument('--evidence_scale', type=float, default=1.0, help='Scale evidence (alpha-1) before computing U')

    # New: normalize U for better cross-system comparability
    parser.add_argument('--u_norm', type=str, default='none', choices=['none', 'minmax', 'robust', 'quantile'],
                        help='Normalize final U matrix (recommended: robust or quantile)')
    parser.add_argument('--u_q_low', type=float, default=0.05, help='Lower quantile for robust U normalization')
    parser.add_argument('--u_q_high', type=float, default=0.95, help='Upper quantile for robust U normalization')

    # --- Visualization controls (paper-friendly defaults) ---
    parser.add_argument('--viz_vmax', type=float, default=0.3,
                        help='Heatmap vmax for better contrast (set <=0 to disable clipping)')
    parser.add_argument('--viz_set_diag_black', action='store_true', default=True,
                        help='Render diagonal as black to avoid a dark stripe dominating contrast')
    parser.add_argument('--viz_cmap', type=str, default='seismic',
                        help='Matplotlib colormap for heatmap (e.g., seismic, bwr, RdBu_r)')

    args = parser.parse_args()

    # --- new: resolve infer output directory ---
    run_dir = None
    if args.run_dir:
        run_dir = args.run_dir
    elif args.run_tag:
        run_dir = os.path.join('data', 'processed', 'edl', 'runs', args.run_tag)

    infer_dir = None
    if run_dir:
        infer_dir = os.path.join(run_dir, 'infer_outputs')
        os.makedirs(infer_dir, exist_ok=True)

    with open(args.class_order, 'r', encoding='utf-8') as f:
        class_order = json.load(f)
    N = len(class_order)

    # 优先使用--X，否则回退到四模态输入
    if args.X is not None:
        X = np.load(args.X)
    else:
        if not (args.emb and args.s_struct and args.s_sem and args.s_temp):
            raise ValueError('必须提供--X，或同时提供--emb、--s_struct、--s_sem、--s_temp')
        emb = np.load(args.emb)
        s_struct = np.load(args.s_struct)
        s_sem = np.load(args.s_sem)
        s_temp = np.load(args.s_temp)
        X = build_all_pairs_features(emb, s_struct, s_sem, s_temp)

    # === 加载scaler并标准化 ===
    scaler_path = args.scaler or args.model.replace('edl_model', 'edl_scaler').replace('.pt', '_scaler.pkl')
    scaler = joblib.load(scaler_path)
    X = scaler.transform(X)

    input_dim = X.shape[1]
    model_inst = EDLNet(input_dim)
    model_inst.load_state_dict(torch.load(args.model, map_location='cpu'))

    u = infer_uncertainty(model_inst, X, evidence_smooth=args.evidence_smooth, evidence_scale=args.evidence_scale)

    # 将 u 还原为 N*N 矩阵（只填上三角）
    U = np.zeros((N, N))
    idx = 0
    for i in range(N):
        for j in range(i+1, N):
            U[i, j] = u[idx]
            U[j, i] = u[idx]
            idx += 1

    # Normalize U if requested
    U_norm, u_norm_meta = _normalize_u_matrix(U, args.u_norm, q_low=args.u_q_low, q_high=args.u_q_high)
    U = U_norm

    # --- new: default output naming ---
    out_path = args.out
    if out_path is None:
        sys_name = args.system or os.path.basename(args.class_order).replace('_class_order.json', '')
        if infer_dir:
            out_path = os.path.join(infer_dir, f'{sys_name}_edl_uncertainty.npy')
        else:
            out_path = 'edl_uncertainty.npy'

    np.save(out_path, U)
    print(f"Uncertainty matrix saved to {out_path}")

    # --- Visualization (improved) ---
    plt.figure(figsize=(8, 6))

    # Copy for visualization so we don't change saved numeric matrix
    U_viz = U.copy()
    if args.viz_set_diag_black:
        # NaN will be rendered with cmap.set_bad('black')
        np.fill_diagonal(U_viz, np.nan)

    cmap = plt.get_cmap(args.viz_cmap).copy()
    if args.viz_set_diag_black:
        cmap.set_bad(color='black')

    vmax = args.viz_vmax if (args.viz_vmax is not None and args.viz_vmax > 0) else None

    im = plt.imshow(U_viz, cmap=cmap, interpolation='nearest', vmin=0.0, vmax=vmax)
    plt.colorbar(im, label='Uncertainty (u)')
    plt.title('EDL Uncertainty Matrix')
    plt.xlabel('Class Index')
    plt.ylabel('Class Index')
    plt.tight_layout()
    plt.savefig(out_path.replace('.npy', '.png'), dpi=300)
    plt.close()
    print(f"Heatmap saved to {out_path.replace('.npy', '.png')}")

    # === 校准层：输出 u 分布直方图、均值、方差 ===
    def debug_u_matrix(u_matrix, system_name):
        plt.figure(figsize=(6,4))
        plt.hist(u_matrix.flatten(), bins=50, color='skyblue', edgecolor='black')
        plt.title(f"Uncertainty Distribution for {system_name}")
        plt.xlabel('u value')
        plt.ylabel('Frequency')
        plt.tight_layout()
        plt.savefig(out_path.replace('.npy', '_u_hist.png'), dpi=300)
        plt.close()
        print(f"[DEBUG] Mean U: {np.mean(u_matrix):.4f}, Std U: {np.std(u_matrix):.4f}")
        print(f"u 分布直方图已保存: {out_path.replace('.npy', '_u_hist.png')}")

    debug_u_matrix(U, os.path.basename(out_path).replace('_edl_uncertainty.npy', ''))

    # --- new: write infer config snapshot ---
    if run_dir:
        infer_cfg = {
            'created_at': time.strftime('%Y-%m-%d %H:%M:%S'),
            'run_tag': args.run_tag,
            'run_dir': os.path.abspath(run_dir),
            'inputs': {
                'X': os.path.abspath(args.X) if args.X else None,
                'emb': os.path.abspath(args.emb) if args.emb else None,
                's_struct': os.path.abspath(args.s_struct) if args.s_struct else None,
                's_sem': os.path.abspath(args.s_sem) if args.s_sem else None,
                's_temp': os.path.abspath(args.s_temp) if args.s_temp else None,
                'class_order': os.path.abspath(args.class_order),
                'model': os.path.abspath(args.model),
                'scaler': os.path.abspath(scaler_path),
            },
            'params': {
                'evidence_smooth': float(args.evidence_smooth),
                'evidence_scale': float(args.evidence_scale),
                'system': args.system,
                'u_norm': str(args.u_norm),
                'u_q_low': float(args.u_q_low),
                'u_q_high': float(args.u_q_high),
                'u_norm_meta': u_norm_meta,
            },
            'outputs': {
                'uncertainty_npy': os.path.abspath(out_path),
                'heatmap_png': os.path.abspath(out_path.replace('.npy', '.png')),
                'u_hist_png': os.path.abspath(out_path.replace('.npy', '_u_hist.png')),
            },
            'note': args.note,
        }
        with open(os.path.join(infer_dir, 'infer_config.json'), 'w', encoding='utf-8') as f:
            json.dump(infer_cfg, f, indent=2, ensure_ascii=False)

if __name__ == '__main__':
    main()
