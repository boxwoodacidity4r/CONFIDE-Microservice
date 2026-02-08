import numpy as np
import json
import argparse
from collections import Counter
from sklearn.preprocessing import StandardScaler
import joblib

# 数据准备脚本：生成 EDL 训练/测试特征集，支持下采样

def load_ground_truth(gt_path, class_order):
    with open(gt_path, 'r', encoding='utf-8') as f:
        gt_map = json.load(f)
    # 保证顺序与 class_order 一致
    return np.array([gt_map.get(cls, -1) for cls in class_order])

def build_pairs(emb, s_struct, s_sem, s_temp, gt, downsample_ratio=2,
                hard_neg_ratio=0.3, hard_neg_static_th=0.6, hard_neg_temp_th=0.3):
    n = emb.shape[0]
    X, y = [], []
    pos_idx, neg_idx = [], []
    hard_neg_idx = []
    # Embedding 维度检查
    expected_dim = emb.shape[1]
    assert emb.shape[1] == expected_dim, f"Embedding dim mismatch: {emb.shape[1]} vs {expected_dim}"
    # 结构矩阵对称化
    s_struct = s_struct + s_struct.T
    s_static = (s_struct + s_sem) / 2
    for i in range(n):
        for j in range(i+1, n):
            if gt[i] == -1 or gt[j] == -1:
                continue
            label = int(gt[i] == gt[j])
            struct_score = max(s_struct[i, j], s_struct[j, i])
            sim_static = s_static[i, j]
            sim_temp = s_temp[i, j]
            D = abs(sim_static - sim_temp)  # 跨模态冲突特征
            feat = np.hstack([emb[i], emb[j], [struct_score, s_sem[i, j], s_temp[i, j], D]])

            if label == 1:
                pos_idx.append((feat, label))
            else:
                # hard negative：静态很像但时序很不像（人为制造冲突样本）
                if (sim_static >= hard_neg_static_th) and (sim_temp <= hard_neg_temp_th):
                    hard_neg_idx.append((feat, label))
                else:
                    neg_idx.append((feat, label))

    # 下采样负样本：优先保留 hard negatives
    n_pos = len(pos_idx)
    n_hard = min(len(hard_neg_idx), int(n_pos * hard_neg_ratio))
    np.random.shuffle(hard_neg_idx)
    hard_keep = hard_neg_idx[:n_hard]

    # 其余普通负样本
    n_neg_budget = max(0, n_pos * downsample_ratio - n_hard)
    np.random.shuffle(neg_idx)
    neg_keep = neg_idx[: min(len(neg_idx), n_neg_budget)]

    pairs = pos_idx + hard_keep + neg_keep
    np.random.shuffle(pairs)
    X = np.array([f for f, _ in pairs], dtype=np.float32)
    y = np.array([l for _, l in pairs], dtype=np.int64)
    return X, y

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--system', type=str, required=True)
    parser.add_argument('--downsample', type=int, default=2)
    parser.add_argument('--out', type=str, default=None)
    parser.add_argument('--save_scaler', type=str, default=None, help='保存scaler的路径')

    # 【新增】冲突负样本增强（hard negatives）
    parser.add_argument('--hard_neg_ratio', type=float, default=0.3)
    parser.add_argument('--hard_neg_static_th', type=float, default=0.6)
    parser.add_argument('--hard_neg_temp_th', type=float, default=0.3)

    args = parser.parse_args()
    sys = args.system
    # 路径约定
    emb = np.load(f'data/processed/fusion/{sys}_S_sem_embedding.npy')
    s_struct = np.load(f'data/processed/fusion/{sys}_S_struct.npy')
    s_sem = np.load(f'data/processed/fusion/{sys}_S_sem_dade.npy')
    s_temp = np.load(f'data/processed/temporal/{sys}_S_temp.npy')
    with open(f'data/processed/fusion/{sys}_class_order.json', 'r', encoding='utf-8') as f:
        class_order = json.load(f)
    # Ground Truth 路径优先查找 groundtruth 目录
    gt_path_fusion = f'data/processed/fusion/{sys}_ground_truth.json'
    gt_path_groundtruth = f'data/processed/groundtruth/{sys}_ground_truth.json'
    import os
    if os.path.exists(gt_path_groundtruth):
        gt_path = gt_path_groundtruth
    else:
        gt_path = gt_path_fusion
    gt = load_ground_truth(gt_path, class_order)
    # Embedding 维度自动检查
    expected_dim = emb.shape[1]
    print(f"Embedding dim: {expected_dim}")
    X, y = build_pairs(
        emb,
        s_struct,
        s_sem,
        s_temp,
        gt,
        downsample_ratio=args.downsample,
        hard_neg_ratio=args.hard_neg_ratio,
        hard_neg_static_th=args.hard_neg_static_th,
        hard_neg_temp_th=args.hard_neg_temp_th,
    )
    # === 全局标准化 ===
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    out_prefix = args.out or f'data/processed/edl/{sys}'
    np.save(f'{out_prefix}_X.npy', X_scaled)
    np.save(f'{out_prefix}_y.npy', y)
    print(f"Saved: {out_prefix}_X.npy, {out_prefix}_y.npy | pos/neg={Counter(y)[1]}/{Counter(y)[0]}")
    # 保存scaler
    scaler_path = args.save_scaler or f'{out_prefix}_scaler.pkl'
    joblib.dump(scaler, scaler_path)
    print(f"Scaler saved to {scaler_path}")

if __name__ == '__main__':
    main()
