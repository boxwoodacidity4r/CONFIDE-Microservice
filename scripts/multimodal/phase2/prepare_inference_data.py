import numpy as np
import argparse

# 通用推理特征生成脚本，支持命令行参数指定系统名

def build_inference_pairs(emb, s_struct, s_sem, s_temp):
    n = emb.shape[0]
    s_static = (s_struct + s_sem) / 2
    feat_dim = emb.shape[1] * 2 + 4  # 增加D特征
    X = np.zeros((n * (n - 1) // 2, feat_dim), dtype=np.float32)
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            struct_score = max(s_struct[i, j], s_struct[j, i])
            sim_static = s_static[i, j]
            sim_temp = s_temp[i, j]
            D = abs(sim_static - sim_temp)
            X[idx] = np.hstack([
                emb[i],
                emb[j],
                [struct_score, s_sem[i, j], s_temp[i, j], D]
            ])
            idx += 1
    return X

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('system', type=str, help='系统名，如 acmeair、jpetstore、plants')
    args = parser.parse_args()
    sys = args.system
    emb = np.load(f'data/processed/fusion/{sys}_S_sem_embedding.npy')
    s_struct = np.load(f'data/processed/fusion/{sys}_S_struct.npy')
    s_sem = np.load(f'data/processed/fusion/{sys}_S_sem_dade.npy')
    s_temp = np.load(f'data/processed/temporal/{sys}_S_temp.npy')
    X_infer = build_inference_pairs(emb, s_struct, s_sem, s_temp)
    np.save(f'data/processed/edl/{sys}_all_X.npy', X_infer)
    print(f"Generated {X_infer.shape[0]} pairs for inference: {sys}")

if __name__ == '__main__':
    main()
