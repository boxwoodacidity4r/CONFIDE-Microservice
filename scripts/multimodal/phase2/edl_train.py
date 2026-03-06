import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import argparse
import os
# --- new ---
import json
import time
import shutil
import platform

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

def kl_divergence(alpha, num_classes=2):
    ones = torch.ones_like(alpha)
    S_alpha = torch.sum(alpha, dim=1, keepdim=True)
    first = torch.lgamma(S_alpha) - torch.sum(torch.lgamma(alpha), dim=1, keepdim=True)
    second = torch.sum((alpha - ones) * (torch.digamma(alpha) - torch.digamma(S_alpha)), dim=1, keepdim=True)
    return torch.mean(first + second)

def edl_loss(alpha, target, epoch_num, num_classes=2, annealing_step=10):
    y = F.one_hot(target, num_classes).float()
    S = torch.sum(alpha, dim=1, keepdim=True)
    probs = alpha / S

    # --- MSE + variance ---
    error_loss = torch.sum((y - probs) ** 2, dim=1, keepdim=True)
    var_loss = torch.sum(probs * (1 - probs) / (S + 1), dim=1, keepdim=True)
    mse_loss = torch.mean(error_loss + var_loss)

    # --- KL annealing (more aggressive) ---
    annealing_coef = min(1.0, epoch_num / annealing_step)
    alp_hat = y + (1 - y) * alpha
    kl_loss = annealing_coef * kl_divergence(alp_hat, num_classes)

    return mse_loss + kl_loss


def edl_loss_scaled(alpha, target, epoch_num, num_classes=2, annealing_step=10, kl_weight=1.0, evidence_scale=1.0):
    """A more aggressive EDL loss:
    - evidence_scale: scales evidence (equivalent to scaling alpha-1), making uncertainty easier to express
    - kl_weight: increases KL regularization weight to penalize overly concentrated evidence distributions
    """
    # Note: alpha >= 1, evidence = alpha - 1
    if evidence_scale != 1.0:
        evidence = (alpha - 1.0) * evidence_scale
        alpha = evidence + 1.0

    y = F.one_hot(target, num_classes).float()
    S = torch.sum(alpha, dim=1, keepdim=True)
    probs = alpha / S

    error_loss = torch.sum((y - probs) ** 2, dim=1, keepdim=True)
    var_loss = torch.sum(probs * (1 - probs) / (S + 1), dim=1, keepdim=True)
    mse_loss = torch.mean(error_loss + var_loss)

    annealing_coef = min(1.0, epoch_num / annealing_step)
    alp_hat = y + (1 - y) * alpha
    kl_loss = annealing_coef * kl_divergence(alp_hat, num_classes)

    return mse_loss + kl_weight * kl_loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_X', type=str, required=True)
    parser.add_argument('--train_y', type=str, required=True)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch', type=int, default=256)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--save', type=str, default='edl_model.pt')

    # Experiment tracking: run directory and tag
    parser.add_argument('--run_tag', type=str, default=None, help='Optional run tag. If set, outputs are saved under data/processed/edl/runs/<run_tag>/')
    parser.add_argument('--run_dir', type=str, default=None, help='Override run directory. If set, --save is relative to this directory unless absolute.')
    parser.add_argument('--note', type=str, default='', help='Optional free-text note saved into train_config.json')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--scaler', type=str, default=None, help='Path to scaler.pkl used in training (optional, for provenance only)')

    # Tuning knobs: evidence scaling + KL weight + KL annealing steps
    parser.add_argument('--evidence_scale', type=float, default=1.0, help='Scale evidence (alpha-1) to reduce overconfidence')
    parser.add_argument('--kl_weight', type=float, default=1.0, help='Weight for KL regularization')
    parser.add_argument('--annealing_step', type=int, default=10, help='KL annealing step (smaller => more aggressive earlier)')
    # Hard negative parameters
    parser.add_argument('--hard_neg_ratio', type=float, default=0.0, help='Ratio of hard negative samples in each batch (0~1, 0=disabled)')

    args = parser.parse_args()

    # --- new: seeds ---
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # --- new: resolve run directory and save path ---
    run_dir = None
    if args.run_dir:
        run_dir = args.run_dir
    elif args.run_tag:
        run_dir = os.path.join('data', 'processed', 'edl', 'runs', args.run_tag)

    if run_dir:
        os.makedirs(run_dir, exist_ok=True)
        ckpt_dir = os.path.join(run_dir, 'checkpoints')
        os.makedirs(ckpt_dir, exist_ok=True)
        # if --save is relative, place it under checkpoints/
        if not os.path.isabs(args.save):
            args.save = os.path.join(ckpt_dir, args.save)

    # --- new: load data ---
    X = np.load(args.train_X)
    y = np.load(args.train_y)
    input_dim = X.shape[1]
    dataset = TensorDataset(torch.FloatTensor(X), torch.LongTensor(y))
    loader = DataLoader(dataset, batch_size=args.batch, shuffle=True)
    model = EDLNet(input_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # --- new: write train config early (so even failed runs are traceable) ---
    if run_dir:
        config = {
            'run_tag': args.run_tag,
            'run_dir': os.path.abspath(run_dir),
            'created_at': time.strftime('%Y-%m-%d %H:%M:%S'),
            'platform': {
                'python': platform.python_version(),
                'os': platform.platform(),
            },
            'data': {
                'train_X': os.path.abspath(args.train_X),
                'train_y': os.path.abspath(args.train_y),
                'num_samples': int(len(y)),
                'input_dim': int(input_dim),
                'label_counts': {str(int(k)): int(v) for k, v in dict(zip(*np.unique(y, return_counts=True))).items()},
                'scaler': os.path.abspath(args.scaler) if args.scaler else None,
            },
            'hyperparams': {
                'epochs': int(args.epochs),
                'batch': int(args.batch),
                'lr': float(args.lr),
                'evidence_scale': float(args.evidence_scale),
                'kl_weight': float(args.kl_weight),
                'annealing_step': int(args.annealing_step),
                'hard_neg_ratio': float(args.hard_neg_ratio),
                'seed': int(args.seed),
            },
            'outputs': {
                'model_path': os.path.abspath(args.save),
            },
            'note': args.note,
        }
        with open(os.path.join(run_dir, 'train_config.json'), 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)

        # best-effort: snapshot the train script itself for provenance
        try:
            shutil.copy2(__file__, os.path.join(run_dir, 'edl_train.snapshot.py'))
        except Exception:
            pass

    # ...existing training loop...
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0
        total_hardneg_loss = 0
        total_hardneg_count = 0
        for xb, yb in loader:
            alpha = model(xb)
            # Hard negative augmentation logic
            if args.hard_neg_ratio > 0.0:
                # Find hard negatives within the batch (label=0)
                hardneg_mask = (yb == 0)
                hardneg_count = hardneg_mask.sum().item()
                hardneg_loss = 0.0
                if hardneg_count > 0:
                    hardneg_alpha = alpha[hardneg_mask]
                    hardneg_yb = yb[hardneg_mask]
                    hardneg_loss = edl_loss_scaled(
                        hardneg_alpha,
                        hardneg_yb,
                        epoch,
                        annealing_step=args.annealing_step,
                        kl_weight=args.kl_weight,
                        evidence_scale=args.evidence_scale,
                    )
                # Normal loss
                normal_loss = edl_loss_scaled(
                    alpha,
                    yb,
                    epoch,
                    annealing_step=args.annealing_step,
                    kl_weight=args.kl_weight,
                    evidence_scale=args.evidence_scale,
                )
                # Total loss = normal_loss + hard_neg_ratio * hardneg_loss
                loss = normal_loss + args.hard_neg_ratio * hardneg_loss
                total_hardneg_loss += hardneg_loss * hardneg_count
                total_hardneg_count += hardneg_count
            else:
                loss = edl_loss_scaled(
                    alpha,
                    yb,
                    epoch,
                    annealing_step=args.annealing_step,
                    kl_weight=args.kl_weight,
                    evidence_scale=args.evidence_scale,
                )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * xb.size(0)
        if args.hard_neg_ratio > 0.0 and total_hardneg_count > 0:
            avg_hardneg_loss = total_hardneg_loss / total_hardneg_count
            print(f"Epoch {epoch}: loss={total_loss / len(dataset):.4f} | hardneg_loss={avg_hardneg_loss:.4f} (count={total_hardneg_count})")
        else:
            print(f"Epoch {epoch}: loss={total_loss / len(dataset):.4f}")

    torch.save(model.state_dict(), args.save)
    print(f"Model saved to {args.save}")

    # --- new: save small metadata next to checkpoint ---
    if run_dir:
        meta = {
            'checkpoint': os.path.abspath(args.save),
            'final_epoch': int(args.epochs),
        }
        with open(os.path.join(run_dir, 'checkpoints', 'checkpoint_meta.json'), 'w', encoding='utf-8') as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

if __name__ == '__main__':
    main()
