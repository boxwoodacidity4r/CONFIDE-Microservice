"""Phase 4: Boundary diagnostics (cross-service risky pairs).

Moved from `scripts/multimodal/phase3/diagnose_boundaries.py`.

Goal:
  Diagnose "boundary conflicts" where ground-truth says two classes belong to different services
  but the final similarity S is high and EDL uncertainty U is low.

Defaults assume:
  - Ground truth:        data/processed/groundtruth/{system}_ground_truth.json
  - Class order (index): data/processed/fusion/{system}_class_order.json
  - Uncertainty matrix:  data/processed/edl/{system}_edl_uncertainty.npy
  - Final similarity:    data/processed/fusion/{system}_S_final.npy

Usage:
  python scripts/multimodal/phase4/diagnose_boundaries.py
  python scripts/multimodal/phase4/diagnose_boundaries.py --system daytrader --s-th 0.5 --u-th 0.3 --topn 20
"""

import argparse
import json

import numpy as np


def diagnose(
    system: str,
    s_th: float = 0.5,
    u_th: float = 0.3,
    topn: int = 10,
):
    gt_path = f"data/processed/groundtruth/{system}_ground_truth.json"
    node_names_path = f"data/processed/fusion/{system}_class_order.json"
    u_matrix_path = f"data/processed/edl/{system}_edl_uncertainty.npy"
    s_matrix_path = f"data/processed/fusion/{system}_S_final.npy"

    with open(gt_path, "r", encoding="utf-8") as f:
        gt = json.load(f)
    with open(node_names_path, "r", encoding="utf-8") as f:
        node_names = [n.replace(".java", "") for n in json.load(f)]

    u = np.load(u_matrix_path)
    s = np.load(s_matrix_path)

    print(f"\n--- {system} 边界冲突诊断 ---")

    conflict_report = []
    for i, name_i in enumerate(node_names):
        for j in range(i + 1, len(node_names)):
            name_j = node_names[j]
            if name_i not in gt or name_j not in gt:
                continue

            if gt[name_i] != gt[name_j]:
                conflict_report.append(
                    {
                        "pair": (name_i, name_j),
                        "S": float(s[i, j]),
                        "U": float(u[i, j]),
                        "gt_diff": f"S{gt[name_i]} vs S{gt[name_j]}",
                    }
                )

    risky_pairs = [p for p in conflict_report if p["S"] > s_th and p["U"] < u_th]
    risky_pairs.sort(key=lambda x: (-(x["S"] - s_th), x["U"]))

    print(
        f"检测到 {len(risky_pairs)} 对高风险‘伪耦合’类（GT要求分开，但 S>{s_th} 且 U<{u_th}）"
    )

    for p in risky_pairs[:topn]:
        print(f"Pair: {p['pair']} | S: {p['S']:.4f} | U: {p['U']:.4f} | {p['gt_diff']}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--system",
        default="all",
        help="System name or 'all' (default).",
    )
    parser.add_argument("--s-th", type=float, default=0.5, help="Similarity risk threshold")
    parser.add_argument("--u-th", type=float, default=0.3, help="Uncertainty risk threshold")
    parser.add_argument("--topn", type=int, default=10, help="Print top-N risky pairs")
    args = parser.parse_args()

    systems = ["acmeair", "daytrader", "jpetstore", "plants"] if args.system == "all" else [args.system]
    for sysname in systems:
        diagnose(sysname, s_th=args.s_th, u_th=args.u_th, topn=args.topn)


if __name__ == "__main__":
    main()
