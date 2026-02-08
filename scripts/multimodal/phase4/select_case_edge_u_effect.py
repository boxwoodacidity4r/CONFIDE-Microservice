"""Select a killer-case edge for U ablation and produce a JSON summary.

Goal: automatically find an edge (i,j) that best demonstrates:
- U(i,j) is high (semantic-topology conflict)
- no-U model places i and j in the same cluster (erroneous merge)
- with-U model separates them (correct split)
- and the uncertainty penalty produces a large drop in final edge weight:
    W_noU(i,j) - W_withU(i,j) is large

We approximate W_noU = S(i,j) (when U≡0) and
             W_withU = S(i,j) * (1 - sigmoid(alpha*(U-beta))) for sigmoid mode.

Outputs:
- results/ablation/case_edge_selected_<system>_<ts>.json

This script does not rerun Phase3; it consumes existing partitions and matrices.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np


ROOT = Path(__file__).resolve().parents[3]


def _load_json(p: Path):
    return json.loads(p.read_text(encoding="utf-8"))


def _sigmoid(z: float) -> float:
    return 1.0 / (1.0 + np.exp(-z))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("system", type=str)
    ap.add_argument("--alpha", type=float, default=15.0)
    ap.add_argument("--beta", type=float, default=None, help="If omitted, use median(U).")
    ap.add_argument("--topk_u", type=int, default=200, help="Search among top-k uncertain edges for efficiency.")
    ap.add_argument(
        "--with_u_partition",
        type=str,
        default=None,
        help="Path to with-U partition JSON (default: fusion/<system>_cac-final_partition.json)",
    )
    ap.add_argument(
        "--no_u_partition",
        type=str,
        default=None,
        help="Path to no-U partition JSON (default: fusion/<system>_cac-final_no_u_partition.json)",
    )
    args = ap.parse_args()

    system = args.system.lower().strip()

    S = np.load(str(ROOT / "data" / "processed" / "fusion" / f"{system}_S_final.npy"))
    U = np.load(str(ROOT / "data" / "processed" / "edl" / f"{system}_edl_uncertainty.npy"))
    umin, umax = float(U.min()), float(U.max())
    U = (U - umin) / (umax - umin) if umax > umin else U

    alpha = float(args.alpha)
    beta = float(np.median(U)) if args.beta is None else float(args.beta)

    with_u_path = Path(args.with_u_partition) if args.with_u_partition else (ROOT / "data" / "processed" / "fusion" / f"{system}_cac-final_partition.json")
    no_u_path = Path(args.no_u_partition) if args.no_u_partition else (ROOT / "data" / "processed" / "fusion" / f"{system}_cac-final_no_u_partition.json")

    if not with_u_path.exists():
        raise FileNotFoundError(f"with-U partition not found: {with_u_path}")
    if not no_u_path.exists():
        raise FileNotFoundError(
            f"no-U partition not found: {no_u_path}. Run Phase3 with --u_ablation no_u and then copy to this path."
        )

    with_u_part_raw = _load_json(with_u_path)
    no_u_part_raw = _load_json(no_u_path)

    # canonicalize to index-keyed
    class_order = _load_json(ROOT / "data" / "processed" / "fusion" / f"{system}_class_order.json")
    idx = {str(n).replace(".java", "").strip(): i for i, n in enumerate(class_order)}

    def to_idx_map(d: Dict[str, Any]) -> Dict[int, int]:
        if all(str(k).isdigit() for k in d.keys()):
            return {int(k): int(v) for k, v in d.items()}
        out = {}
        for k, v in d.items():
            key = str(k).replace(".java", "").strip()
            if key in idx:
                out[int(idx[key])] = int(v)
        return out

    with_u = to_idx_map(with_u_part_raw)
    no_u = to_idx_map(no_u_part_raw)

    n = S.shape[0]
    iu = np.triu_indices(n, k=1)
    uvals = U[iu]
    # top-k by uncertainty
    topk = int(min(args.topk_u, len(uvals)))
    cand_idx = np.argsort(uvals)[::-1][:topk]

    best = None  # (score, i, j, payload)

    for k in cand_idx:
        i = int(iu[0][k])
        j = int(iu[1][k])
        if i not in with_u or j not in with_u or i not in no_u or j not in no_u:
            continue

        # must be: no-U merges, with-U splits
        if no_u[i] != no_u[j]:
            continue
        if with_u[i] == with_u[j]:
            continue

        s = float(S[i, j])
        u = float(U[i, j])
        w_no = s
        w_with = s * (1.0 - _sigmoid(alpha * (u - beta)))
        drop = w_no - w_with

        # score: strong semantic edge + high uncertainty + big drop
        score = (drop * 2.0) + (s * 1.0) + (u * 0.5)

        payload = {
            "i": i,
            "j": j,
            "class_i": str(class_order[i]),
            "class_j": str(class_order[j]),
            "S": s,
            "U": u,
            "alpha": alpha,
            "beta": beta,
            "W_noU": w_no,
            "W_withU": w_with,
            "drop": drop,
            "noU_same_cluster": True,
            "withU_same_cluster": False,
            "noU_cluster": int(no_u[i]),
            "withU_cluster_i": int(with_u[i]),
            "withU_cluster_j": int(with_u[j]),
            "score": float(score),
        }

        if best is None or score > best[0]:
            best = (score, i, j, payload)

    if best is None:
        raise RuntimeError("No edge found that satisfies: noU merges AND withU splits within top uncertain edges. Try increasing --topk_u")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = ROOT / "results" / "ablation"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"case_edge_selected_{system}_{ts}.json"
    out_path.write_text(json.dumps(best[3], indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"[OK] selected edge saved: {out_path}")
    print(json.dumps(best[3], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
