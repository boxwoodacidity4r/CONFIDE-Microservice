"""Make a 4-panel (2x2) case-study figure for U effect across all systems.

For each system:
1) Ensure we have with-U and no-U partitions:
   - with-U: data/processed/fusion/<system>_cac-final_partition.json
   - no-U:  data/processed/fusion/<system>_cac-final_no_u_partition.json
2) Auto-pick a killer edge using select_case_edge_u_effect.py
3) Render a subplot using the same style as plot_case_study_u_effect.py:
   - KNN ego graph around (i,j)
   - Node colors = cluster ids
   - Edge widths = S(i,j)
   - Annotation: Without U: W=S (Merged/Separated) | With U: W=S*exp(-U) (...)

Output:
- results/plots/case_study_u_effect_4panel_<ts>.png

Usage:
  python scripts/multimodal/phase4/make_four_panel_case_study.py
  python scripts/multimodal/phase4/make_four_panel_case_study.py --systems acmeair,daytrader,plants,jpetstore

Note:
- This script does not re-run Phase3; it consumes existing matrices/partitions.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


ROOT = Path(__file__).resolve().parents[3]


def _load_json(p: Path):
    return json.loads(p.read_text(encoding="utf-8"))


def _sigmoid(z: float) -> float:
    return 1.0 / (1.0 + np.exp(-z))


def _to_index_labels(part: Dict[str, int], class_order: list[str]) -> Dict[int, int]:
    if all(str(k).isdigit() for k in part.keys()):
        return {int(k): int(v) for k, v in part.items()}
    idx = {str(n).replace(".java", "").strip(): i for i, n in enumerate(class_order)}
    out = {}
    for k, v in part.items():
        kk = str(k).replace(".java", "").strip()
        if kk in idx:
            out[int(idx[kk])] = int(v)
    return out


def _build_knn_graph(S: np.ndarray, k: int) -> nx.Graph:
    n = S.shape[0]
    G = nx.Graph()
    for i in range(n):
        G.add_node(i)
        idx = np.argsort(S[i])[::-1]
        cnt = 0
        for j in idx:
            if i == j:
                continue
            w = float(S[i, j])
            if w <= 0:
                continue
            G.add_edge(i, int(j), weight=w)
            cnt += 1
            if cnt >= k:
                break
    return G


def _panel(ax, G: nx.Graph, labels: Dict[int, int], focus: Tuple[int, int], title: str):
    i, j = focus
    nodes = set([i, j])
    nodes |= set(G.neighbors(i))
    nodes |= set(G.neighbors(j))
    H = G.subgraph(nodes).copy()

    pos = nx.spring_layout(H, seed=42, k=0.6)

    clus = [labels.get(n, -1) for n in H.nodes()]
    nx.draw_networkx_nodes(H, pos, node_color=clus, cmap=plt.cm.tab20, node_size=220, ax=ax, linewidths=0.5, edgecolors="#333")

    widths = [0.4 + 4.5 * float(d.get("weight", 0.0)) for _u, _v, d in H.edges(data=True)]
    nx.draw_networkx_edges(H, pos, width=widths, alpha=0.30, ax=ax)

    if H.has_edge(i, j):
        nx.draw_networkx_edges(H, pos, edgelist=[(i, j)], width=3.5, edge_color="#d62728", alpha=0.95, ax=ax)

    nx.draw_networkx_labels(H, pos, labels={i: str(i), j: str(j)}, font_size=9, font_color="#111", ax=ax)

    ax.set_title(title, fontsize=11)
    ax.axis("off")


def _select_killer_edge(system: str, *, alpha: float, topk_u: int) -> dict:
    """Inline selection (same logic as select_case_edge_u_effect.py) to avoid subprocess."""
    S = np.load(str(ROOT / "data" / "processed" / "fusion" / f"{system}_S_final.npy"))
    U = np.load(str(ROOT / "data" / "processed" / "edl" / f"{system}_edl_uncertainty.npy"))
    umin, umax = float(U.min()), float(U.max())
    if umax > umin:
        U = (U - umin) / (umax - umin)

    class_order = _load_json(ROOT / "data" / "processed" / "fusion" / f"{system}_class_order.json")
    idx = {str(n).replace(".java", "").strip(): i for i, n in enumerate(class_order)}

    with_u_path = ROOT / "data" / "processed" / "fusion" / f"{system}_cac-final_partition.json"
    no_u_path = ROOT / "data" / "processed" / "fusion" / f"{system}_cac-final_no_u_partition.json"
    if not with_u_path.exists() or not no_u_path.exists():
        raise FileNotFoundError(f"Missing partitions for {system}: {with_u_path} or {no_u_path}")

    with_u_raw = _load_json(with_u_path)
    no_u_raw = _load_json(no_u_path)

    def to_idx_map(d: Dict[str, int]) -> Dict[int, int]:
        if all(str(k).isdigit() for k in d.keys()):
            return {int(k): int(v) for k, v in d.items()}
        out = {}
        for k, v in d.items():
            key = str(k).replace(".java", "").strip()
            if key in idx:
                out[int(idx[key])] = int(v)
        return out

    with_u = to_idx_map(with_u_raw)
    no_u = to_idx_map(no_u_raw)

    n = S.shape[0]
    iu = np.triu_indices(n, k=1)
    uvals = U[iu]
    cand = np.argsort(uvals)[::-1][: int(min(topk_u, len(uvals)))]

    beta = float(np.median(U))
    best = None
    for k in cand:
        i = int(iu[0][k])
        j = int(iu[1][k])
        if i not in with_u or j not in with_u or i not in no_u or j not in no_u:
            continue
        if no_u[i] != no_u[j]:
            continue
        if with_u[i] == with_u[j]:
            continue

        s = float(S[i, j])
        u = float(U[i, j])
        w_no = s
        w_with_sigmoid = s * (1.0 - _sigmoid(alpha * (u - beta)))
        drop = w_no - w_with_sigmoid
        score = (drop * 2.0) + (s * 1.0) + (u * 0.5)

        payload = {
            "system": system,
            "i": i,
            "j": j,
            "class_i": str(class_order[i]),
            "class_j": str(class_order[j]),
            "S": s,
            "U": u,
            "alpha": float(alpha),
            "beta": float(beta),
            "W_noU": w_no,
            "W_withU_sigmoid": float(w_with_sigmoid),
            "drop": float(drop),
            "score": float(score),
        }
        if best is None or score > best["score"]:
            best = payload

    if best is None:
        raise RuntimeError(f"No killer edge found for {system}. Try increasing --topk_u")
    return best


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--systems", type=str, default="acmeair,daytrader,plants,jpetstore")
    ap.add_argument("--knn", type=int, default=8)
    ap.add_argument("--alpha", type=float, default=15.0, help="For killer-edge selection scoring (sigmoid mode).")
    ap.add_argument("--topk_u", type=int, default=400)
    args = ap.parse_args()

    systems = [s.strip() for s in args.systems.split(",") if s.strip()]
    if len(systems) != 4:
        raise ValueError("This script expects exactly 4 systems for a 2x2 panel. Provide 4 via --systems.")

    fig, axes = plt.subplots(2, 2, figsize=(13, 9), dpi=200)
    axes = axes.flatten()

    for ax, system in zip(axes, systems):
        system = system.lower().strip()
        S = np.load(str(ROOT / "data" / "processed" / "fusion" / f"{system}_S_final.npy"))
        U = np.load(str(ROOT / "data" / "processed" / "edl" / f"{system}_edl_uncertainty.npy"))
        umin, umax = float(U.min()), float(U.max())
        if umax > umin:
            U = (U - umin) / (umax - umin)

        class_order = _load_json(ROOT / "data" / "processed" / "fusion" / f"{system}_class_order.json")
        with_u_part = _load_json(ROOT / "data" / "processed" / "fusion" / f"{system}_cac-final_partition.json")
        no_u_part = _load_json(ROOT / "data" / "processed" / "fusion" / f"{system}_cac-final_no_u_partition.json")

        with_u_labels = _to_index_labels(with_u_part, class_order)
        no_u_labels = _to_index_labels(no_u_part, class_order)

        edge = _select_killer_edge(system, alpha=float(args.alpha), topk_u=int(args.topk_u))
        i, j = int(edge["i"]), int(edge["j"])

        # render: single panel showing both states via title+annotation (compact for 4-panel)
        G = _build_knn_graph(S, int(args.knn))
        # Use no-U labels for coloring (left state), but annotation includes both; this keeps each subplot uncluttered.
        _panel(ax, G, no_u_labels, (i, j), f"{system}: edge ({i},{j})")

        s = float(S[i, j])
        u = float(U[i, j])
        w_no = s
        w_with = s * float(np.exp(-u))
        merged = (no_u_labels.get(i, -9999) == no_u_labels.get(j, -9999))
        separated = (with_u_labels.get(i, -9999) != with_u_labels.get(j, -9999))

        line1 = f"Without U: W={w_no:.3f} ({'Merged' if merged else 'Separated'})"
        line2 = f"With U: W={w_with:.3f} ({'Separated' if separated else 'Merged'})"
        line3 = f"S={s:.3f}, U={u:.3f}, W_withU=S·exp(-U)"
        ax.text(0.02, -0.08, line1 + "\n" + line2 + "\n" + line3, transform=ax.transAxes, fontsize=8, va="top")

    fig.suptitle("Case study: U changes edge weights and prevents erroneous merges (4 systems)", y=0.98, fontsize=14)
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = ROOT / "results" / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"case_study_u_effect_4panel_{ts}.png"
    plt.savefig(out_path, bbox_inches="tight")
    plt.close(fig)

    print(f"[OK] saved: {out_path}")


if __name__ == "__main__":
    main()
