"""Plot a paper-ready case-study figure for uncertainty U effect.

Given:
- system name
- a selected edge (i,j) (or auto-pick from most-uncertain list)
- partitions for with-U vs no-U

This script generates a 2-panel figure:
  Left:  w/o U (U≡0)
  Right: with U

Each panel shows a small ego-graph around nodes i and j, with:
- node colors = predicted cluster id
- edge widths = similarity weight (S_final or CAC weight proxy)
- highlighted edge (i,j)

Outputs:
- results/plots/case_study_u_effect_<system>_<ts>.png

Note: This is a visualization helper; it does not rerun Phase3.
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


def _load_partitions(system: str, *, with_u_path: Path | None = None, no_u_path: Path | None = None) -> Tuple[Dict[str, int], Dict[str, int]]:
    if with_u_path is None:
        with_u_path = ROOT / "data" / "processed" / "fusion" / f"{system}_cac-final_partition.json"
    with_u = _load_json(with_u_path)

    if no_u_path is None:
        # 优先使用修正后的标准 no-U 快照文件
        no_u_path = ROOT / "data" / "processed" / "fusion" / f"{system}_cac-final_no_u_partition.json"
        if not no_u_path.exists():
            # 兼容旧产物
            no_u_path = ROOT / "data" / "processed" / "fusion" / f"{system}_cac-final_best_by_bcubed_partition.json"
    no_u = _load_json(no_u_path)
    return with_u, no_u


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
        # top-k neighbors excluding self
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
    # ego nodes: neighbors of i and j (1-hop)
    nodes = set([i, j])
    nodes |= set(G.neighbors(i))
    nodes |= set(G.neighbors(j))
    H = G.subgraph(nodes).copy()

    # layout
    pos = nx.spring_layout(H, seed=42, k=0.6)

    # colors by cluster
    clus = [labels.get(n, -1) for n in H.nodes()]
    nx.draw_networkx_nodes(H, pos, node_color=clus, cmap=plt.cm.tab20, node_size=260, ax=ax, linewidths=0.5, edgecolors="#333")

    # edges
    widths = []
    for u, v, d in H.edges(data=True):
        widths.append(0.5 + 5.0 * float(d.get("weight", 0.0)))
    nx.draw_networkx_edges(H, pos, width=widths, alpha=0.35, ax=ax)

    # highlight focus edge if exists
    if H.has_edge(i, j):
        nx.draw_networkx_edges(H, pos, edgelist=[(i, j)], width=4.0, edge_color="#d62728", alpha=0.95, ax=ax)

    # labels only for focus nodes
    nx.draw_networkx_labels(H, pos, labels={i: str(i), j: str(j)}, font_size=11, font_color="#111", ax=ax)

    ax.set_title(title)
    ax.axis("off")


def _format_case_line(*, w_no: float, w_with: float, merged: bool, separated: bool) -> str:
    # merged/separated 是互斥期望，但这里用布尔更直观
    left = f"Without U: W={w_no:.3f} ({'Merged' if merged else 'Separated'})"
    right = f"With U: W={w_with:.3f} ({'Separated' if separated else 'Merged'})"
    return left + " | " + right


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("system", type=str)
    ap.add_argument("--edge", type=str, default=None, help="Edge as 'i,j'. If omitted, pick the top-1 from case_study_most_uncertain JSON.")
    ap.add_argument("--case_json", type=str, default=None, help="Path to case_study_most_uncertain_<system>_<ts>.json")
    ap.add_argument("--selected_edge_json", type=str, default=None, help="Optional: path to case_edge_selected_<system>_<ts>.json (for annotations).")
    ap.add_argument("--with_u_partition", type=str, default=None)
    ap.add_argument("--no_u_partition", type=str, default=None)
    ap.add_argument("--knn", type=int, default=8, help="k for KNN graph used in visualization")
    args = ap.parse_args()

    system = args.system.lower().strip()
    S = np.load(str(ROOT / "data" / "processed" / "fusion" / f"{system}_S_final.npy"))
    U = np.load(str(ROOT / "data" / "processed" / "edl" / f"{system}_edl_uncertainty.npy"))
    # 归一化 U（与 Phase3 保持一致）
    umin, umax = float(U.min()), float(U.max())
    if umax > umin:
        U = (U - umin) / (umax - umin)

    # decide edge
    focus = None
    ann = None
    if args.selected_edge_json:
        ann = _load_json(Path(args.selected_edge_json))
        focus = (int(ann["i"]), int(ann["j"]))
    elif args.edge:
        a, b = [int(x.strip()) for x in args.edge.split(",")]
        focus = (a, b)
    else:
        case_path = Path(args.case_json) if args.case_json else None
        if case_path is None:
            cand = sorted((ROOT / "results" / "ablation").glob(f"case_study_most_uncertain_{system}_*.json"))
            if not cand:
                raise FileNotFoundError("No case_study_most_uncertain JSON found. Run run_u_ablation_and_daytrader_mu.py first.")
            case_path = cand[-1]
        case = _load_json(case_path)
        e0 = (case.get("edges") or [])[0]
        focus = (int(e0["i"]), int(e0["j"]))

    assert focus is not None

    with_u_part, no_u_part = _load_partitions(
        system,
        with_u_path=(Path(args.with_u_partition) if args.with_u_partition else None),
        no_u_path=(Path(args.no_u_partition) if args.no_u_partition else None),
    )

    class_order = _load_json(ROOT / "data" / "processed" / "fusion" / f"{system}_class_order.json")
    with_u_labels = _to_index_labels(with_u_part, class_order)
    no_u_labels = _to_index_labels(no_u_part, class_order)

    G = _build_knn_graph(S, int(args.knn))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=180)
    _panel(axes[0], G, no_u_labels, focus, "w/o U (U≡0)")
    _panel(axes[1], G, with_u_labels, focus, "with U (uncertainty-aware)")

    # --- B 补强：自动会说话的标注（不再依赖 selected_edge_json） ---
    i, j = int(focus[0]), int(focus[1])
    s = float(S[i, j])
    u = float(U[i, j])
    w_no = s
    w_with = s * float(np.exp(-u))

    merged = (no_u_labels.get(i, -9999) == no_u_labels.get(j, -9999))
    separated = (with_u_labels.get(i, -9999) != with_u_labels.get(j, -9999))

    main_line = _format_case_line(w_no=w_no, w_with=w_with, merged=merged, separated=separated)
    sub_line = f"S(i,j)={s:.3f}  U(i,j)={u:.3f}   =>  W_withU=S·exp(-U)"

    fig.text(0.5, -0.03, main_line + "\n" + sub_line, ha="center", va="top", fontsize=10)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = ROOT / "results" / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"case_study_u_effect_{system}_{ts}.png"
    fig.suptitle(f"Case study: edge ({focus[0]}, {focus[1]}) under U ablation", y=1.02)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close(fig)

    print(f"[OK] saved: {out_path}")


if __name__ == "__main__":
    main()
