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
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple, List, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

# NEW: chord-like (circos) plotting
try:
    import plotly.graph_objects as go
except Exception:  # pragma: no cover
    go = None


ROOT = Path(__file__).resolve().parents[3]


def _load_json(p: Path):
    return json.loads(p.read_text(encoding="utf-8"))


def _load_partitions(system: str, *, with_u_path: Path | None = None, no_u_path: Path | None = None) -> Tuple[Dict[str, int], Dict[str, int]]:
    if with_u_path is None:
        with_u_path = ROOT / "data" / "processed" / "fusion" / f"{system}_cac-final_partition.json"
    with_u = _load_json(with_u_path)

    if no_u_path is None:
        # Prefer the corrected standard no-U snapshot file
        no_u_path = ROOT / "data" / "processed" / "fusion" / f"{system}_cac-final_no_u_partition.json"
        if not no_u_path.exists():
            # Backward compatibility: legacy artifact
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


# Paper palette (user-provided). Will cycle if clusters > len(palette).
PAPER_NODE_PALETTE = [
    "#499BC0",  # normalized from user input '#499BCO'
    "#8FDEE3",
    "#FDD786",
    "#FAAF7F",
    "#F78779",
]


def _labels_to_colors(labels: Dict[int, int]):
    """Map cluster ids to a stable list of hex colors."""
    order = sorted({int(v) for v in labels.values()})
    color_by_cluster = {c: PAPER_NODE_PALETTE[i % len(PAPER_NODE_PALETTE)] for i, c in enumerate(order)}
    return color_by_cluster


def _short_class_name(s: str) -> str:
    s = str(s).strip()
    if not s:
        return s
    # drop package
    if "." in s:
        s = s.split(".")[-1]
    # drop extension
    s = s.replace(".java", "")
    return s


def _panel(ax, G: nx.Graph, labels: Dict[int, int], focus: Tuple[int, int], title: str, *, class_order: list[str] | None = None):
    i, j = focus
    # ego nodes: neighbors of i and j (1-hop)
    nodes = set([i, j])
    nodes |= set(G.neighbors(i))
    nodes |= set(G.neighbors(j))
    H = G.subgraph(nodes).copy()

    # layout
    pos = nx.spring_layout(H, seed=42, k=0.6)

    # colors by cluster (stable, paper palette)
    color_by_cluster = _labels_to_colors(labels)
    node_colors = [color_by_cluster.get(labels.get(n, -1), "#CCCCCC") for n in H.nodes()]
    nx.draw_networkx_nodes(
        H,
        pos,
        node_color=node_colors,
        node_size=260,
        ax=ax,
        linewidths=0.6,
        edgecolors="#2b2b2b",
    )

    # edges
    widths = []
    for u, v, d in H.edges(data=True):
        widths.append(0.5 + 5.0 * float(d.get("weight", 0.0)))
    nx.draw_networkx_edges(H, pos, width=widths, alpha=0.35, ax=ax)

    # highlight focus edge if exists
    if H.has_edge(i, j):
        nx.draw_networkx_edges(H, pos, edgelist=[(i, j)], width=4.0, edge_color="#d62728", alpha=0.95, ax=ax)

    # labels: ONLY show node indices (reviewer-friendly, auditable)
    lbl = {n: str(n) for n in H.nodes()}
    nx.draw_networkx_labels(H, pos, labels=lbl, font_size=9, font_color="#111", ax=ax)

    ax.set_title(title)
    ax.axis("off")


def _format_case_line(*, w_no: float, w_with: float, merged: bool, separated: bool) -> str:
    # merged/separated are mutually exclusive expectations, but booleans are clearer here
    left = f"Without U: W={w_no:.3f} ({'Merged' if merged else 'Separated'})"
    right = f"With U: W={w_with:.3f} ({'Separated' if separated else 'Merged'})"
    return left + " | " + right


def _jaccard_weight_no_u(s: float) -> float:
    # baseline: W = S
    return float(s)


def _weight_with_u_exp(s: float, u: float) -> float:
    # visualization proxy used in this file: W_withU = S * exp(-U)
    return float(s) * float(np.exp(-float(u)))


def _build_chord_edges(G: nx.Graph, nodes: list[int]) -> list[tuple[int, int, float]]:
    """Return undirected edges (u,v,w) within `nodes` from graph G."""
    node_set = set(nodes)
    edges = []
    for u, v, d in G.edges(data=True):
        if u in node_set and v in node_set:
            edges.append((int(u), int(v), float(d.get("weight", 0.0))))
    return edges


def _chord_positions(nodes: list[int]) -> dict[int, tuple[float, float, float]]:
    """Place nodes on a circle.

    Returns mapping: node -> (theta, x, y)
    """
    n = len(nodes)
    out = {}
    for idx, node in enumerate(nodes):
        theta = 2.0 * np.pi * (idx / max(1, n))
        out[int(node)] = (float(theta), float(np.cos(theta)), float(np.sin(theta)))
    return out


def _chord_curve(x0: float, y0: float, x1: float, y1: float, *, bend: float = 0.55, steps: int = 60):
    """Quadratic Bezier curve going through a control point near the center."""
    cx, cy = 0.0, 0.0
    t = np.linspace(0.0, 1.0, int(steps))
    # control point pulls the curve inward
    cpx = cx * (1 - bend)
    cpy = cy * (1 - bend)
    xs = (1 - t) ** 2 * x0 + 2 * (1 - t) * t * cpx + t**2 * x1
    ys = (1 - t) ** 2 * y0 + 2 * (1 - t) * t * cpy + t**2 * y1
    return xs, ys


def _bezier_curve(p0, p1, p2, num=100):
    """Cubic Bezier curve from p0 to p2 with control point at p1."""
    t = np.linspace(0.0, 1.0, num=num)
    mt = 1 - t
    return (
        mt**2 * p0[0] + 2 * mt * t * p1[0] + t**2 * p2[0],
        mt**2 * p0[1] + 2 * mt * t * p1[1] + t**2 * p2[1],
    )


def _plot_chord(
    system: str,
    nodes: List[int],
    edges: List[Tuple[int, int, float]],
    labels: Dict[int, int],
    focus: Tuple[int, int],
    title: str,
    out_path: Path,
    class_order: List[str],
    uminmax_text: str,
):
    """Make a circos/chord-like plot using Plotly.

    Contrastive visualization intent:
      * All normal edges are drawn as faint light-gray.
      * The focus (high-uncertainty / suppressed) edge is emphasized in red with dash style.
      * Place the U annotation near the red edge.
    """

    # stable order around the circle (by node id)
    nodes = sorted({int(n) for n in nodes})
    pos = _chord_positions(nodes)
    color_by_cluster = _labels_to_colors(labels)

    i, j = int(focus[0]), int(focus[1])

    # scale weights to edge width / alpha
    ws = [w for _, _, w in edges] or [0.0]
    w_min, w_max = float(min(ws)), float(max(ws))

    def norm_w(w: float) -> float:
        if w_max > w_min:
            return (float(w) - w_min) / (w_max - w_min)
        return 0.0

    fig = go.Figure()

    # edges
    i0, j0 = int(focus[0]), int(focus[1])
    for a, b, w in edges:
        xa, ya = pos[a][1], pos[a][2]
        xb, yb = pos[b][1], pos[b][2]

        is_focus = (int(a) == i0 and int(b) == j0) or (int(a) == j0 and int(b) == i0)

        # For the focus edge, draw a straight chord so it is unambiguous/clearly visible.
        # For normal edges, keep the curved chord look.
        if is_focus:
            xs = np.array([xa, xb], dtype=float)
            ys = np.array([ya, yb], dtype=float)
        else:
            xs, ys = _bezier_curve((xa, ya), (0.0, 0.0), (xb, yb), num=40)

        if is_focus:
            col = "rgba(220, 38, 38, 0.95)"  # red
            width = 6.0
            dash = "dash"
        else:
            col = "rgba(170, 170, 170, 0.18)"  # faint light gray
            width = 1.0
            dash = "solid"

        fig.add_trace(
            go.Scatter(
                x=xs,
                y=ys,
                mode="lines",
                line=dict(color=col, width=float(width), dash=dash),
                hoverinfo="skip",
                showlegend=False,
            )
        )

    # nodes
    node_x = [pos[n][1] for n in nodes]
    node_y = [pos[n][2] for n in nodes]

    node_text = []
    node_color = []
    for n in nodes:
        cls = _short_class_name(class_order[n]) if 0 <= n < len(class_order) else str(n)
        cid = labels.get(int(n), -1)
        node_text.append(f"{n}: {cls}<br>cluster={cid}")
        node_color.append(color_by_cluster.get(int(cid), "#CCCCCC"))

    fig.add_trace(
        go.Scatter(
            x=node_x,
            y=node_y,
            mode="markers+text",
            text=[str(n) for n in nodes],
            textposition="top center",
            marker=dict(size=16, color=node_color, line=dict(color="#2b2b2b", width=1)),
            hovertext=node_text,
            hoverinfo="text",
            showlegend=False,
        )
    )

    fig.update_layout(
        title=dict(text=title, x=0.5, y=0.98),
        margin=dict(l=10, r=10, t=60, b=40),
        width=980,
        height=540,
        paper_bgcolor="white",
        plot_bgcolor="white",
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, scaleanchor="x", scaleratio=1),
        annotations=[
            dict(
                text=uminmax_text,
                x=0.5,
                y=-0.10,
                xref="paper",
                yref="paper",
                showarrow=False,
                font=dict(size=12, color="#111"),
            )
        ],
    )

    # Add local annotation near the focus edge (place it on the chord mid-point)
    # Compute mid point on the same geometry used above.
    if i0 in pos and j0 in pos:
        xa, ya = pos[i0][1], pos[i0][2]
        xb, yb = pos[j0][1], pos[j0][2]

        # Mid-point of straight chord for the focus edge
        mx, my = (float(xa + xb) / 2.0), (float(ya + yb) / 2.0)
        fig.add_annotation(
            x=mx,
            y=my,
            xref="x",
            yref="y",
            text=uminmax_text.split("<br>")[1] if "<br>" in uminmax_text else uminmax_text,
            showarrow=True,
            arrowhead=2,
            arrowsize=1,
            arrowwidth=1,
            arrowcolor="rgba(220, 38, 38, 0.9)",
            ax=30,
            ay=-30,
            bgcolor="rgba(255, 255, 255, 0.85)",
            bordercolor="rgba(220, 38, 38, 0.6)",
            borderwidth=1,
            font=dict(size=12, color="#111"),
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Try static image export; if it fails, fall back to HTML and do NOT crash.
    try:
        fig.write_image(str(out_path), scale=2)
    except Exception:  # pragma: no cover
        html_path = out_path.with_suffix(".html")
        # IMPORTANT: use self-contained HTML (embed plotly.js) to avoid blank page
        # when CDN is blocked or the user opens file:// without network access.
        fig.write_html(str(html_path), include_plotlyjs=True, full_html=True)
        print(
            "[WARN] PNG export failed (Kaleido/Chrome issue). Wrote interactive HTML instead:\n"
            f"  {html_path}\n"
            "[INFO] HTML is self-contained (plotly.js embedded), should open without internet."
        )
        return


def _combine_html_chord_plots(html_paths: List[Path], out_path: Path, *, layout: str, title: str):
    """Combine multiple self-contained Plotly HTML files into a single HTML page.

    This is intended for quick paper layout comparison (1x4 vs 2x2).
    """
    blocks: List[str] = []
    plotly_js_block: str | None = None

    for idx, p in enumerate(html_paths):
        txt = p.read_text(encoding="utf-8", errors="ignore")

        scripts = re.findall(r"<script[^>]*>[\s\S]*?</script>", txt, flags=re.IGNORECASE)

        # Extract plotly.js from the first file (self-contained output from fig.write_html(include_plotlyjs=True))
        if idx == 0:
            for s in scripts:
                if "window.Plotly" in s or "Plotly.register" in s or "plotly.min.js" in s:
                    plotly_js_block = s
                    break

        # Extract div
        mdiv = re.search(r"(<div[^>]*class=\"plotly-graph-div\"[\s\S]*?</div>)", txt)
        if not mdiv:
            mdiv = re.search(r"(<div[^>]*class='plotly-graph-div'[\s\S]*?</div>)", txt)
        if not mdiv:
            raise RuntimeError(f"Cannot find plotly graph div in {p}")

        # Extract the Plotly.newPlot initializer
        newplot_script = None
        for s in reversed(scripts):
            if "Plotly.newPlot" in s:
                newplot_script = s
                break
        if not newplot_script:
            raise RuntimeError(f"Cannot find Plotly.newPlot script in {p}")

        blocks.append(f"<div class='panel'>{mdiv.group(1)}\n{newplot_script}</div>")

    # Fallback if we couldn't extract plotly.js (should be rare)
    if not plotly_js_block:
        plotly_js_block = "<script src='https://cdn.plot.ly/plotly-2.30.0.min.js'></script>"

    grid_cols = 4 if layout == "1x4" else 2

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        f"""<!doctype html>
<html>
<head>
  <meta charset='utf-8'/>
  <meta name='viewport' content='width=device-width, initial-scale=1'/>
  <title>{title}</title>
  <style>
    body {{ margin: 0; font-family: Arial, Helvetica, sans-serif; background: #fff; }}
    .header {{ padding: 10px 14px; border-bottom: 1px solid #eee; }}
    .header h1 {{ font-size: 16px; margin: 0; font-weight: 600; }}
    .grid {{
      display: grid;
      grid-template-columns: repeat({grid_cols}, minmax(320px, 1fr));
      gap: 10px;
      padding: 10px;
    }}
    .panel {{ border: 1px solid #f0f0f0; border-radius: 8px; padding: 6px; }}
    .plotly-graph-div {{ width: 100% !important; height: 100% !important; }}
    @media (max-width: 1200px) {{
      .grid {{ grid-template-columns: repeat(2, minmax(320px, 1fr)); }}
    }}
    @media (max-width: 720px) {{
      .grid {{ grid-template-columns: repeat(1, minmax(320px, 1fr)); }}
    }}
  </style>
</head>
<body>
  <div class='header'><h1>{title} ({layout})</h1></div>
  {plotly_js_block}
  <div class='grid'>
    {''.join(blocks)}
  </div>
</body>
</html>
""",
        encoding="utf-8",
    )

    print(f"[INFO] Combined HTML written: {out_path}")
    return


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("system", type=str, help="System name: daytrader/acmeair/jpetstore/plants")
    ap.add_argument("--edge", type=str, default=None, help="Edge as 'i,j'. If omitted (chord), auto-pick by max U.")
    ap.add_argument("--selected_edge_json", type=str, default=None, help="Optional: path to case_edge_selected_<system>_<ts>.json (for annotations).")
    ap.add_argument("--with_u_partition", type=str, default=None)
    ap.add_argument("--no_u_partition", type=str, default=None)
    ap.add_argument("--knn", type=int, default=8, help="k for KNN graph used in visualization")

    ap.add_argument("--style", type=str, default="spring", choices=["spring", "chord"], help="Visualization style")
    ap.add_argument("--out", type=str, default=None, help="Optional output path (png). If not set, write to results/plots/")

    ap.add_argument(
        "--combine",
        type=str,
        default=None,
        choices=["1x4", "2x2"],
        help="Combine 4 chord HTML files (daytrader/acmeair/jpetstore/plants) into one HTML page using the given layout.",
    )
    ap.add_argument(
        "--combine_paths",
        type=str,
        default=None,
        help="Optional: comma-separated 4 html paths to combine (overrides auto-pick latest per system).",
    )

    args = ap.parse_args()

    # combine mode
    if args.combine:
        out_dir = ROOT / "results" / "plots"
        out_dir.mkdir(parents=True, exist_ok=True)

        if args.combine_paths:
            htmls = [Path(x.strip()) for x in args.combine_paths.split(",") if x.strip()]
            if len(htmls) != 4:
                raise ValueError("--combine_paths must contain exactly 4 comma-separated html paths")
        else:
            htmls = []
            for s in ["daytrader", "acmeair", "jpetstore", "plants"]:
                cand = sorted(out_dir.glob(f"case_study_u_effect_{s}_*.html"))
                if not cand:
                    raise FileNotFoundError(f"No chord HTML found for {s} under {out_dir}")
                htmls.append(cand[-1])

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = out_dir / f"case_study_u_effect_combined_{args.combine}_{ts}.html"
        _combine_html_chord_plots(htmls, out_path, layout=args.combine, title="Case study U-effect (Chord, Contrastive) | 4 systems")
        return

    # normal plotting mode
    system = (args.system or "").lower().strip()
    if system in {"plantsbywebsphere", "plants_by_websphere"}:
        system = "plants"

    S = np.load(str(ROOT / "data" / "processed" / "fusion" / f"{system}_S_final.npy"))
    U = np.load(str(ROOT / "data" / "processed" / "edl" / f"{system}_edl_uncertainty.npy"))

    umin, umax = float(U.min()), float(U.max())
    if umax > umin:
        U = (U - umin) / (umax - umin)

    focus = None
    ann = None
    if args.selected_edge_json:
        ann = _load_json(Path(args.selected_edge_json))
        focus = (int(ann["i"]), int(ann["j"]))
    elif args.edge:
        a, b = [int(x.strip()) for x in args.edge.split(",")]
        focus = (a, b)
    else:
        if args.style == "chord":
            U2 = np.array(U, copy=True)
            np.fill_diagonal(U2, -1.0)
            iu, ju = np.unravel_index(int(np.argmax(U2)), U2.shape)
            focus = (int(iu), int(ju))
        else:
            raise FileNotFoundError("For spring style you must pass --edge (auto-pick is only enabled for chord).")

    class_order = _load_json(ROOT / "data" / "processed" / "fusion" / f"{system}_class_order.json")

    with_u_part, no_u_part = _load_partitions(
        system,
        with_u_path=(Path(args.with_u_partition) if args.with_u_partition else None),
        no_u_path=(Path(args.no_u_partition) if args.no_u_partition else None),
    )
    with_u_labels = _to_index_labels(with_u_part, class_order)
    no_u_labels = _to_index_labels(no_u_part, class_order)

    i, j = int(focus[0]), int(focus[1])
    s = float(S[i, j])
    u = float(U[i, j])

    # weights used only for annotation
    w_no = _jaccard_weight_no_u(s)
    w_with = _weight_with_u_exp(s, u)

    merged = (no_u_labels.get(i, -9999) == no_u_labels.get(j, -9999))
    separated = (with_u_labels.get(i, -9999) != with_u_labels.get(j, -9999))
    main_line = _format_case_line(w_no=w_no, w_with=w_with, merged=merged, separated=separated)
    sub_line = f"S(i,j)={s:.3f}  U(i,j)={u:.3f}   =>  W_withU=S·exp(-U) | U_norm=minmax[{umin:.3f},{umax:.3f}]"

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = ROOT / "results" / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.out) if args.out else (out_dir / f"case_study_u_effect_{system}_{ts}.png")

    # graph neighborhood
    G = _build_knn_graph(S, int(args.knn))
    nodes = sorted(set([i, j]) | set(G.neighbors(i)) | set(G.neighbors(j)))

    cls_i = _short_class_name(class_order[i]) if 0 <= i < len(class_order) else str(i)
    cls_j = _short_class_name(class_order[j]) if 0 <= j < len(class_order) else str(j)

    if args.style == "chord":
        if go is None:
            raise RuntimeError("Plotly not available. Install plotly and kaleido.")
        edge_list = _build_chord_edges(G, nodes)
        title = f"{system}: edge ({i},{j}) [{cls_i}, {cls_j}]"
        utext = f"{main_line}<br>{sub_line}"
        _plot_chord(
            system=system,
            nodes=nodes,
            edges=edge_list,
            labels=with_u_labels,
            focus=(i, j),
            title=title,
            out_path=out_path,
            class_order=class_order,
            uminmax_text=utext,
        )
        print(f"[OK] saved (chord): {out_path}")
        return

    # spring (kept minimal)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=180)
    _panel(axes[0], G, no_u_labels, (i, j), "w/o U (U≡0)")
    _panel(axes[1], G, with_u_labels, (i, j), "with U (uncertainty-aware)")
    fig.text(0.5, -0.03, main_line + "\n" + sub_line, ha="center", va="top", fontsize=10)
    fig.suptitle(f"Case study: edge ({i}, {j}) [{cls_i}, {cls_j}] under U ablation", y=1.02)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] saved: {out_path}")


if __name__ == "__main__":
    main()
