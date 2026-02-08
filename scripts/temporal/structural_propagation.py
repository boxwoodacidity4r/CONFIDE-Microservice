"""Structural propagation helpers for temporal modality densification.

Design goal
-----------
Densify S_temp *without hallucinating new edges*.

Rule: only boost edges (i,j) that already have dynamic evidence (S[i,j] > 0).
The boost strength is proportional to a structural adjacency signal (call graph).

We use the processed call graph JSON (method-level) and lift it to class-level:
- build undirected class adjacency A where A[u,v]=1 if any method in class u calls
  any method in class v (either direction).

Then propagation is:
  S'[i,j] = S[i,j] + factor * A[i,j] * S[i,j]
          = S[i,j] * (1 + factor) if structurally adjacent else S[i,j]

So:
- no new nonzero entries are created
- intra edges typically get larger because same-service classes have more calls

This is intentionally conservative and paper-defensible.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple, Set

import numpy as np


def _class_of_method(m: str) -> str:
    # method string format: fqcn.methodName
    return (m or "").rsplit(".", 1)[0]


def load_class_call_adjacency(callgraph_json_path: Path, class_order: List[str]) -> np.ndarray:
    """Load a method-level callgraph and return undirected class adjacency matrix.

    Returns A (NxN) with entries in {0,1}.
    """
    data = json.loads(callgraph_json_path.read_text(encoding="utf-8"))
    edges = data.get("edges", []) or []

    class_to_idx: Dict[str, int] = {c: i for i, c in enumerate(class_order)}
    n = len(class_order)
    A = np.zeros((n, n), dtype=np.float32)

    for e in edges:
        s = _class_of_method(e.get("source", ""))
        t = _class_of_method(e.get("target", ""))
        if not s or not t or s == t:
            continue
        i = class_to_idx.get(s)
        j = class_to_idx.get(t)
        if i is None or j is None:
            continue
        A[i, j] = 1.0
        A[j, i] = 1.0

    np.fill_diagonal(A, 0.0)
    return A


def apply_structural_edge_boost(S: np.ndarray, A: np.ndarray, *, factor: float) -> np.ndarray:
    """Boost existing edges using structural adjacency.

    Only affects entries where S>0. Does not create new edges.
    """
    if factor <= 0:
        return S
    if S.shape != A.shape:
        raise ValueError(f"shape mismatch: S={S.shape} A={A.shape}")

    out = np.array(S, dtype=float, copy=True)
    mask = (out > 0) & (A > 0)
    out[mask] = out[mask] * (1.0 + float(factor))
    return out


def offdiag_nonzero(M: np.ndarray) -> int:
    x = np.array(M, copy=True)
    np.fill_diagonal(x, 0.0)
    return int((x > 0).sum())


def _is_entity_class(c: str) -> bool:
    s = (c or "")
    if ".jpa." in s:
        return True

    # Heuristic: treat WAR-layer data-holding objects as entities/DTOs.
    # This is intentionally narrow: only obvious domain DTO names.
    # Examples seen in Plants callgraph: OrderInfo, ShoppingItem, BackOrderItem, LoginInfo.
    tail = s.rsplit(".", 1)[-1]
    if ".war." in s and (
        tail.endswith("Info")
        or tail.endswith("Item")
        or tail.endswith("Data")
        or tail.endswith("DTO")
        or tail.endswith("Model")
    ):
        return True

    return False


def extract_class_to_entities_from_callgraph(
    callgraph_json_path: Path,
    class_order: List[str],
) -> Dict[int, Set[int]]:
    """Extract a many-to-many mapping: class_idx -> {entity_idx}.

    We treat classes under '*.jpa.*' as entities.

    Evidence sources (callgraph edges, method-level):
    - class(method(source)) -> class(method(target))

    If either endpoint is an entity class, we record the relationship.
    """
    data = json.loads(callgraph_json_path.read_text(encoding="utf-8"))
    edges = data.get("edges", []) or []

    class_to_idx: Dict[str, int] = {c: i for i, c in enumerate(class_order)}

    out: Dict[int, Set[int]] = {}

    for e in edges:
        s_cls = _class_of_method(e.get("source", ""))
        t_cls = _class_of_method(e.get("target", ""))
        if not s_cls or not t_cls:
            continue

        s_is_ent = _is_entity_class(s_cls)
        t_is_ent = _is_entity_class(t_cls)
        if not (s_is_ent or t_is_ent):
            continue

        si = class_to_idx.get(s_cls)
        ti = class_to_idx.get(t_cls)
        if si is None or ti is None:
            continue

        # relation in either direction: non-entity -> entity
        if s_is_ent and not t_is_ent:
            out.setdefault(ti, set()).add(si)
        elif t_is_ent and not s_is_ent:
            out.setdefault(si, set()).add(ti)
        # entity->entity edges are ignored

    return out


def apply_entity_bridging(
    S: np.ndarray,
    class_to_entities: Dict[int, Set[int]],
    *,
    base_weight: float = 0.05,
    boost_factor: float = 0.1,
    max_added_edges_per_entity: int = 2000,
) -> Tuple[np.ndarray, Dict[str, int]]:
    """Entity-based augmentation.

    For any two classes (i,j) that share at least one entity, we add/boost S[i,j].

    - If S[i,j]==0: set to base_weight
    - If S[i,j]>0: multiply by (1+boost_factor)

    This **can create new edges** but they are constrained to a data-layer rationale.

    Returns (new_S, stats).
    """
    n = S.shape[0]
    out = np.array(S, dtype=float, copy=True)

    # Build inverted index: entity_idx -> [class_idx]
    ent_to_classes: Dict[int, List[int]] = {}
    for ci, ents in class_to_entities.items():
        for ei in ents:
            ent_to_classes.setdefault(int(ei), []).append(int(ci))

    added = 0
    boosted = 0

    for ei, cls_list in ent_to_classes.items():
        if len(cls_list) < 2:
            continue

        # bound worst-case quadratic explosion
        cap = int(max_added_edges_per_entity)
        pairs_done = 0

        for a in range(len(cls_list)):
            i = cls_list[a]
            for b in range(a + 1, len(cls_list)):
                j = cls_list[b]
                if i == j:
                    continue
                if out[i, j] > 0:
                    out[i, j] *= (1.0 + float(boost_factor))
                    out[j, i] *= (1.0 + float(boost_factor))
                    boosted += 1
                else:
                    out[i, j] = float(base_weight)
                    out[j, i] = float(base_weight)
                    added += 1

                pairs_done += 1
                if pairs_done >= cap:
                    break
            if pairs_done >= cap:
                break

    np.fill_diagonal(out, np.diag(S))
    stats = {
        "entities": len(ent_to_classes),
        "added_edges": int(added),
        "boosted_edges": int(boosted),
    }
    return out, stats
