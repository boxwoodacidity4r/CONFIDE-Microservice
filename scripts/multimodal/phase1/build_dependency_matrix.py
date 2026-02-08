"""Build dependency matrix JSON for Phase 4 architecture metrics.

Phase 1/structural extraction produces dependency edge lists:
  data/processed/dependency/<app>_dependency.json
with schema:
  {"nodes": [...], "edges": [{"source": s, "target": t}, ...]}

Phase 4 architecture metrics expect a dict-of-dict weighted adjacency:
  {"ClassA": {"ClassB": weight, ...}, ...}

This script converts edge lists into dependency matrices and standardizes names to
Phase1/Phase4 system ids (acmeair/daytrader/jpetstore/plants).

Usage:
  python scripts/multimodal/phase1/build_dependency_matrix.py --system acmeair
  python scripts/multimodal/phase1/build_dependency_matrix.py --system all

Outputs:
  data/processed/dependency/{system}_dependency_matrix.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict


ROOT = Path(__file__).resolve().parents[3]
DEP_DIR = ROOT / "data" / "processed" / "dependency"

# Map Phase systems -> structural extractor app names / filenames
SYSTEM_TO_APP = {
    "acmeair": "acmeair",
    "daytrader": "daytrader7",
    "jpetstore": "jPetStore",
    "plants": "plantsbywebsphere",
}


def _load_edge_list_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict) or "edges" not in data:
        raise ValueError(f"Unexpected dependency format: {path} (expect dict with 'edges')")
    return data


def _normalize_class_name(x: str) -> str:
    # Keep aligned with other scripts: strip trailing .java
    x = (x or "").strip()
    if x.endswith(".java"):
        x = x[: -len(".java")]
    return x


def edge_list_to_matrix(dep_edge_list: dict) -> Dict[str, Dict[str, float]]:
    """Convert {'edges':[{'source':..,'target':..},..]} to weighted adjacency dict."""
    mat: Dict[str, Dict[str, float]] = {}

    edges = dep_edge_list.get("edges", [])
    if not isinstance(edges, list):
        raise ValueError("dep_edge_list['edges'] must be a list")

    for e in edges:
        if not isinstance(e, dict):
            continue
        src = _normalize_class_name(str(e.get("source", "")))
        dst = _normalize_class_name(str(e.get("target", "")))
        if not src or not dst:
            continue

        mat.setdefault(src, {})
        mat[src][dst] = float(mat[src].get(dst, 0.0) + 1.0)  # count duplicates if any

    return mat


def build_for_system(system: str) -> Path:
    system = system.lower().strip()
    if system not in SYSTEM_TO_APP:
        raise ValueError(f"Unknown system: {system}. Expected one of {sorted(SYSTEM_TO_APP.keys())}")

    app = SYSTEM_TO_APP[system]
    in_path = DEP_DIR / f"{app}_dependency.json"
    if not in_path.exists():
        raise FileNotFoundError(
            f"Missing structural dependency edge list: {in_path}.\n"
            "Hint: run scripts/structural/extract_features.py to generate data/processed/dependency/*_dependency.json"
        )

    edge_list = _load_edge_list_json(in_path)
    mat = edge_list_to_matrix(edge_list)

    out_path = DEP_DIR / f"{system}_dependency_matrix.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(mat, f, indent=2, ensure_ascii=False)

    print(f"[OK] {system}: {in_path.name} -> {out_path} | nodes={len(mat)}")
    return out_path


def main():
    p = argparse.ArgumentParser(description="Convert dependency edge list to dependency_matrix.json for Phase4.")
    p.add_argument("--system", default="all", help="acmeair/daytrader/jpetstore/plants or all")
    args = p.parse_args()

    systems = ["acmeair", "daytrader", "jpetstore", "plants"] if args.system == "all" else [args.system]
    for s in systems:
        build_for_system(s)


if __name__ == "__main__":
    main()
