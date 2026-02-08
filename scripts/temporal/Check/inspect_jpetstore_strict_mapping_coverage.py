"""Inspect strict JPetStore Flow->class mapping coverage.

This script is used to debug why strict temporal becomes sparse.
It reports:
- Flow parent label counts
- Per-flow mapping hit size distribution (#classes mapped per Flow event)
- Missing mapping rate

Run (PowerShell):
  .\.venv\Scripts\python.exe scripts\temporal\inspect_jpetstore_strict_mapping_coverage.py
"""

from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd

try:
    # When run as a module: python -m scripts.temporal.inspect_jpetstore_strict_mapping_coverage
    from .build_S_temp import (
        _expand_endpoint_to_classes,
        _filter_non_business_labels,
        load_class_order,
    )
except Exception:  # pragma: no cover
    # When run as a script from repo root
    from build_S_temp import (
        _expand_endpoint_to_classes,
        _filter_non_business_labels,
        load_class_order,
    )


def _is_flow(label: str) -> bool:
    s = str(label or "").strip().upper()
    return s.startswith("FLOWA_") or s.startswith("FLOWB_") or s.startswith("FLOWC_")


def main() -> None:
    df = pd.read_csv("results/jmeter/jpetstore_results.jtl")
    df = _filter_non_business_labels(df)
    if "success" in df.columns:
        df = df[df["success"].astype(str).str.lower() == "true"]
    if "timeStamp" in df.columns:
        df = df.sort_values("timeStamp")

    class_order, class_to_idx = load_class_order("jpetstore")

    flow_labels = df["label"].dropna().astype(str)
    flow_labels = [x for x in flow_labels if _is_flow(x)]

    print("flow_labels_total", len(flow_labels))
    print("flow_labels_counts", dict(Counter(flow_labels)))

    sizes = []
    missing = 0
    per_flow = Counter()

    for lbl in flow_labels:
        idxs = _expand_endpoint_to_classes(lbl, "jpetstore", class_to_idx, strict=True)
        per_flow[str(lbl)] += len(idxs)
        sizes.append(len(idxs))
        if not idxs:
            missing += 1

    if not sizes:
        print("No Flow labels found in JTL")
        return

    a = np.array(sizes, dtype=int)
    print("flow_events", int(a.size))
    print("missing_events", int(missing), "missing_rate", float(missing / max(1, a.size)))
    print(
        "hit_size min/p50/mean/p90/max",
        int(a.min()),
        float(np.percentile(a, 50)),
        float(a.mean()),
        float(np.percentile(a, 90)),
        int(a.max()),
    )
    print("hit_size_counts", {k: int((a == k).sum()) for k in sorted(set(a.tolist()))})

    # Also show the average hit size per flow label variant
    print("avg_hit_size_by_flow", {k: float(v / 80.0) for k, v in per_flow.items()})


if __name__ == "__main__":
    main()
