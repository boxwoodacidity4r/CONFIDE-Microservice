"""Time/space alignment (clock-skew) checker between JMeter JTL and OTEL trace NDJSON.

Goal
----
Validate that (1) counts and (2) action-frequency and (3) time offset (Δt) between
JMeter request timestamps and corresponding SERVER span start timestamps are stable.

Inputs
------
- --jtl: JMeter results .jtl (CSV)
- --traces: OTEL collector output NDJSON file (each line is one OTLP JSON export)

Matching strategy
-----------------
We avoid 1:1 global matching by traceId because JTL doesn't carry trace context.
Instead we match by request *signature*:
- For action requests: key = (method inferred, "Catalog.action", actionLabel)
  where actionLabel is viewCategory/viewProduct/viewItem/search extracted from URL.
- For home: key = ("GET", "/jpetstore/", "home")

Then for each key we perform a monotonic time match: sort JTL events and spans by
start time, and pair in order. This is robust when concurrency is low (as in your
current runs).

Outputs
-------
Prints a small report and exits non-zero if it detects severe mismatch.

"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


@dataclass(frozen=True)
class JtlEvent:
    t_ms: int
    label: str
    method: str
    url: str
    key: Tuple[str, str, str]


@dataclass(frozen=True)
class SpanEvent:
    t_ms: int
    method: str
    path: str
    query: str
    route: str
    key: Tuple[str, str, str]


def _safe_int(v: str, default: int = 0) -> int:
    try:
        return int(v)
    except Exception:
        return default


def _parse_query_action(url_or_query: str) -> Optional[str]:
    # Works for both full URL and the otel url.query value.
    # Example: "...Catalog.action?viewCategory=&categoryId=FISH" or "viewCategory=&categoryId=FISH"
    s = url_or_query
    if "?" in s:
        s = s.split("?", 1)[1]
    # action is the first key
    if not s:
        return None
    first = s.split("&", 1)[0]
    if "=" in first:
        return first.split("=", 1)[0] or None
    return None


def _infer_method_from_label(label: str) -> str:
    # In this project JMeter labels map to known methods.
    return "POST" if label == "search" else "GET"


def _make_jtl_key(label: str, url: str) -> Tuple[str, str, str]:
    method = _infer_method_from_label(label)
    if label == "home":
        return ("GET", "/jpetstore/", "home")
    if label in {"viewCategory", "viewProduct", "viewItem", "search"}:
        return (method, "/jpetstore/actions/Catalog.action", label)
    # Fallback: try to extract action from url
    action = _parse_query_action(url) or label
    path = "/jpetstore/" if "/jpetstore/" in url and "Catalog.action" not in url else "/jpetstore/actions/Catalog.action"
    return (method, path, action)


def read_jtl_events(jtl_path: Path) -> List[JtlEvent]:
    events: List[JtlEvent] = []
    with jtl_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            label = row[2]
            # Skip Transaction Controller rows
            if label.startswith("Flow") or label.startswith("Transaction"):
                continue
            t_ms = _safe_int(row[0])
            url = row[13] if len(row) > 13 else ""
            method = _infer_method_from_label(label)
            key = _make_jtl_key(label, url)
            events.append(JtlEvent(t_ms=t_ms, label=label, method=method, url=url, key=key))
    events.sort(key=lambda e: e.t_ms)
    return events


def _attr(attrs: Iterable[Dict[str, Any]], key: str, default: str = "") -> str:
    for a in attrs:
        if a.get("key") != key:
            continue
        v = a.get("value") or {}
        if "stringValue" in v:
            return str(v["stringValue"])
        if "intValue" in v:
            return str(v["intValue"])
    return default


def iter_spans_from_ndjson(traces_path: Path) -> Iterable[Dict[str, Any]]:
    with traces_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            for rs in obj.get("resourceSpans", []) or []:
                resource_attrs = (rs.get("resource") or {}).get("attributes") or []
                service_name = _attr(resource_attrs, "service.name")
                if service_name != "jpetstore":
                    continue
                for ss in rs.get("scopeSpans", []) or []:
                    for span in ss.get("spans", []) or []:
                        yield span


def read_span_events(traces_path: Path) -> List[SpanEvent]:
    spans: List[SpanEvent] = []
    for span in iter_spans_from_ndjson(traces_path):
        # Only SERVER spans
        if span.get("kind") != 2:
            continue
        attrs = span.get("attributes") or []
        method = _attr(attrs, "http.request.method")
        path = _attr(attrs, "url.path")
        query = _attr(attrs, "url.query")
        route = _attr(attrs, "http.route")
        start_ns = _safe_int(str(span.get("startTimeUnixNano", "0")))
        t_ms = start_ns // 1_000_000

        action = None
        if route.endswith("*.action") or path.endswith("Catalog.action"):
            action = _parse_query_action(query)
        if path == "/jpetstore/" or route == "/jpetstore/*":
            key = ("GET", "/jpetstore/", "home")
        else:
            key = (method or "GET", "/jpetstore/actions/Catalog.action", action or "unknown")

        spans.append(SpanEvent(t_ms=t_ms, method=method or "", path=path, query=query, route=route, key=key))

    spans.sort(key=lambda s: s.t_ms)
    return spans


def _pair_by_key(jtls: List[JtlEvent], spans: List[SpanEvent]) -> List[Tuple[JtlEvent, SpanEvent]]:
    by_key_j: Dict[Tuple[str, str, str], List[JtlEvent]] = {}
    by_key_s: Dict[Tuple[str, str, str], List[SpanEvent]] = {}

    for e in jtls:
        by_key_j.setdefault(e.key, []).append(e)
    for s in spans:
        by_key_s.setdefault(s.key, []).append(s)

    pairs: List[Tuple[JtlEvent, SpanEvent]] = []
    for key, jlist in by_key_j.items():
        slist = by_key_s.get(key, [])
        n = min(len(jlist), len(slist))
        for i in range(n):
            pairs.append((jlist[i], slist[i]))

    pairs.sort(key=lambda p: p[0].t_ms)
    return pairs


def _summary(values: List[int]) -> Dict[str, float]:
    if not values:
        return {"n": 0, "mean": math.nan, "stdev": math.nan, "min": math.nan, "max": math.nan}
    if len(values) == 1:
        return {"n": 1, "mean": float(values[0]), "stdev": 0.0, "min": float(values[0]), "max": float(values[0])}
    return {
        "n": len(values),
        "mean": float(statistics.mean(values)),
        "stdev": float(statistics.pstdev(values)),
        "min": float(min(values)),
        "max": float(max(values)),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jtl", required=True, type=Path)
    ap.add_argument("--traces", required=True, type=Path)
    ap.add_argument("--max-stdev-ms", default=2000, type=int)
    args = ap.parse_args()

    jtls = read_jtl_events(args.jtl)
    spans = read_span_events(args.traces)

    # 1) Count checks
    j_count = len(jtls)
    s_count = len(spans)

    # 2) Frequency checks
    def count_j(label: str) -> int:
        return sum(1 for e in jtls if e.label == label)

    def count_s(action: str) -> int:
        return sum(1 for sp in spans if _parse_query_action(sp.query) == action)

    # 3) Δt distribution
    pairs = _pair_by_key(jtls, spans)
    deltas = [(sp.t_ms - jt.t_ms) for jt, sp in pairs]
    summ = _summary(deltas)

    print("=== Temporal alignment report (JPetStore) ===")
    print(f"JTL events (non-transaction): {j_count}")
    print(f"Trace SERVER spans (service.name=jpetstore): {s_count}")
    print(f"Paired samples: {len(pairs)}")
    print("--- Action frequency (JTL vs Traces) ---")
    for lbl in ["home", "search", "viewCategory", "viewProduct", "viewItem"]:
        if lbl == "home":
            print(f"{lbl:12s}: {count_j(lbl)} (JTL)  |  {sum(1 for sp in spans if sp.key == ('GET','/jpetstore/','home'))} (spans)")
        elif lbl == "search":
            print(f"{lbl:12s}: {count_j(lbl)} (JTL)  |  {count_s('search')} (spans)")
        else:
            print(f"{lbl:12s}: {count_j(lbl)} (JTL)  |  {count_s(lbl)} (spans)")

    print("--- Clock skew Δt = spanStartMs - jtlTimestampMs (ms) ---")
    print(f"n={summ['n']}, mean={summ['mean']:.1f}, stdev={summ['stdev']:.1f}, min={summ['min']:.1f}, max={summ['max']:.1f}")

    # Heuristics
    exit_code = 0
    if j_count == 0 or s_count == 0:
        print("[FAIL] Empty input detected.")
        return 2

    if abs(j_count - s_count) > max(2, int(0.05 * j_count)):
        print("[WARN] Count mismatch > 5% (expected near 1:1).")
        exit_code = max(exit_code, 1)

    if summ["n"] >= 5 and summ["stdev"] > args.max_stdev_ms:
        print(f"[FAIL] High Δt stdev detected (> {args.max_stdev_ms} ms). Clock skew/jitter likely.")
        exit_code = max(exit_code, 2)
    elif summ["n"] >= 5 and summ["stdev"] > 500:
        print("[WARN] Moderate Δt variation detected (> 500 ms). Might affect sessionization if window is tight.")
        exit_code = max(exit_code, 1)
    else:
        print("[OK] Δt is stable enough for sessionization.")

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
