"""Analyze per-class session coverage from JMeter JTL sessions.

Goal
----
For a given system (especially Plants), compute how often each *class* appears across
JTL-derived sessions:

    coverage(class) = sessions_with_class / total_sessions

This is used to identify high-frequency "glue" / hub classes (often WAR-layer) that
appear in most sessions and destroy intra/inter separation.

This script reuses `build_S_temp._build_sessions_from_jtl` to ensure the sessionization
and strict mapping match the temporal build pipeline.

Usage (PowerShell)
------------------
  .\.venv\Scripts\python.exe scripts\temporal\analyze_jtl_session_coverage.py --system plants --top-k 20

  # Also dump a JSON report for later mapping surgery
  .\.venv\Scripts\python.exe scripts\temporal\analyze_jtl_session_coverage.py --system plants --top-k 50 --out data/processed/temporal/plants_jtl_session_coverage.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[2]


# -----------------------------------------------------------------------------
# Self-contained fallback sessionization (Paper-friendly)
# - Used only when build_S_temp._build_sessions_from_jtl is unavailable.
# - Implements strict mapping for DayTrader and thread_iteration grouping.
# -----------------------------------------------------------------------------
try:
    from scripts.temporal.temporal_gate_report import _load_jtl_rows as _tg_load_jtl_rows
    from scripts.temporal.temporal_gate_report import _extract_thread_iteration as _tg_extract_thread_iteration
except Exception:  # keep standalone

    def _tg_load_jtl_rows(jtl_path: Path, *, max_rows: int) -> List[Dict[str, str]]:
        import csv

        rows: List[Dict[str, str]] = []
        if not jtl_path.is_file():
            return rows
        with jtl_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                if max_rows and i >= max_rows:
                    break
                if row:
                    rows.append(row)
        return rows

    def _tg_extract_thread_iteration(row: Dict[str, str]) -> Tuple[str, int]:
        thread_name = (row.get("threadName") or row.get("thread") or "").strip()
        it_raw = row.get("threadIteration") or row.get("iteration") or ""
        try:
            it = int(float(it_raw))
        except Exception:
            it = -1
        return thread_name, it


DAYTRADER_STRICT_LABEL_TO_ENTRYPOINT: Dict[str, List[str]] = {
    # Main servlet entrypoint for most DT workloads
    "portfolio": ["com.ibm.websphere.samples.daytrader.web.TradeAppServlet"],
    "quotes": ["com.ibm.websphere.samples.daytrader.web.TradeAppServlet"],
    "tradestock": ["com.ibm.websphere.samples.daytrader.web.TradeAppServlet"],
    "buystock": ["com.ibm.websphere.samples.daytrader.web.TradeAppServlet"],
    "sellstock": ["com.ibm.websphere.samples.daytrader.web.TradeAppServlet"],
    "login": ["com.ibm.websphere.samples.daytrader.web.TradeAppServlet"],
    "userlogin": ["com.ibm.websphere.samples.daytrader.web.TradeAppServlet"],
    "account": ["com.ibm.websphere.samples.daytrader.web.TradeAppServlet"],
    "userinfo": ["com.ibm.websphere.samples.daytrader.web.TradeAppServlet"],
    "home": ["com.ibm.websphere.samples.daytrader.web.TradeAppServlet"],
    "buy": ["com.ibm.websphere.samples.daytrader.web.TradeAppServlet"],
    "sell": ["com.ibm.websphere.samples.daytrader.web.TradeAppServlet"],
    "order": ["com.ibm.websphere.samples.daytrader.web.TradeAppServlet"],
    "logout": ["com.ibm.websphere.samples.daytrader.web.TradeAppServlet"],
    "register": ["com.ibm.websphere.samples.daytrader.web.TradeScenarioServlet"],
    "registeruser": ["com.ibm.websphere.samples.daytrader.web.TradeScenarioServlet"],

    # TransactionController parent labels
    "t_home": ["com.ibm.websphere.samples.daytrader.web.TradeAppServlet"],
    "t_login": ["com.ibm.websphere.samples.daytrader.web.TradeAppServlet"],
    "t_register": ["com.ibm.websphere.samples.daytrader.web.TradeScenarioServlet"],
    "t_portfolio": ["com.ibm.websphere.samples.daytrader.web.TradeAppServlet"],
    "t_quotes": ["com.ibm.websphere.samples.daytrader.web.TradeAppServlet"],
    "t_buy": ["com.ibm.websphere.samples.daytrader.web.TradeAppServlet"],
    "t_sell": ["com.ibm.websphere.samples.daytrader.web.TradeAppServlet"],
    "t_accountlogout": ["com.ibm.websphere.samples.daytrader.web.TradeAppServlet"],
}


def _norm_label(s: str) -> str:
    return (s or "").strip().lower().replace(" ", "").replace("_", "").replace("-", "")


def _build_sessions_from_jtl_daytrader_fallback(
    system: str,
    *,
    group_by: str,
    max_events: int,
    min_events_per_session: int,
    class_to_idx: Dict[str, int],
) -> List[List[int]]:
    jtl = ROOT / "results" / "jmeter" / "daytrader_results.jtl"
    rows = _tg_load_jtl_rows(jtl, max_rows=max_events if max_events > 0 else 0)

    groups: Dict[str, List[int]] = {}
    for row in rows:
        label = (row.get("label") or row.get("Label") or "").strip()
        if not label:
            continue

        mapped = DAYTRADER_STRICT_LABEL_TO_ENTRYPOINT.get(_norm_label(label), [])
        idxs: List[int] = []
        for fqcn in mapped:
            idx = class_to_idx.get(fqcn)
            if idx is not None:
                idxs.append(int(idx))
        if not idxs:
            continue

        if group_by == "thread":
            key = (row.get("threadName") or row.get("thread") or "").strip()
        else:
            tname, it = _tg_extract_thread_iteration(row)
            key = f"{tname}::it={it}"

        if not key:
            continue
        groups.setdefault(key, []).extend(idxs)

    sessions = [s for s in groups.values() if len(s) >= int(min_events_per_session)]
    return sessions


def _load_class_order(system: str) -> List[str]:
    # Match build_S_temp's physical naming
    physical = {"daytrader7": "daytrader", "plantsbywebsphere": "plants"}.get(system, system)
    p = ROOT / "data" / "processed" / "fusion" / f"{physical}_class_order.json"
    if not p.is_file():
        alt = ROOT / "data" / "processed" / "fusion" / f"{system}_class_order.json"
        if alt.is_file():
            p = alt
        else:
            raise FileNotFoundError(f"class_order not found for {system}: {p}")
    return json.loads(p.read_text(encoding="utf-8"))


def _coverage_from_sessions(sessions: List[List[int]], n: int) -> Tuple[List[int], List[float]]:
    total = len(sessions)
    counts = [0] * n
    if total == 0:
        return counts, [0.0] * n

    for s in sessions:
        for idx in set(int(i) for i in s):
            if 0 <= idx < n:
                counts[idx] += 1

    cov = [c / float(total) for c in counts]
    return counts, cov


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", required=True, help="acmeair|daytrader|daytrader7|jpetstore|plants|plantsbywebsphere")
    ap.add_argument("--group-by", default="thread_iteration", choices=["thread", "thread_iteration", "sliding_window"])
    ap.add_argument("--max-events", type=int, default=80)
    ap.add_argument("--min-events", type=int, default=2)
    ap.add_argument("--window-size", type=int, default=12)
    ap.add_argument("--stride", type=int, default=6)
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--war-threshold", type=float, default=0.5, help="flag war.* classes with coverage >= this")
    ap.add_argument("--out", default="", help="optional JSON output path")
    return ap.parse_args()


def main() -> None:
    args = parse_args()

    # Try to import the canonical sessionization first; fall back if missing.
    _build_sessions_from_jtl = None
    try:
        from .build_S_temp import _build_sessions_from_jtl as _b

        _build_sessions_from_jtl = _b
    except Exception:
        try:
            from build_S_temp import _build_sessions_from_jtl as _b  # type: ignore

            _build_sessions_from_jtl = _b
        except Exception:
            _build_sessions_from_jtl = None

    order = _load_class_order(args.system)
    class_to_idx = {c: i for i, c in enumerate(order)}

    if _build_sessions_from_jtl is not None:
        sessions = _build_sessions_from_jtl(
            args.system,
            strict=True,
            group_by=args.group_by,
            window_size=args.window_size,
            stride=args.stride,
            max_events=args.max_events,
            min_events_per_session=args.min_events,
        )
    else:
        if args.system not in {"daytrader", "daytrader7"}:
            raise ImportError(
                "build_S_temp._build_sessions_from_jtl is unavailable and no fallback is implemented for this system. "
                "Fix build_S_temp.py or run the gate script for Plants only."
            )
        sessions = _build_sessions_from_jtl_daytrader_fallback(
            args.system,
            group_by=args.group_by if args.group_by in {"thread", "thread_iteration"} else "thread_iteration",
            max_events=args.max_events,
            min_events_per_session=args.min_events,
            class_to_idx=class_to_idx,
        )

    counts, cov = _coverage_from_sessions(sessions, n=len(order))

    ranked = sorted(
        [(i, order[i], counts[i], cov[i]) for i in range(len(order)) if counts[i] > 0],
        key=lambda x: (x[3], x[2]),
        reverse=True,
    )

    print("=" * 80)
    print(f"[JTL SESSION COVERAGE] system={args.system} sessions={len(sessions)} group_by={args.group_by}")
    print(f"top{args.top_k} classes by coverage:")
    for (i, cls, c, r) in ranked[: args.top_k]:
        flag = ""
        if (".war." in cls or cls.endswith(".war") or ".war." in cls.lower()) and r >= float(args.war_threshold):
            flag = "  <-- WAR glue candidate"
        print(f"  {r:6.2%}  ({c:4d})  idx={i:4d}  {cls}{flag}")

    war_hits = [(i, cls, c, r) for (i, cls, c, r) in ranked if ".war." in cls.lower() and r >= float(args.war_threshold)]
    if war_hits:
        print("\nWAR-layer candidates (coverage >= threshold):")
        for (i, cls, c, r) in war_hits:
            print(f"  {r:6.2%}  ({c:4d})  idx={i:4d}  {cls}")

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload: Dict[str, object] = {
            "system": args.system,
            "group_by": args.group_by,
            "sessions": len(sessions),
            "params": {
                "max_events": args.max_events,
                "min_events": args.min_events,
                "window_size": args.window_size,
                "stride": args.stride,
            },
            "ranked": [
                {"idx": int(i), "class": cls, "sessions_with_class": int(c), "coverage": float(r)}
                for (i, cls, c, r) in ranked
            ],
            "war_candidates": [
                {"idx": int(i), "class": cls, "sessions_with_class": int(c), "coverage": float(r)}
                for (i, cls, c, r) in war_hits
            ],
        }
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nwrote: {out_path}")


if __name__ == "__main__":
    main()
