from pathlib import Path
import json
import argparse
from collections import Counter
from typing import Dict, Any

ROOT = Path(__file__).resolve().parents[2]


def make_paths(system: str) -> tuple[Path, Path]:
    """Build TRACE_PATH / LOG_PATH for a given OTEL service (system)."""
    # Alias: Plants traces are stored as plants.json in this workspace
    trace_system = "plants" if system == "plantsbywebsphere" else system
    trace_path = ROOT / "data" / "processed" / "traces" / f"{trace_system}.json"
    log_path = ROOT / "data" / "processed" / "logs" / f"{system}_only.log"
    return trace_path, log_path


def inspect_traces(trace_path: Path, max_lines: int = 200) -> None:
    print("==== TRACE INSPECTION ====")
    attr_key_counter: Counter[str] = Counter()
    span_name_counter: Counter[str] = Counter()
    sample_spans = []

    if not trace_path.exists():
        print(f"[TRACE] Trace file not found: {trace_path}")
        return

    with trace_path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= max_lines:
                break
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue

            for resource_span in data.get("resourceSpans", []):
                for scope_span in resource_span.get("scopeSpans", []):
                    for span in scope_span.get("spans", []):
                        name = span.get("name", "")
                        span_name_counter[name] += 1
                        attrs = span.get("attributes", [])
                        for a in attrs:
                            key = a.get("key")
                            if key:
                                attr_key_counter[key] += 1
                        # Collect a few sample spans for inspection.
                        if len(sample_spans) < 10:
                            sample_spans.append(span)

    print(f"[TRACE] Unique span names: {len(span_name_counter)}")
    for name, cnt in span_name_counter.most_common(10):
        print(f"[TRACE] span.name sample: '{name}' -> {cnt} spans")

    print(f"[TRACE] Unique attribute keys: {len(attr_key_counter)}")
    for key, cnt in attr_key_counter.most_common(20):
        print(f"[TRACE] attr key: '{key}' -> {cnt} occurrences")

    print("[TRACE] Sample span details (up to 3):")
    for s in sample_spans[:3]:
        print(json.dumps(s, indent=2))


def inspect_logs(log_path: Path, max_lines: int = 200) -> None:
    print("\n==== LOG INSPECTION ====")
    scope_name_counter: Counter[str] = Counter()
    has_trace_id = 0
    sample_records = []

    if not log_path.exists():
        print(f"[LOG] Log file not found: {log_path}")
        return

    with log_path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= max_lines:
                break
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue

            for resource_log in data.get("resourceLogs", []):
                for scope_log in resource_log.get("scopeLogs", []):
                    scope = scope_log.get("scope", {}) or {}
                    scope_name = scope.get("name", "")
                    if scope_name:
                        scope_name_counter[scope_name] += 1
                    for record in scope_log.get("logRecords", []):
                        if record.get("traceId"):
                            has_trace_id += 1
                        # Collect a few sample log records.
                        if len(sample_records) < 10:
                            sample_records.append({
                                "traceId": record.get("traceId"),
                                "spanId": record.get("spanId"),
                                "body": record.get("body"),
                                "severityText": record.get("severityText"),
                                "scopeName": scope_name,
                            })

    print(f"[LOG] Unique scope names: {len(scope_name_counter)}")
    for name, cnt in scope_name_counter.most_common(10):
        print(f"[LOG] scope.name sample: '{name}' -> {cnt} logs")

    print(f"[LOG] Records with traceId: {has_trace_id}")

    print("[LOG] Sample log records (up to 3):")
    for r in sample_records[:3]:
        print(json.dumps(r, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect OTEL traces/logs fields for a given system (service.name).",
    )
    parser.add_argument(
        "--system",
        type=str,
        default="acmeair",
        choices=["acmeair", "daytrader7", "jpetstore", "plantsbywebsphere"],
        help="Target system / OTEL service.name.",
    )
    parser.add_argument(
        "--max-lines",
        type=int,
        default=200,
        help="Maximum number of JSON lines to inspect for traces/logs.",
    )
    args = parser.parse_args()

    trace_path, log_path = make_paths(args.system)
    print(f"ROOT: {ROOT}")
    print(f"TRACE_PATH: {trace_path}")
    print(f"LOG_PATH:   {log_path}")
    inspect_traces(trace_path, max_lines=args.max_lines)
    inspect_logs(log_path, max_lines=args.max_lines)


if __name__ == "__main__":
    main()
