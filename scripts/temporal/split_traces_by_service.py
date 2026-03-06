from pathlib import Path
import json
import argparse
from typing import Dict, List, TextIO

# List of service.name values to split (corresponding to the four monolith applications)
APPS = ["acmeair", "daytrader7", "jpetstore", "plantsbywebsphere"]

# Output filename aliases (service.name -> output stem)
OUT_NAME_ALIAS = {
    "plantsbywebsphere": "plants",
}


def _out_stem(service_name: str) -> str:
    return OUT_NAME_ALIAS.get(service_name, service_name)


def _get_service_name_from_resource(resource: Dict) -> str:
    """Extract service.name (stringValue) from OTEL resource.attributes."""
    for a in resource.get("attributes", []):
        if a.get("key") == "service.name":
            v = a.get("value", {})
            if isinstance(v, dict):
                return v.get("stringValue", "") or ""
    return ""


def split_traces_by_service(
    src_path: Path,
    out_dir: Path,
    services: List[str],
) -> None:
    """Split all_traces.json into per-app trace files by service.name.

    Each output line contains one or more resourceSpans, but only the subset
    belonging to the target service.

    Output: data/processed/traces/{service}.json

    Note: some services may be aliased to different output names (e.g.,
    plantsbywebsphere -> plants) via OUT_NAME_ALIAS.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    writers: Dict[str, "TextIO"] = {}

    try:
        # Pre-open all output files
        for svc in services:
            dst = out_dir / f"{_out_stem(svc)}.json"
            writers[svc] = dst.open("w", encoding="utf-8")

        with src_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue

                resource_spans = obj.get("resourceSpans", [])
                # Collect resourceSpans per service
                svc_to_rs: Dict[str, List[Dict]] = {svc: [] for svc in services}

                for rs in resource_spans:
                    resource = rs.get("resource", {}) or {}
                    svc_name = _get_service_name_from_resource(resource)
                    if svc_name in services:
                        svc_to_rs[svc_name].append(rs)

                # Write to the corresponding per-service file
                for svc in services:
                    if not svc_to_rs[svc]:
                        continue
                    out_obj = {
                        "resourceSpans": svc_to_rs[svc],
                    }
                    json.dump(out_obj, writers[svc], ensure_ascii=False)
                    writers[svc].write("\n")
    finally:
        for w in writers.values():
            try:
                w.close()
            except Exception:
                pass


# NOTE:
# Temporal S_temp is currently built from JTL + traces only.
# We intentionally do NOT split logs here anymore.


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Split OTEL all_traces.json into per-service files for "
            "acmeair/daytrader7/jpetstore/plantsbywebsphere."
        )
    )
    parser.add_argument(
        "--project-root",
        type=str,
        default=None,
        help="Optional project root. Defaults to repository root (2 levels up).",
    )
    args = parser.parse_args()

    root = (
        Path(args.project_root).resolve()
        if args.project_root is not None
        else Path(__file__).resolve().parents[2]
    )

    # Traces: split from all_traces.json
    trace_src = root / "data" / "processed" / "traces" / "all_traces.json"
    trace_out_dir = root / "data" / "processed" / "traces"
    if trace_src.is_file():
        print(f"[TRACE] splitting from {trace_src}")
        split_traces_by_service(trace_src, trace_out_dir, APPS)
    else:
        print(f"[TRACE] source not found: {trace_src}")

    print("Done.")


if __name__ == "__main__":
    main()