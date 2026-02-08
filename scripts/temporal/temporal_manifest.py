import json
import os
from pathlib import Path
import argparse
from typing import List, Optional


# Canonical service names as used in OTEL service.name and per-app trace filenames
APPS = ["acmeair", "daytrader7", "jpetstore", "plantsbywebsphere"]


def count_lines(path: Path) -> int:
    if not path.is_file():
        return 0
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        return sum(1 for _ in f)


def build_manifest(project_root: Path, apps: Optional[List[str]] = None) -> dict:
    """Build a manifest summarizing temporal raw data availability for each app.

    Current temporal modality uses:
      - per-app trace json (data/processed/traces/{app}.json)
      - JMeter results (results/jmeter/{app}_results.jtl)

    Logs are intentionally not part of the temporal matrix construction anymore.
    """
    if apps is None:
        apps = APPS

    data: dict[str, dict] = {}

    traces_root = project_root / "data" / "processed" / "traces"
    jmeter_root = project_root / "results" / "jmeter"

    for app in apps:
        trace_file = traces_root / f"{app}.json"
        jmeter_file = jmeter_root / f"{app}_results.jtl"

        entry = {
            "trace_file": str(trace_file.as_posix()),
            "jmeter_file": str(jmeter_file.as_posix()),
            "trace_exists": trace_file.is_file(),
            "jmeter_exists": jmeter_file.is_file(),
            "trace_bytes": os.path.getsize(trace_file) if trace_file.is_file() else 0,
            "jmeter_bytes": os.path.getsize(jmeter_file) if jmeter_file.is_file() else 0,
            "jmeter_lines": max(count_lines(jmeter_file) - 1, 0) if jmeter_file.is_file() else 0,
        }

        data[app] = entry

    return data


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build temporal data manifest for all systems (traces/logs/JMeter).",
    )
    parser.add_argument(
        "--project-root",
        type=str,
        default=None,
        help="Optional project root. Defaults to repository root (2 levels up).",
    )
    args = parser.parse_args()

    project_root = (
        Path(args.project_root).resolve()
        if args.project_root is not None
        else Path(__file__).resolve().parents[2]
    )

    manifest_dir = project_root / "data" / "processed" / "temporal"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = manifest_dir / "temporal_manifest.json"

    manifest = build_manifest(project_root)

    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(f"Temporal manifest written to: {manifest_path}")
    for app, meta in manifest.items():
        print(
            f"[{app}] traces={meta['trace_exists']}({meta['trace_bytes']}B), "
            f"jtl={meta['jmeter_exists']}({meta['jmeter_lines']} samples)"
        )


if __name__ == "__main__":
    main()
