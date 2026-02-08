"""Analyze JMeter JTL label frequencies and basic response-code stats.

Usage (PowerShell):
  python scripts/jmeter/analyze_jtl_labels.py --jtl results/jmeter/jpetstore_results.jtl

Outputs:
  - Top labels by count
  - ResponseCode distribution
  - For each label: total, success count, failure count, top response codes

This is intentionally dependency-minimal (uses only Python stdlib).
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--jtl", required=True, help="Path to JTL CSV")
    p.add_argument("--top", type=int, default=30, help="How many labels to show")
    return p.parse_args()


def _open_jtl_as_dict_reader(jtl_path: str):
    f = open(jtl_path, "r", encoding="utf-8", errors="replace", newline="")
    sample = f.read(4096)
    f.seek(0)
    has_header = sample.lower().lstrip().startswith("timestamp,")

    if has_header:
        return f, csv.DictReader(f)

    # Headerless JTL (common in our repo for some systems): use standard JMeter CSV order
    fieldnames = [
        "timeStamp",
        "elapsed",
        "label",
        "responseCode",
        "responseMessage",
        "threadName",
        "dataType",
        "success",
        "failureMessage",
        "bytes",
        "sentBytes",
        "grpThreads",
        "allThreads",
        "URL",
        "Latency",
        "IdleTime",
        "Connect",
    ]
    return f, csv.DictReader(f, fieldnames=fieldnames)


def main() -> None:
    args = parse_args()
    jtl_path = Path(args.jtl)

    label_counter: Counter[str] = Counter()
    code_counter: Counter[str] = Counter()

    total_by_label: Counter[str] = Counter()
    success_by_label: Counter[str] = Counter()
    failure_by_label: Counter[str] = Counter()
    codes_by_label: dict[str, Counter[str]] = defaultdict(Counter)

    f, reader = _open_jtl_as_dict_reader(str(jtl_path))
    with f:
        for row in reader:
            label = (row.get("label") or "").strip()
            code = (row.get("responseCode") or "").strip()
            success_raw = (row.get("success") or "").strip().lower()
            success = success_raw == "true"

            label_counter[label] += 1
            code_counter[code] += 1

            total_by_label[label] += 1
            if success:
                success_by_label[label] += 1
            else:
                failure_by_label[label] += 1
            codes_by_label[label][code] += 1

    print(f"JTL: {jtl_path}")
    print(f"Total samples: {sum(label_counter.values())}")
    print()

    print(f"Top {args.top} labels by count:")
    for label, cnt in label_counter.most_common(args.top):
        ok = success_by_label.get(label, 0)
        fail = failure_by_label.get(label, 0)
        print(f"  {label:30s}  total={cnt:5d}  ok={ok:5d}  fail={fail:5d}")
    print()

    print("Top responseCode values:")
    for code, cnt in code_counter.most_common(20):
        print(f"  {code:40s}  {cnt}")
    print()

    print("Per-label top response codes (top 3):")
    for label, cnt in label_counter.most_common(args.top):
        top_codes = ", ".join([f"{c}:{n}" for c, n in codes_by_label[label].most_common(3)])
        print(f"  {label:30s}  {top_codes}")


if __name__ == "__main__":
    main()
