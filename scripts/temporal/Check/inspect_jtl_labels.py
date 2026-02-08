"""Inspect JMeter JTL labels.

PowerShell-friendly helper to avoid heredoc.

Usage:
  python scripts/temporal/inspect_jtl_labels.py --jtl results/jmeter/plants_results.jtl --top 50 --only-success

Prints the most frequent labels and basic stats.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jtl", required=True)
    ap.add_argument("--top", type=int, default=30)
    ap.add_argument("--only-success", action="store_true")
    ap.add_argument("--contains", type=str, default="", help="Optional substring filter for labels")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    p = Path(args.jtl)
    df = pd.read_csv(p)

    if args.only_success and "success" in df.columns:
        df = df[df["success"].astype(str).str.lower() == "true"]

    if "label" not in df.columns:
        raise SystemExit("JTL missing 'label' column")

    labels = df["label"].dropna().astype(str)
    if args.contains:
        labels = labels[labels.str.contains(args.contains, case=False, na=False)]

    print(f"JTL: {p}")
    print(f"rows: {len(df)}")
    print(f"labels_total: {len(labels)}")
    print(f"labels_unique: {labels.nunique()}")
    print("\nTop labels:")
    print(labels.value_counts().head(int(args.top)).to_string())


if __name__ == "__main__":
    main()
