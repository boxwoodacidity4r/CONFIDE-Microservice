"""Inspect JMeter JTL for TransactionController labels and thread distribution.

This script is intentionally PowerShell-friendly (no heredoc required).

Example:
  python scripts/temporal/inspect_jtl_tx_threads.py --jtl results/jmeter/plants_results.jtl
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Inspect JTL transaction labels per thread")
    p.add_argument("--jtl", required=True, help="Path to JTL (CSV)")
    p.add_argument("--tx-prefix", default="T_", help="Transaction label prefix (default: T_)")
    p.add_argument("--top", type=int, default=15, help="Top N threads/labels to show")
    p.add_argument(
        "--drop-label-regex",
        default=r"INTERNAL|DIAG",
        help="Regex to drop non-business labels (default: INTERNAL|DIAG)",
    )
    p.add_argument("--keep-only-success", action="store_true", help="Keep only success==true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    jtl_path = Path(args.jtl)
    df = pd.read_csv(jtl_path)

    if args.keep_only_success and "success" in df.columns:
        df = df[df["success"].astype(str).str.lower() == "true"]

    if args.drop_label_regex and "label" in df.columns:
        df = df[~df["label"].astype(str).str.contains(args.drop_label_regex, case=False, na=False)]

    print(f"JTL: {jtl_path}")
    print(f"Rows after filters: {len(df)}")

    tx_prefix = str(args.tx_prefix)
    tx_rx = re.compile(rf"^{re.escape(tx_prefix)}", re.IGNORECASE)

    df["_is_tx"] = df.get("label", "").astype(str).str.match(tx_rx, na=False)

    tx = df[df["_is_tx"]]
    print(f"Tx rows: {len(tx)}")
    print(f"Unique tx labels: {tx['label'].nunique() if len(tx) else 0}")

    if "threadName" in df.columns:
        print(f"Threads total: {df['threadName'].nunique()}")
        print(f"Threads with tx: {tx['threadName'].nunique() if len(tx) else 0}")

        if len(tx):
            print("\nTop tx labels:")
            print(tx["label"].value_counts().head(args.top).to_string())
            print("\nTop threads by tx count:")
            print(tx["threadName"].value_counts().head(args.top).to_string())

    # Also show a few sample tx rows to detect whitespace/trailing characters
    if len(tx):
        cols = [c for c in ["timeStamp", "elapsed", "label", "threadName", "success", "responseCode"] if c in tx.columns]
        print("\nSample tx rows:")
        print(tx[cols].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
