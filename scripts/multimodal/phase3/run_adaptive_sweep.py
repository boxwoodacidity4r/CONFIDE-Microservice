"""Adaptive sigmoid+DPEP sweep runner for Phase 3.

This script is an experiment harness around
`scripts/multimodal/phase3/phase3_cac_evaluation.py`.

It sweeps sigmoid hyperparameters (alpha, dpep_cap) across systems and
records paper-friendly metrics into a CSV (incrementally, after each run).

JPetStore diagnosis mode
------------------------
JPetStore frequently fails when the allowed service-count range is too narrow.
For system == 'jpetstore', this runner can automatically widen the target range
(e.g., 2..20) to test whether the failure is due to target-range constraints.

Usage (PowerShell)
------------------
python scripts\multimodal\phase3\run_adaptive_sweep.py \
  --systems daytrader plants jpetstore \
  --alphas 15 20 \
  --cap_min 0.14 --cap_max 0.26 --cap_step 0.02
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from typing import Iterable, Optional

# Example line (may wrap visually in terminal):
# [GraphDiag] daytrader | nodes=34 | edge_min_weight=0.2 | Baseline edges=463 (density=0.8253) | CAC edges=100 (density=0.1783) | mode=sigmoid k=6.0 n_power=4.0 alpha=15.0 beta=0.0823
GRAPH_DIAG_RE = re.compile(
    r"\[GraphDiag\]\s+(?P<system>\w+)\s+\|\s+nodes=(?P<nodes>\d+)\s+\|\s+edge_min_weight=(?P<edge_min_weight>[0-9.eE+-]+)\s+\|\s+Baseline edges=(?P<baseline_edges>\d+)\s+\(density=(?P<baseline_density>[0-9.]+)\)\s+\|\s+CAC edges=(?P<cac_edges>\d+)\s+\(density=(?P<cac_density>[0-9.]+)\).+?alpha=(?P<alpha>[0-9.]+)\s+beta=(?P<beta>[0-9.eE+-]+)",
    re.MULTILINE,
)

# Example:
# [GraphPolicy] jpetstore: DPEP enabled | p=70 cap=0.200 => tau=0.200000 | final edge_min_weight=0.200000
DPEP_POLICY_RE = re.compile(
    r"\[GraphPolicy\]\s+(?P<system>\w+):\s+DPEP enabled\s+\|\s+p=(?P<p>[0-9.]+)\s+cap=(?P<cap>[0-9.]+)\s+=>\s+tau=(?P<tau>[0-9.eE+-]+)\s+\|\s+final edge_min_weight=(?P<edge_min_weight>[0-9.eE+-]+)",
    re.MULTILINE,
)

# Example scoreboard row:
# CAC-Final       | 0.0439   | 793      | 0.6390         | 12
CAC_FINAL_LINE_RE = re.compile(
    r"^CAC-Final\s+\|\s+(?P<q>[0-9.]+)\s+\|\s+(?P<ifn>\d+)\s+\|\s+(?P<ifn_ratio>[0-9.]+)\s+\|\s+(?P<services>\d+)\s*$",
    re.MULTILINE,
)

# Example summary row:
# jpetstore       | 12         | 0.0439     | PASS
FINAL_ROW_RE = re.compile(
    r"^(?P<system>\w+)\s+\|\s+(?P<services>\d+)\s+\|\s+(?P<q>[0-9.]+)\s+\|\s+(?P<result>PASS|FAIL)\s*$",
    re.MULTILINE,
)


@dataclass
class SweepResult:
    timestamp: str
    system: str
    alpha: float
    dpep_cap: float
    dpep_percentile: float
    target_min: Optional[int]
    target_max: Optional[int]

    # parsed metrics
    result: str
    services: Optional[int]
    q: Optional[float]
    ifn: Optional[int]
    ifn_ratio: Optional[float]
    nodes: Optional[int]
    baseline_density: Optional[float]
    cac_density: Optional[float]
    baseline_edges: Optional[int]
    cac_edges: Optional[int]
    beta: Optional[float]
    edge_min_weight_final: Optional[float]
    dpep_tau: Optional[float]

    exit_code: int


def frange(start: float, stop: float, step: float) -> Iterable[float]:
    x = start
    while x <= stop + 1e-12:
        yield float(round(x, 10))
        x += step


def run_one(
    system: str,
    alpha: float,
    dpep_cap: float,
    dpep_percentile: float,
    target_min: Optional[int],
    target_max: Optional[int],
) -> tuple[int, str]:
    cmd = [
        sys.executable,
        "scripts/multimodal/phase3/phase3_cac_evaluation.py",
        system,
        "--mode",
        "sigmoid",
        "--alpha",
        str(alpha),
        "--dpep_cap",
        str(dpep_cap),
        "--dpep_percentile",
        str(dpep_percentile),
    ]
    if target_min is not None:
        cmd += ["--target_min", str(target_min)]
    if target_max is not None:
        cmd += ["--target_max", str(target_max)]

    print("[RUN]", " ".join(cmd), flush=True)
    p = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    out, _ = p.communicate()
    return int(p.returncode or 0), out


def parse_output(system: str, output: str) -> dict:
    parsed: dict = {
        "nodes": None,
        "baseline_density": None,
        "cac_density": None,
        "baseline_edges": None,
        "cac_edges": None,
        "beta": None,
        "edge_min_weight_final": None,
        "dpep_tau": None,
        "services": None,
        "q": None,
        "result": "FAIL",
        "ifn": None,
        "ifn_ratio": None,
    }

    m = GRAPH_DIAG_RE.search(output)
    if m and m.group("system").lower() == system.lower():
        parsed.update(
            {
                "nodes": int(m.group("nodes")),
                "baseline_density": float(m.group("baseline_density")),
                "cac_density": float(m.group("cac_density")),
                "baseline_edges": int(m.group("baseline_edges")),
                "cac_edges": int(m.group("cac_edges")),
                "beta": float(m.group("beta")),
                "edge_min_weight_final": float(m.group("edge_min_weight")),
            }
        )

    m = DPEP_POLICY_RE.search(output)
    if m and m.group("system").lower() == system.lower():
        parsed["dpep_tau"] = float(m.group("tau"))
        parsed["edge_min_weight_final"] = float(m.group("edge_min_weight"))

    m = CAC_FINAL_LINE_RE.search(output)
    if m:
        parsed["q"] = float(m.group("q"))
        parsed["ifn"] = int(m.group("ifn"))
        parsed["ifn_ratio"] = float(m.group("ifn_ratio"))
        parsed["services"] = int(m.group("services"))

    m = FINAL_ROW_RE.search(output)
    if m and m.group("system").lower() == system.lower():
        parsed["result"] = str(m.group("result"))
        parsed["services"] = int(m.group("services"))
        parsed["q"] = float(m.group("q"))

    return parsed


def write_csv(path: str, rows: list[SweepResult]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "timestamp",
                "system",
                "alpha",
                "dpep_cap",
                "dpep_percentile",
                "target_min",
                "target_max",
                "result",
                "services",
                "q",
                "ifn",
                "ifn_ratio",
                "nodes",
                "baseline_density",
                "cac_density",
                "baseline_edges",
                "cac_edges",
                "beta",
                "edge_min_weight_final",
                "dpep_tau",
                "exit_code",
            ],
        )
        w.writeheader()
        for r in rows:
            w.writerow(r.__dict__)


def main() -> None:
    ap = argparse.ArgumentParser(description="Adaptive Phase3 sigmoid sweep + CSV summary")
    ap.add_argument(
        "--systems",
        nargs="*",
        default=["daytrader", "plants", "jpetstore"],
        help="Systems to sweep",
    )
    ap.add_argument("--alphas", nargs="*", type=float, default=[15.0, 20.0])

    ap.add_argument("--cap_min", type=float, default=0.15)
    ap.add_argument("--cap_max", type=float, default=0.25)
    ap.add_argument("--cap_step", type=float, default=0.02)
    ap.add_argument("--dpep_percentile", type=float, default=70.0)

    ap.add_argument("--jpetstore_target_min", type=int, default=2)
    ap.add_argument("--jpetstore_target_max", type=int, default=20)

    ap.add_argument(
        "--out_csv",
        type=str,
        default=None,
        help="Output CSV path (default: results/ablation/phase3_sigmoid_sweep_<timestamp>.csv)",
    )

    args = ap.parse_args()

    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_csv = args.out_csv or f"results/ablation/phase3_sigmoid_sweep_{ts}.csv"

    caps = list(frange(float(args.cap_min), float(args.cap_max), float(args.cap_step)))
    rows: list[SweepResult] = []

    for system in args.systems:
        for alpha in args.alphas:
            for cap in caps:
                target_min = None
                target_max = None
                if system.lower() == "jpetstore":
                    target_min = int(args.jpetstore_target_min)
                    target_max = int(args.jpetstore_target_max)

                exit_code, output = run_one(
                    system=system,
                    alpha=float(alpha),
                    dpep_cap=float(cap),
                    dpep_percentile=float(args.dpep_percentile),
                    target_min=target_min,
                    target_max=target_max,
                )

                parsed = parse_output(system, output)

                rows.append(
                    SweepResult(
                        timestamp=ts,
                        system=system,
                        alpha=float(alpha),
                        dpep_cap=float(cap),
                        dpep_percentile=float(args.dpep_percentile),
                        target_min=target_min,
                        target_max=target_max,
                        result=str(parsed["result"]),
                        services=parsed["services"],
                        q=parsed["q"],
                        ifn=parsed["ifn"],
                        ifn_ratio=parsed["ifn_ratio"],
                        nodes=parsed["nodes"],
                        baseline_density=parsed["baseline_density"],
                        cac_density=parsed["cac_density"],
                        baseline_edges=parsed["baseline_edges"],
                        cac_edges=parsed["cac_edges"],
                        beta=parsed["beta"],
                        edge_min_weight_final=parsed["edge_min_weight_final"],
                        dpep_tau=parsed["dpep_tau"],
                        exit_code=int(exit_code),
                    )
                )

                write_csv(out_csv, rows)

                print(
                    f"[DONE] {system:10s} alpha={alpha:<5} cap={cap:<5} => result={parsed['result']} services={parsed['services']} Q={parsed['q']} density={parsed['cac_density']}",
                    flush=True,
                )

    print(f"\n[OK] Sweep finished. CSV saved to: {out_csv}")


if __name__ == "__main__":
    main()
