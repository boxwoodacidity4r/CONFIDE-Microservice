import csv
import json
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
PHASE3 = ROOT / "scripts" / "multimodal" / "phase3" / "phase3_cac_evaluation.py"

# Golden settings (edit here)
GOLDEN = {
    # system: (target_min, target_max, dpep_cap)
    "acmeair": (5, 5, 0.14),
    "daytrader": (5, 8, 0.18),
    "plants": (3, 6, 0.22),
    "jpetstore": (4, 8, 0.14),
}

COMMON = ["--mode", "sigmoid", "--alpha", "15", "--merge_small_clusters", "--min_cluster_size", "3"]

# Parse the metrics block printed by phase3_cac_evaluation.py
_METRIC_RE = re.compile(
    r"^(Baseline|CAC-Final)\s+\|\s+([0-9.]+)\s+\|\s+(\d+)\s+\|\s+([0-9.]+)\s+\|\s+(\d+)\s*$"
)


def run_one(system: str, tmin: int, tmax: int, cap: float) -> dict:
    cmd = [
        sys.executable,
        str(PHASE3),
        system,
        "--dpep_cap",
        str(cap),
        "--target_min",
        str(tmin),
        "--target_max",
        str(tmax),
        *COMMON,
    ]

    p = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)
    out = (p.stdout or "") + "\n" + (p.stderr or "")

    if p.returncode != 0:
        return {
            "system": system,
            "error": f"returncode={p.returncode}",
            "stdout_tail": "\n".join(out.splitlines()[-50:]),
        }

    metrics = {}
    for line in out.splitlines():
        m = _METRIC_RE.match(line.strip())
        if not m:
            continue
        name, q, ifn, ifn_ratio, services = m.groups()
        metrics[name] = {
            "Q": float(q),
            "IFN": int(ifn),
            "IFN_Ratio": float(ifn_ratio),
            "Services": int(services),
        }

    return {
        "system": system,
        "target_min": int(tmin),
        "target_max": int(tmax),
        "dpep_cap": float(cap),
        "mode": "sigmoid",
        "alpha": 15.0,
        "merge_small_clusters": True,
        "min_cluster_size": 3,
        "baseline": metrics.get("Baseline"),
        "cac": metrics.get("CAC-Final"),
        "stdout_tail": "\n".join(out.splitlines()[-60:]),
    }


def main():
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = ROOT / "results" / "ablation"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for system, (tmin, tmax, cap) in GOLDEN.items():
        print(f"[FinalEval] Running {system} (target={tmin}-{tmax}, cap={cap})", flush=True)
        rows.append(run_one(system, tmin, tmax, cap))

    # Write CSV
    csv_path = out_dir / f"phase3_best_all_systems_{ts}.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "system",
                "mode",
                "alpha",
                "dpep_cap",
                "target_min",
                "target_max",
                "baseline_services",
                "baseline_q",
                "baseline_ifn",
                "baseline_ifn_ratio",
                "cac_services",
                "cac_q",
                "cac_ifn",
                "cac_ifn_ratio",
                "error",
            ]
        )
        for r in rows:
            b = r.get("baseline") or {}
            c = r.get("cac") or {}
            w.writerow(
                [
                    r.get("system"),
                    r.get("mode"),
                    r.get("alpha"),
                    r.get("dpep_cap"),
                    r.get("target_min"),
                    r.get("target_max"),
                    b.get("Services"),
                    b.get("Q"),
                    b.get("IFN"),
                    b.get("IFN_Ratio"),
                    c.get("Services"),
                    c.get("Q"),
                    c.get("IFN"),
                    c.get("IFN_Ratio"),
                    r.get("error"),
                ]
            )

    # Write markdown table (paper-ready)
    md_path = out_dir / f"phase3_best_all_systems_{ts}.md"
    with md_path.open("w", encoding="utf-8") as f:
        f.write("# Phase 3 Best Points (All Systems)\n\n")
        f.write(f"Generated: {ts}\n\n")
        f.write(
            "| System | Baseline Services | Baseline Q | Baseline IFN_Ratio | CAC Services | CAC Q | CAC IFN_Ratio | dpep_cap | target_range |\n"
        )
        f.write("|---|---:|---:|---:|---:|---:|---:|---:|---|\n")
        for r in rows:
            b = r.get("baseline") or {}
            c = r.get("cac") or {}
            tr = f"[{r.get('target_min')},{r.get('target_max')}]"
            f.write(
                "| {sys} | {bs} | {bq:.4f} | {br:.4f} | {cs} | {cq:.4f} | {cr:.4f} | {cap:.2f} | {tr} |\n".format(
                    sys=r.get("system"),
                    bs=b.get("Services", "-"),
                    bq=float(b.get("Q", 0.0) or 0.0),
                    br=float(b.get("IFN_Ratio", 0.0) or 0.0),
                    cs=c.get("Services", "-"),
                    cq=float(c.get("Q", 0.0) or 0.0),
                    cr=float(c.get("IFN_Ratio", 0.0) or 0.0),
                    cap=float(r.get("dpep_cap", 0.0) or 0.0),
                    tr=tr,
                )
            )

        f.write("\n## Raw stdout tails (debug)\n")
        for r in rows:
            f.write(f"\n### {r.get('system')}\n\n")
            f.write("```\n")
            f.write(r.get("stdout_tail", ""))
            f.write("\n```\n")

    print(f"[FinalEval] CSV saved: {csv_path}")
    print(f"[FinalEval] MD saved:  {md_path}")


if __name__ == "__main__":
    main()
