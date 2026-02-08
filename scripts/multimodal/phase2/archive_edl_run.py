import argparse
import json
import os
import shutil
import time
from pathlib import Path


def _copy_if_exists(src: Path, dst: Path) -> bool:
    if not src.exists():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return True


def main():
    p = argparse.ArgumentParser(description="Archive an existing EDL run (legacy outputs) into a reproducible run folder.")
    p.add_argument("--run_tag", required=True, help="Run tag used as folder name under data/processed/edl/runs/<run_tag>/")
    p.add_argument("--note", default="", help="Optional note saved into archive_config.json")

    p.add_argument("--train_system", default="acmeair", help="Which system was used to train the EDL model (default: acmeair)")
    p.add_argument("--model", default="data/processed/edl/edl_model_acmeair_kl5_hardneg02.pt", help="Path to the trained model checkpoint")
    p.add_argument("--scaler", default="data/processed/edl/acmeair_scaler.pkl", help="Path to the scaler used")
    p.add_argument("--train_X", default="data/processed/edl/acmeair_X.npy", help="Path to training feature matrix")
    p.add_argument("--train_y", default="data/processed/edl/acmeair_y.npy", help="Path to training labels")

    p.add_argument(
        "--systems",
        nargs="*",
        default=["acmeair", "daytrader", "jpetstore", "plants"],
        help="Systems to archive uncertainty outputs for",
    )
    p.add_argument(
        "--u_suffix",
        default="kl5_hardneg02",
        help="Suffix used in legacy uncertainty file names: <system>_edl_uncertainty_<suffix>.npy",
    )

    args = p.parse_args()

    run_dir = Path("data/processed/edl/runs") / args.run_tag
    ckpt_dir = run_dir / "checkpoints"
    infer_dir = run_dir / "infer_outputs"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    infer_dir.mkdir(parents=True, exist_ok=True)

    # --- copy checkpoint + scaler + train data for provenance ---
    copied = {
        "model": _copy_if_exists(Path(args.model), ckpt_dir / Path(args.model).name),
        "scaler": _copy_if_exists(Path(args.scaler), run_dir / Path(args.scaler).name),
        "train_X": _copy_if_exists(Path(args.train_X), run_dir / Path(args.train_X).name),
        "train_y": _copy_if_exists(Path(args.train_y), run_dir / Path(args.train_y).name),
    }

    # --- copy U matrices and associated plots if present ---
    u_files = []
    for sys in args.systems:
        base = Path("data/processed/edl")
        npy = base / f"{sys}_edl_uncertainty_{args.u_suffix}.npy"
        png = base / f"{sys}_edl_uncertainty_{args.u_suffix}.png"
        hist = base / f"{sys}_edl_uncertainty_{args.u_suffix}_u_hist.png"

        u_entry = {
            "system": sys,
            "npy": str(npy),
            "copied_npy": _copy_if_exists(npy, infer_dir / npy.name),
            "copied_png": _copy_if_exists(png, infer_dir / png.name),
            "copied_hist": _copy_if_exists(hist, infer_dir / hist.name),
        }
        u_files.append(u_entry)

    archive_cfg = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "run_tag": args.run_tag,
        "run_dir": str(run_dir.resolve()),
        "note": args.note,
        "legacy": {
            "train_system": args.train_system,
            "u_suffix": args.u_suffix,
        },
        "paths": {
            "model": args.model,
            "scaler": args.scaler,
            "train_X": args.train_X,
            "train_y": args.train_y,
        },
        "copied": copied,
        "uncertainty_outputs": u_files,
    }

    with open(run_dir / "archive_config.json", "w", encoding="utf-8") as f:
        json.dump(archive_cfg, f, indent=2, ensure_ascii=False)

    print(f"[OK] Archived legacy EDL run into: {run_dir}")
    print("[INFO] Written:", str((run_dir / "archive_config.json").resolve()))


if __name__ == "__main__":
    main()
