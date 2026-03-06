import argparse
import json
import os

import numpy as np


def _resolve_partition_path(partition_path: str, system: str) -> str:
    """Resolve partition path.

    This repo's Phase3 outputs standardized names:
      - data/processed/fusion/<system>_baseline_partition.json
      - data/processed/fusion/<system>_cac-final_partition.json

    Older docs sometimes referenced <system>_partition.json.
    """

    # If the user provided an existing path, keep it.
    if partition_path and os.path.exists(partition_path):
        return partition_path

    # Try common standardized outputs.
    candidates = []
    if system:
        candidates.extend(
            [
                os.path.join("data", "processed", "fusion", f"{system}_cac-final_partition.json"),
                os.path.join("data", "processed", "fusion", f"{system}_baseline_partition.json"),
            ]
        )

    # If user passed something like data/processed/fusion/<system>_partition.json, try swapping suffix.
    if partition_path:
        base = os.path.basename(partition_path)
        if base.endswith("_partition.json") and system:
            candidates.extend(
                [
                    partition_path.replace(f"{system}_partition.json", f"{system}_cac-final_partition.json"),
                    partition_path.replace(f"{system}_partition.json", f"{system}_baseline_partition.json"),
                ]
            )

    for p in candidates:
        if p and os.path.exists(p):
            print(f"[Info] Resolved partition file -> {p}")
            return p

    msg = (
        f"Partition file not found. Given: {partition_path}\n"
        f"Tried: {candidates}\n"
        "Hint: run Phase3 CAC evaluation to generate partition files, or pass an existing partition json."
    )
    raise FileNotFoundError(msg)


def load_data(u_matrix_path, partition_path, class_order_path, *, system: str | None = None):
    # 1. Load U matrix
    u_matrix = np.load(u_matrix_path)
    print(f"[Info] Loaded U matrix shape: {u_matrix.shape}")

    # 2. Load class order (order used during inference)
    with open(class_order_path, "r", encoding="utf-8") as f:
        class_list = json.load(f)

    # 3. Load clustering result (ClassName -> ServiceID)
    partition_path = _resolve_partition_path(partition_path, system=system or "")
    with open(partition_path, "r", encoding="utf-8") as f:
        partition = json.load(f)

    return u_matrix, class_list, partition


def reorder_matrix(u_matrix, class_list, partition):
    """Core logic: reorder matrix rows/cols by Service ID."""

    meta_data = []
    for idx, class_name in enumerate(class_list):
        s_id = partition.get(class_name, -1)
        if isinstance(s_id, list):
            s_id = s_id[0]
        meta_data.append({"service_id": s_id, "class_name": class_name, "original_idx": idx})

    sorted_meta = sorted(meta_data, key=lambda x: x["service_id"])

    new_indices = [item["original_idx"] for item in sorted_meta]

    reordered_u = u_matrix[new_indices, :][:, new_indices]

    boundaries = []
    if sorted_meta:
        current_service = sorted_meta[0]["service_id"]
        for i, item in enumerate(sorted_meta):
            if item["service_id"] != current_service:
                boundaries.append(i)
                current_service = item["service_id"]

    return reordered_u, boundaries, sorted_meta


def plot_heatmap(u_matrix, boundaries, system_name, output_path, vmax=0.5):
    # lazy import so this script doesn't force matplotlib/seaborn for non-visual pipelines
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    plt.figure(figsize=(12, 10))

    ax = sns.heatmap(
        u_matrix,
        cmap="RdBu_r",
        vmin=0,
        vmax=float(vmax),
        cbar_kws={"label": "Uncertainty (U)"},
        xticklabels=False,
        yticklabels=False,
    )

    for b in boundaries:
        plt.axhline(b, color="white", linewidth=1, linestyle="--")
        plt.axvline(b, color="white", linewidth=1, linestyle="--")

    plt.title(f"Reordered Uncertainty Matrix: {system_name}\n(Grouped by CAC Clusters)", fontsize=15)
    plt.xlabel("Classes (Grouped by Service)", fontsize=12)
    plt.ylabel("Classes (Grouped by Service)", fontsize=12)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[Success] Heatmap saved to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("system", type=str, help="System name (e.g., daytrader)")
    parser.add_argument("--u_path", type=str, required=True, help="Path to .npy U matrix")
    parser.add_argument("--part_path", type=str, required=True, help="Path to partition.json")
    parser.add_argument("--order_path", type=str, required=True, help="Path to class_order.json")
    parser.add_argument("--out_dir", type=str, default="data/processed/visualization")
    parser.add_argument("--vmax", type=float, default=0.5, help="Upper bound for colormap scale")

    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    out_file = os.path.join(args.out_dir, f"{args.system}_reordered_u_heatmap.png")

    u, cls_list, part = load_data(args.u_path, args.part_path, args.order_path, system=args.system)
    u_new, bounds, _meta = reorder_matrix(u, cls_list, part)
    plot_heatmap(u_new, bounds, args.system, out_file, vmax=float(args.vmax))
