import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.cluster import SpectralClustering


ROOT = Path(__file__).resolve().parents[2]
FUSION_DIR = ROOT / "data" / "processed" / "fusion"


def run_spectral(system: str, n_clusters: int | None = None) -> dict[int, list[str]]:
    system = system.lower()
    s_final_path = FUSION_DIR / f"{system}_S_final.npy"
    class_order_path = FUSION_DIR / f"{system}_class_order.json"

    if not s_final_path.is_file():
        raise FileNotFoundError(f"S_final not found: {s_final_path}")
    if not class_order_path.is_file():
        raise FileNotFoundError(f"class_order not found: {class_order_path}")

    S_final = np.load(s_final_path)
    with class_order_path.open("r", encoding="utf-8") as f:
        class_order = json.load(f)

    n = S_final.shape[0]
    if len(class_order) != n:
        raise ValueError(
            f"class_order length {len(class_order)} does not match S_final shape {S_final.shape}"
        )

    if n_clusters is None:
        if system == "acmeair":
            n_clusters = 6
        else:
            n_clusters = max(2, int(np.sqrt(n)))

    model = SpectralClustering(
        n_clusters=n_clusters,
        affinity="precomputed",
        assign_labels="discretize",
        random_state=42,
    )
    labels = model.fit_predict(S_final)

    clusters: dict[int, list[str]] = defaultdict(list)
    for i, cid in enumerate(labels):
        clusters[int(cid)].append(class_order[i])

    return clusters


def main():
    parser = argparse.ArgumentParser(
        description="Run spectral clustering on S_final for a given system and print clusters."
    )
    parser.add_argument(
        "--system",
        required=True,
        choices=["acmeair", "daytrader", "jpetstore", "plants"],
        help="Target system name.",
    )
    parser.add_argument(
        "--n-clusters",
        type=int,
        default=None,
        help="Number of clusters. If omitted, use a heuristic based on system / N.",
    )
    args = parser.parse_args()

    clusters = run_spectral(args.system, n_clusters=args.n_clusters)
    # Ensure keys are plain ints for JSON serialization
    serializable = {int(k): v for k, v in clusters.items()}
    print(json.dumps(serializable, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
