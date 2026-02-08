import json
from pathlib import Path

SYSTEMS = ["acmeair", "daytrader", "plants", "jpetstore"]


def load_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def main():
    print(
        "System\tmode\tedge_min_weight\tbest_resolution\ttarget_range\tservices(after_merge)"
    )

    for s in SYSTEMS:
        p = Path("data/processed/fusion") / f"{s}_cac_params.json"
        if not p.exists():
            print(f"{s}\t(missing)\t-\t-\t-\t-")
            continue

        meta = load_json(p) or {}
        cac = meta.get("cac", {})
        target = meta.get("target_policy", {}).get("target_range")
        svc = (
            meta.get("cluster_size_stats", {})
            .get("CAC-Final", {})
            .get("after_merge", {})
            .get("n_clusters")
        )

        print(
            f"{s}\t{cac.get('mode')}\t{cac.get('edge_min_weight')}\t{cac.get('best_resolution')}\t{target}\t{svc}"
        )


if __name__ == "__main__":
    main()
