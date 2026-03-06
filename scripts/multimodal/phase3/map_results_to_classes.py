import json
import os

def map_clusters(system: str, method: str = "cac-final"):
    
    order_path = f"data/processed/fusion/{system}_class_order.json"
    with open(order_path, 'r', encoding='utf-8') as f:
        class_names = json.load(f)

    
    if method.lower() in {"baseline", "base"}:
        result_path = f"data/processed/fusion/{system}_baseline_partition.json"
    else:
        result_path = f"data/processed/fusion/{system}_cac-final_partition.json"

    if not os.path.exists(result_path):
        print(f"Partition file for '{system}' not found: {result_path}")
        return

    with open(result_path, 'r', encoding='utf-8') as f:
        partition = json.load(f)

    
    clusters = {}
    for k, cluster_id in partition.items():
        try:
            idx = int(k)
            class_name = class_names[idx]
        except Exception:
            class_name = k
        clusters.setdefault(cluster_id, []).append(class_name)

   
    print(f"\n{'='*20} {system.upper()} {method.upper()} Report {'='*20}")
    for cid, members in sorted(clusters.items(), key=lambda x: int(x[0]) if str(x[0]).isdigit() else str(x[0])):
        print(f"\n[Service #{cid}] ({len(members)} classes):")
        for member in sorted(members):
            print(f"  - {member}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('system', nargs='?', default='acmeair')
    parser.add_argument('--method', choices=['cac-final', 'baseline'], default='cac-final')
    args = parser.parse_args()

    map_clusters(args.system, args.method)
