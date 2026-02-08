import json
import os

def map_clusters(system: str, method: str = "cac-final"):
    # 1. 加载类名列表（索引->类名）
    order_path = f"data/processed/fusion/{system}_class_order.json"
    with open(order_path, 'r', encoding='utf-8') as f:
        class_names = json.load(f)

    # 2. 加载 Phase 3 的聚类划分结果（当前输出为 *_baseline_partition.json / *_cac-final_partition.json）
    if method.lower() in {"baseline", "base"}:
        result_path = f"data/processed/fusion/{system}_baseline_partition.json"
    else:
        result_path = f"data/processed/fusion/{system}_cac-final_partition.json"

    if not os.path.exists(result_path):
        print(f"未找到 {system} 的划分结果文件: {result_path}")
        return

    with open(result_path, 'r', encoding='utf-8') as f:
        partition = json.load(f)

    # 3. 按 Cluster ID 组织类名（兼容 key 为 index 或 key 直接为类名）
    clusters = {}
    for k, cluster_id in partition.items():
        try:
            idx = int(k)
            class_name = class_names[idx]
        except Exception:
            class_name = k
        clusters.setdefault(cluster_id, []).append(class_name)

    # 4. 打印报告
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
