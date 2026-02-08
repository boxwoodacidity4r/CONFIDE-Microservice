"""Phase 4: Evaluation (GT alignment + architecture-oriented metrics).

This file is moved from `scripts/multimodal/phase3/evaluate_partition_f1.py`.
It evaluates a predicted partition against ground-truth using pairwise F1 and MoJoSim.
Optionally, when a dependency matrix is provided, it reports architecture quality metrics.

Usage:
  python scripts/multimodal/phase4/evaluate_partition_f1.py --gt <gt.json> --pred <pred.json>
  python scripts/multimodal/phase4/evaluate_partition_f1.py --gt <gt.json> --pred <pred.json> --dep <dep.json>

Notes:
  - JSON formats:
      gt:   {"ClassName": service_id}
      pred: {"ClassName": service_id}
      dep:  {"ClassA": {"ClassB": weight}}
  - Class names are treated as raw keys; this script also strips trailing `.java` in dep keys
    to reduce matching issues.
"""

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import precision_recall_fscore_support


def _normalize_class_key(k):
    """Normalize class keys across GT / pred / dep.

    - Trim whitespace
    - Strip trailing '.java'
    - Convert digit-like keys (e.g., 0, '0', '000') to canonical string form: '0'

    This is critical because Phase3 partitions may be emitted using class-index keys
    (strings like '0', '1', ...), while some GT/dep sources use class names.
    """
    if k is None:
        return None
    s = str(k).strip()
    if s.endswith(".java"):
        s = s[: -len(".java")].strip()
    # canonicalize pure-numeric ids
    if s.isdigit():
        s = str(int(s))
    return s


def _normalize_mapping(d):
    if not isinstance(d, dict):
        return d
    out = {}
    for k, v in d.items():
        nk = _normalize_class_key(k)
        out[nk] = v
    return out


def calculate_f1(gt_dict, pred_dict):
    """Pairwise F1 by converting clustering into a binary classification over class pairs."""
    common_classes = list(set(gt_dict.keys()) & set(pred_dict.keys()))
    n = len(common_classes)
    if n < 2:
        return 0.0, 0.0, 0.0

    gt_pairs = []
    pred_pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            class_i = common_classes[i]
            class_j = common_classes[j]
            gt_same = 1 if gt_dict[class_i] == gt_dict[class_j] else 0
            gt_pairs.append(gt_same)
            pred_same = 1 if pred_dict[class_i] == pred_dict[class_j] else 0
            pred_pairs.append(pred_same)

    if not gt_pairs:
        return 0.0, 0.0, 0.0

    p, r, f1, _ = precision_recall_fscore_support(gt_pairs, pred_pairs, average="binary", zero_division=0)
    return float(p), float(r), float(f1)


def calculate_bcubed(gt_dict, pred_dict):
    """BCubed precision/recall/f1.

    定义：对每个元素 i
      - P_i = |C_pred(i) ∩ C_gt(i)| / |C_pred(i)|
      - R_i = |C_pred(i) ∩ C_gt(i)| / |C_gt(i)|
    全局为各元素平均。

    参考：Amigó et al., "A comparison of extrinsic clustering evaluation metrics..." (BCubed)
    """
    common = list(set(gt_dict.keys()) & set(pred_dict.keys()))
    if not common:
        return 0.0, 0.0, 0.0

    gt_groups = {}
    pred_groups = {}
    for c in common:
        gt_groups.setdefault(gt_dict[c], set()).add(c)
        pred_groups.setdefault(pred_dict[c], set()).add(c)

    p_sum = 0.0
    r_sum = 0.0
    for c in common:
        inter = pred_groups[pred_dict[c]] & gt_groups[gt_dict[c]]
        p_i = len(inter) / max(1, len(pred_groups[pred_dict[c]]))
        r_i = len(inter) / max(1, len(gt_groups[gt_dict[c]]))
        p_sum += p_i
        r_sum += r_i

    p = p_sum / len(common)
    r = r_sum / len(common)
    f1 = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0
    return float(p), float(r), float(f1)


def calculate_architecture_metrics(pred_dict, dependency_matrix):
    """Compute IFN and NED."""
    # IFN: cross-service weighted calls
    ifn = 0
    for class_a, service_a in pred_dict.items():
        for class_b, service_b in pred_dict.items():
            if service_a != service_b:
                if class_a in dependency_matrix and class_b in dependency_matrix[class_a]:
                    ifn += dependency_matrix[class_a][class_b]

    # NED: normalized std of service sizes
    service_sizes = {}
    for s_id in pred_dict.values():
        service_sizes[s_id] = service_sizes.get(s_id, 0) + 1
    sizes = list(service_sizes.values())
    avg_size = sum(sizes) / len(sizes)
    ned = np.std(sizes) / avg_size if avg_size > 0 else 0.0
    return ifn, ned


def calculate_advanced_metrics(pred_dict, dep_matrix):
    """Compute SM and ICP."""
    services = {}
    for c, s in pred_dict.items():
        services.setdefault(s, []).append(c)

    # normalize dep keys to reduce mismatches
    clean_dep = {}
    for k, v in dep_matrix.items():
        clean_k = k.replace(".java", "").strip()
        clean_v = {nk.replace(".java", "").strip(): nw for nk, nw in v.items()}
        clean_dep[clean_k] = clean_v

    n_s = len(services)
    if n_s <= 1:
        return 0.0, 0.0

    cohesion_list = []
    coupling_list = []

    for s_id, class_list in services.items():
        # cohesion (internal)
        n = len(class_list)
        internal_weight = 0
        if n > 1:
            for c1 in class_list:
                for c2 in class_list:
                    if c1 != c2 and c1 in clean_dep and c2 in clean_dep[c1]:
                        internal_weight += clean_dep[c1][c2]
            cohesion = internal_weight / (n * (n - 1))
        else:
            cohesion = 0.0
        cohesion_list.append(cohesion)

        # coupling (external)
        external_weight = 0
        other_classes_count = len(pred_dict) - n
        if other_classes_count > 0:
            for c1 in class_list:
                for c2_other, s2_id in pred_dict.items():
                    if s2_id != s_id and c1 in clean_dep and c2_other in clean_dep[c1]:
                        external_weight += clean_dep[c1][c2_other]
            coupling = external_weight / (n * other_classes_count)
        else:
            coupling = 0.0
        coupling_list.append(coupling)

    # SM
    avg_cohesion = sum(cohesion_list) / n_s
    avg_coupling = sum(coupling_list) / n_s
    sm = avg_cohesion - avg_coupling

    # ICP
    total_links = 0
    cross_links = 0
    for c1, neighbors in clean_dep.items():
        if c1 not in pred_dict:
            continue
        for c2, _weight in neighbors.items():
            if c2 in pred_dict:
                total_links += 1
                if pred_dict[c1] != pred_dict[c2]:
                    cross_links += 1
    icp = cross_links / total_links if total_links > 0 else 0.0

    return sm, icp


def calculate_mojo(gt_dict, pred_dict):
    """MoJo distance using greedy maximum overlap matching."""
    common_classes = list(set(gt_dict.keys()) & set(pred_dict.keys()))
    n = len(common_classes)
    if n == 0:
        return 0, 0

    gt_clusters = {}
    for c in common_classes:
        gt_clusters.setdefault(gt_dict[c], set()).add(c)

    pred_clusters = {}
    for c in common_classes:
        pred_clusters.setdefault(pred_dict[c], set()).add(c)

    pred_keys = list(pred_clusters.keys())
    gt_keys = list(gt_clusters.keys())

    max_overlap_sum = 0
    used_gt = set()
    sorted_pred_keys = sorted(pred_keys, key=lambda k: len(pred_clusters[k]), reverse=True)
    for pk in sorted_pred_keys:
        max_overlap = 0
        best_gt = None
        for gk in gt_keys:
            if gk in used_gt:
                continue
            overlap = len(pred_clusters[pk] & gt_clusters[gk])
            if overlap > max_overlap:
                max_overlap = overlap
                best_gt = gk
        if best_gt is not None:
            max_overlap_sum += max_overlap
            used_gt.add(best_gt)

    mojo_dist = n - max_overlap_sum
    return mojo_dist, n


def calculate_mojosim(mojo_dist, n):
    if n == 0:
        return 0.0
    return (1 - mojo_dist / n) * 100


def random_baseline(gt_dict, k=None, seed=42):
    random.seed(seed)
    class_names = list(gt_dict.keys())
    if k is None:
        k = len(set(gt_dict.values()))
    return {c: random.randint(0, k - 1) for c in class_names}


def monolith_baseline(gt_dict):
    return {c: 0 for c in gt_dict.keys()}


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _remap_pred_with_class_order(pred_dict, class_order_list):
    """Remap a pred partition emitted as {"0": sid, "1": sid, ...} to {className: sid}.

    Phase3 may output partitions keyed by class index (as strings). Phase4 GT is keyed by class name.
    With class_order (same ordering used to build matrices), we can remap deterministically.
    """
    if not isinstance(pred_dict, dict) or not isinstance(class_order_list, list):
        return pred_dict

    # check if pred looks like index-keyed
    keys = list(pred_dict.keys())
    if not keys:
        return pred_dict
    if not all(str(k).strip().isdigit() for k in keys):
        return pred_dict

    out = {}
    for k, sid in pred_dict.items():
        idx = int(str(k).strip())
        if 0 <= idx < len(class_order_list):
            out[str(class_order_list[idx]).strip()] = sid
    return out


def check_dep_format(dep_matrix):
    if not isinstance(dep_matrix, dict):
        raise ValueError("依赖矩阵应为字典格式！")
    for k, v in dep_matrix.items():
        if not isinstance(v, dict):
            raise ValueError(f"依赖矩阵 {k} 的值应为字典格式！")


def main():
    parser = argparse.ArgumentParser(description="Evaluate partition vs ground truth")
    parser.add_argument("--gt", type=str, required=True, help="Ground truth json (class_name -> service_id)")
    parser.add_argument("--pred", type=str, required=True, help="Predicted partition json (class_name -> service_id)")
    parser.add_argument("--dep", type=str, required=False, help="Dependency matrix json (classA: {classB: weight})")
    parser.add_argument(
        "--class_order",
        type=str,
        default=None,
        help="Optional class_order.json (list of class names). Required if --pred is index-keyed (e.g., {'0':sid}).",
    )
    parser.add_argument("--sanity", action="store_true", help="输出Random/Monolith极端基准")
    parser.add_argument(
        "--out_json",
        type=str,
        default=None,
        help="Optional: write metrics as JSON to this path (for reproducible tables; avoids regex parsing).",
    )
    # 新增：策略A（只评估 GT>=0 的 business class universe）
    parser.add_argument(
        "--filter_gt_negative",
        action="store_true",
        default=True,
        help="(默认开启) 过滤掉 GT 中 label<0 的类，并同步过滤 pred/dep，只在 GT>=0 上评估",
    )
    parser.add_argument(
        "--no_filter_gt_negative",
        action="store_false",
        dest="filter_gt_negative",
        help="关闭过滤：把 GT<0 的类也纳入评估（一般不推荐）",
    )
    args = parser.parse_args()

    gt_dict = _normalize_mapping(load_json(args.gt))
    pred_raw = load_json(args.pred)

    if args.class_order:
        class_order_list = load_json(args.class_order)
        pred_raw = _remap_pred_with_class_order(pred_raw, class_order_list)

    pred_dict = _normalize_mapping(pred_raw)

    # --- 策略A：过滤 GT=-1 等 out-of-scope 类 ---
    if bool(args.filter_gt_negative):
        valid_classes = {c for c, sid in gt_dict.items() if isinstance(sid, (int, float)) and sid >= 0}
        gt_dict = {c: sid for c, sid in gt_dict.items() if c in valid_classes}
        pred_dict = {c: sid for c, sid in pred_dict.items() if c in valid_classes}
        print(
            f"[EvalPolicy] filter_gt_negative=ON | valid_classes={len(valid_classes)} | gt_used={len(gt_dict)} | pred_used={len(pred_dict)}"
        )

        if len(gt_dict) > 0 and len(pred_dict) == 0:
            gt_sample = sorted(list(gt_dict.keys()))[:5]
            pred_sample = sorted(list(pred_dict.keys()))[:5]
            print(
                "[EvalDiag] pred_used=0. This usually means GT keys and pred keys don't match. "
                "If your pred JSON is index-keyed, pass --class_order data/processed/fusion/<system>_class_order.json"
            )
            print(f"[EvalDiag] sample_gt_keys={gt_sample}")
            print(f"[EvalDiag] sample_pred_keys={pred_sample}")

    start_time = time.time()
    p, r, f1 = calculate_f1(gt_dict, pred_dict)
    bc_p, bc_r, bc_f1 = calculate_bcubed(gt_dict, pred_dict)
    mojo_dist, n = calculate_mojo(gt_dict, pred_dict)
    mojosim = calculate_mojosim(mojo_dist, n)
    elapsed = time.time() - start_time

    gt_k = len(set(gt_dict.values()))
    pred_k = len(set(pred_dict.values()))
    k_diff = abs(pred_k - gt_k)

    metrics = {
        "f1": float(f1),
        "precision": float(p),
        "recall": float(r),
        "bcubed_f1": float(bc_f1),
        "bcubed_precision": float(bc_p),
        "bcubed_recall": float(bc_r),
        "mojosim": float(mojosim),
        "gt_k": int(gt_k),
        "pred_k": int(pred_k),
        "k_diff": int(k_diff),
        "n_classes": int(n),
        "elapsed_sec": float(elapsed),
        "filter_gt_negative": bool(args.filter_gt_negative),
    }

    print("-" * 40)
    print("视角 A：语义准确性（与专家对齐）")
    print(f"  Pairwise F1: {f1:.4f} (越高越好)")
    print(f"  BCubed F1:  {bc_f1:.4f} (越高越好)")
    print(f"  MoJoSim: {mojosim:.2f}% (越高越好)")
    print(f"  Pairwise Precision: {p:.4f}, Recall: {r:.4f}")
    print(f"  BCubed Precision:  {bc_p:.4f}, Recall: {bc_r:.4f}")
    print(f"  GT 服务数: {gt_k}, 预测服务数: {pred_k}, K-Diff: {k_diff}")
    print(f"  系统规模 (Classes): {n}")
    print(f"  评估耗时: {elapsed:.2f} 秒")

    if args.dep:
        dep_matrix = load_json(args.dep)
        check_dep_format(dep_matrix)

        # normalize dep keys to reduce mismatches
        dep_matrix = {
            _normalize_class_key(a): {_normalize_class_key(b): w for b, w in nbrs.items()}
            for a, nbrs in dep_matrix.items()
        }

        # 如果开启了策略A过滤，则对 dep_matrix 也做同步裁剪，避免把 out-of-scope 依赖计入 IFN/SM/ICP。
        if bool(args.filter_gt_negative):
            used = set(pred_dict.keys())
            dep_matrix = {
                a: {b: w for b, w in nbrs.items() if b in used}
                for a, nbrs in dep_matrix.items()
                if a in used
            }

        ifn, ned = calculate_architecture_metrics(pred_dict, dep_matrix)
        sm, icp = calculate_advanced_metrics(pred_dict, dep_matrix)

        metrics.update({
            "ifn": float(ifn),
            "ned": float(ned),
            "sm": float(sm),
            "icp": float(icp),
        })

        print("视角 B：架构演进质量（工程落地价值）")
        print(f"  IFN (跨服务调用数): {ifn:.2f} (越低越好)")
        print(f"  SM (结构模块度): {sm:.4f} (越高越好)")
        print(f"  ICP (接口传播代价): {icp:.4f} (越低越好)")
        print(f"  NED (分布均匀度): {ned:.4f}")
        if ned > 1.0:
            print("[警告] NED > 1.0，存在超级服务/上帝类，建议结合不确定性热力图分析。")

    if args.sanity:
        print("\n[Sanity Check: Random Baseline]")
        rand_pred = random_baseline(gt_dict, k=gt_k)
        p_r, r_r, f1_r = calculate_f1(gt_dict, rand_pred)
        mojo_r, _ = calculate_mojo(gt_dict, rand_pred)
        mojosim_r = calculate_mojosim(mojo_r, n)
        print(f"Random F1: {f1_r:.4f}, MoJoSim: {mojosim_r:.2f}%")

        print("[Sanity Check: Monolith Baseline]")
        mono_pred = monolith_baseline(gt_dict)
        p_m, r_m, f1_m = calculate_f1(gt_dict, mono_pred)
        mojo_m, _ = calculate_mojo(gt_dict, mono_pred)
        mojosim_m = calculate_mojosim(mojo_m, n)
        print(f"Monolith F1: {f1_m:.4f}, MoJoSim: {mojosim_m:.2f}%")

    if args.out_json:
        out_path = args.out_json
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)
        print(f"[OK] Saved metrics json -> {out_path}")


if __name__ == "__main__":
    main()
