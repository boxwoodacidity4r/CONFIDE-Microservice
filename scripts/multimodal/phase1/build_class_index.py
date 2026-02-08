# d:/multimodal_microservice_extraction/scripts/multimodal/build_class_index.py

import json
import argparse
from pathlib import Path
from typing import Optional

# 修改 ROOT 路径，回退3层到项目根目录
ROOT = Path(__file__).resolve().parents[3]

SYSTEM_CONFIG = {
    "acmeair": {
        "callgraph_file": "acmeair_callgraph.json",
        "dependency_file": "acmeair_dependency.json",
        "semantic_file": "acmeair_semantic.json",
        "class_index_output": "acmeair_classes.json",
        "filter_strategy": "acmeair",
        "base_package": "com.acmeair",
    },
    "daytrader": {
        "callgraph_file": "daytrader7_callgraph.json",
        "dependency_file": "daytrader7_dependency.json",
        "semantic_file": "daytrader_semantic.json",
        "class_index_output": "daytrader_classes.json",
        "filter_strategy": "daytrader",
        "base_package": "com.ibm.websphere.samples.daytrader",
    },
    "jpetstore": {
        "callgraph_file": "jPetStore_callgraph.json",
        "dependency_file": "jPetStore_dependency.json",
        "semantic_file": "jpetstore_semantic.json",
        "class_index_output": "jpetstore_classes.json",
        "filter_strategy": "jpetstore",
        "base_package": "org.springframework.samples.jpetstore",
    },
    "plants": {
        "callgraph_file": "plantsbywebsphere_callgraph.json",
        "dependency_file": "plantsbywebsphere_dependency.json",
        "semantic_file": "plants_semantic.json",
        "class_index_output": "plants_classes.json",
        "filter_strategy": "plants",
        "base_package": "com.ibm.websphere.samples.pbw",
    },
}


def normalize_class_name(raw: str) -> str:
    """把各种形式的类名字符串统一成我们约定的 Class ID.

    目前先做最简单的清洗：去空格。后续如果你想统一成全限定名，可以在这里集中改规则。
    """
    if raw is None:
        return ""
    return raw.strip()


def _extract_class_from_method_sig(sig: str, system: str) -> Optional[str]:
    """从方法签名中抽取所属类名。

    例如：
    - "com.acmeair.loader.FlightLoader.getArrivalTime(java.util.Date, int)"
      -> "com.acmeair.loader.FlightLoader"
    - "String getAirportCode()" -> 无法得到类名，返回 None。
    """
    if not sig:
        return None
    # 兼容冒号和括号
    name = sig.split('(')[0].split(':')[0].strip()
    if " " in name:
        name = name.split(" ")[-1]
    base_pkg = SYSTEM_CONFIG[system].get("base_package", "")
    if base_pkg and name.startswith(base_pkg):
        parts = name.split(".")
        # 只要层级比 base_pkg 深，我们就尝试去掉最后一段（方法名）
        if len(parts) > len(base_pkg.split(".")):
            # 只有最后一段首字母是小写时才去掉，否则可能误删类名
            if parts[-1] and parts[-1][0].islower():
                return ".".join(parts[:-1])
        return name
    return None


def load_callgraph_classes(system: str) -> set:
    """从 acmeair_callgraph.json 中抽取类 ID 集合。"""
    cfg = SYSTEM_CONFIG[system]
    path = ROOT / "data" / "processed" / "callgraph" / cfg["callgraph_file"]
    classes: set = set()

    if not path.is_file():
        return classes

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    # 1) nodes 里有时也会包含带类名的方法
    for node in data.get("nodes", []):
        cls = _extract_class_from_method_sig(node, system)
        if cls:
            classes.add(cls)

    # 2) edges 里的 source / target
    for edge in data.get("edges", []):
        src = edge.get("source")
        tgt = edge.get("target")
        if src:
            cls = _extract_class_from_method_sig(src, system)
            if cls:
                classes.add(cls)
        if tgt:
            cls = _extract_class_from_method_sig(tgt, system)
            if cls:
                classes.add(cls)

    return classes


def load_dependency_classes(system: str) -> set:
    """从 acmeair_dependency.json 中抽取类 ID 集合（粗粒度，仅基于 nodes）。"""
    cfg = SYSTEM_CONFIG[system]
    path = ROOT / "data" / "processed" / "dependency" / cfg["dependency_file"]
    classes: set = set()

    if not path.is_file():
        return classes

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    for n in data.get("nodes", []):
        # nodes 里既有类名也有方法/文件名，这里先做一个很宽松的过滤：
        # 1) 跳过明显是方法签名的（包含空格或括号）
        # 2) 其余全部当作“类或包名候选”
        if "(" in n or ")" in n or " " in n:
            continue
        # 这里也用统一的业务包过滤
        cls = _extract_class_from_method_sig(n, system) if "." in n else None
        if not cls:
            cls = normalize_class_name(n)
        if cls:
            classes.add(cls)

    return classes


def load_semantic_classes(system: str) -> set:
    """从 semantic/acmeair_semantic.json 中抽取类 ID 集合。"""
    cfg = SYSTEM_CONFIG[system]
    path = ROOT / "data" / "processed" / "semantic" / cfg["semantic_file"]
    classes: set = set()

    if not path.is_file():
        print(f"ERROR: File not found: {path}")
        return classes

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    # 兼容性处理
    if isinstance(data, list):
        for item in data:
            if isinstance(item, dict):
                cls = item.get("class")
            else:
                cls = str(item)
            if cls:
                classes.add(normalize_class_name(cls))
    elif isinstance(data, dict):
        for cls_name in data.keys():
            classes.add(normalize_class_name(cls_name))
    return classes


def _filter_candidate_classes(system: str, all_classes: set, semantic_classes: set) -> set:
    filtered = set()
    base_pkg = SYSTEM_CONFIG[system].get("base_package", "")
    for cls in all_classes:
        if not cls:
            continue
        # 通用判定：必须属于业务根包
        if base_pkg and cls.startswith(base_pkg):
            # 通用排除：排除明显的解析残留或工具类
            if any(garbage in cls.lower() for garbage in [".em", ".q", ".tx", "util"]):
                continue
            filtered.add(cls)
    # 如果过滤后全没了，可能是解析太严，回退到包含语义的类
    if not filtered and semantic_classes:
        filtered = semantic_classes
    print(f"[{system}] Filtered {len(all_classes)} down to {len(filtered)} high-quality classes.")
    return filtered


def build_class_index(system: str):
    ast_classes: set = set()
    callgraph_classes = load_callgraph_classes(system)
    dependency_classes = load_dependency_classes(system)
    semantic_classes = load_semantic_classes(system)
    embedding_classes: set = set()
    trace_classes: set = set()
    jmeter_classes: set = set()

    # 调试信息
    print(f"DEBUG [{system}]: Found {len(callgraph_classes)} classes from CallGraph")
    print(f"DEBUG [{system}]: Found {len(dependency_classes)} classes from Dependency")
    print(f"DEBUG [{system}]: Found {len(semantic_classes)} classes from Semantic")

    all_classes = (
        ast_classes
        | callgraph_classes
        | dependency_classes
        | embedding_classes
        | semantic_classes
        | trace_classes
        | jmeter_classes
    )

    filtered_classes = _filter_candidate_classes(system, all_classes, semantic_classes)

    if not filtered_classes:
        filtered_classes = callgraph_classes or dependency_classes

    index = {}
    for cls in sorted(filtered_classes):
        index[cls] = {
            "has_callgraph": cls in callgraph_classes,
            "has_dependency": cls in dependency_classes,
            "has_semantic": cls in semantic_classes,
        }

    cfg = SYSTEM_CONFIG[system]
    out_path = ROOT / "data" / "processed" / "fusion" / cfg["class_index_output"]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(index, f, indent=2, ensure_ascii=False)

    print(f"[{system}] saved class index to {out_path} (N={len(index)})")


def main():
    parser = argparse.ArgumentParser(description="Build class index for a given system (acmeair/daytrader/jpetstore/plants)")
    parser.add_argument("--system", choices=sorted(SYSTEM_CONFIG.keys()), required=True)
    args = parser.parse_args()
    build_class_index(args.system)


if __name__ == "__main__":
    main()