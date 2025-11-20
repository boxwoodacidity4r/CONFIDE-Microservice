import os
import subprocess
import json
from pathlib import Path

# -------------------- 配置路径 --------------------
BASE_DIR = Path(__file__).resolve().parent.parent
TOOLS_DIR = BASE_DIR / "tools"
RAW_DIR = BASE_DIR / "data/raw"
AST_DIR = BASE_DIR / "data/processed/ast"
CG_DIR = BASE_DIR / "data/processed/callgraph"
DEP_DIR = BASE_DIR / "data/processed/dependency"
SEM_DIR = BASE_DIR / "data/processed/semantic"   # 新增语义目录

AST_DIR.mkdir(parents=True, exist_ok=True)
CG_DIR.mkdir(parents=True, exist_ok=True)
DEP_DIR.mkdir(parents=True, exist_ok=True)
SEM_DIR.mkdir(parents=True, exist_ok=True)

# Maven 绝对路径
MAVEN_CMD = r"D:\apache-maven-3.9.10\bin\mvn.cmd"

# 四个应用
APPS = ["acmeair", "daytrader7", "jPetStore", "plantsbywebsphere"]

# -------------------- JSON 去重函数 --------------------
def deduplicate_json(file_path: Path):
    if not file_path.exists():
        return

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, list):
            return  # 只处理列表形式的 JSON（如 semantic）

        seen = set()
        unique_items = []
        for item in data:
            key = (
                item.get("class", ""),
                item.get("method_name", ""),
                " ".join(item.get("variables", []))
            )
            if key not in seen:
                seen.add(key)
                unique_items.append(item)

        if len(unique_items) < len(data):
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(unique_items, f, ensure_ascii=False, indent=2)
            print(f"✨ Deduplicated {file_path.name}: {len(data)} -> {len(unique_items)}")

    except Exception as e:
        print(f"⚠️ Deduplication skipped for {file_path.name}, error: {e}")

# -------------------- 提取器运行函数 --------------------
def run_extractor(app_name, extractor_type, output_path):
    src_path = RAW_DIR / app_name / "src" / "main" / "java"
    if not src_path.exists():
        print(f"⚠️  Warning: {src_path} not found, fallback to {RAW_DIR / app_name}")
        src_path = RAW_DIR / app_name

    cmd = [
        MAVEN_CMD, "exec:java",
        "-Dexec.mainClass=extractor.Main",
        f"-Dexec.args={extractor_type} {src_path} {output_path}"
    ]

    print(f"\n>>> Running {extractor_type} extractor on {app_name} ...")
    print("Command:", " ".join(str(c) for c in cmd))

    result = subprocess.run(cmd, cwd=TOOLS_DIR, shell=True)
    if result.returncode == 0:
        print(f"✅ {extractor_type} finished for {app_name}")
        # 如果是 semantic，做一次去重
        if extractor_type == "semantic":
            deduplicate_json(output_path)
    else:
        print(f"❌ {extractor_type} failed for {app_name}")

# -------------------- 主函数 --------------------
def main():
    for app in APPS:
        ast_output = AST_DIR / f"{app}_ast.json"
        cg_output = CG_DIR / f"{app}_callgraph.json"
        dep_output = DEP_DIR / f"{app}_dependency.json"
        sem_output = SEM_DIR / f"{app}_semantic.json"   # 新增语义输出路径

        run_extractor(app, "ast", ast_output)
        run_extractor(app, "callgraph", cg_output)
        run_extractor(app, "dependency", dep_output)
        run_extractor(app, "semantic", sem_output)      # 新增语义 case

if __name__ == "__main__":
    main()
