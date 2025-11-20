import subprocess
from pathlib import Path

# 四个应用（指向源码根目录或模块集合）
APPS = {
    "acmeair": "data/raw/acmeair",
    "daytrader": "data/raw/daytrader7",
    "plants": "data/raw/plantsbywebsphere",
    "jpetstore": "data/raw/jpetstore",
}

OUTPUT_DIR = Path("data/processed/semantic")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# fat-jar 路径
FAT_JAR = Path("tools/target/tools-fat.jar").resolve()

successful = []
failed = []

def run_semantic_extraction(app: str, src_root: str):
    src_path = Path(src_root)
    # 优先使用 src/main/java，如果不存在就用模块根目录
    java_src = src_path / "src/main/java"
    if java_src.exists() and java_src.is_dir():
        final_src = java_src
    else:
        final_src = src_path

    output = OUTPUT_DIR / f"{app}.json"
    cmd = [
        "java", "-jar", str(FAT_JAR),
        "semantic", str(final_src.resolve()), str(output.resolve())
    ]
    print(f"\n🚀 Running semantic extraction for {app} ...")
    try:
        subprocess.run(cmd, check=True)
        print(f"✅ {app} finished -> {output}")
        successful.append(app)
    except subprocess.CalledProcessError as e:
        print(f"❌ {app} failed with exit code {e.returncode}")
        failed.append(app)

if __name__ == "__main__":
    for app, src in APPS.items():
        run_semantic_extraction(app, src)

    print("\n=== Summary ===")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
