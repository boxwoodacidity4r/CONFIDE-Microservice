import subprocess
from pathlib import Path

# 四个应用对应的语义 JSON
SEMANTIC_JSONS = {
    "acmeair": "data/processed/semantic/acmeair.json",
    "daytrader": "data/processed/semantic/daytrader.json",
    "plants": "data/processed/semantic/plants.json",
    "jpetstore": "data/processed/semantic/jpetstore.json",
}

OUTPUT_DIR = Path("data/processed/embedding")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

successful = []
failed = []

def run_embedding_extraction(app: str, input_json: str):
    input_path = Path(input_json).resolve()
    output_path = OUTPUT_DIR / f"{app}.json"  # 或者 .npy，根据你extract_embeddings.py设定
    cmd = [
        "python", "scripts/extract_embeddings.py",
        "--input", str(input_path),
        "--output", str(output_path)
    ]
    print(f"\n🚀 Running embedding extraction for {app} ...")
    try:
        subprocess.run(cmd, check=True)
        print(f"✅ {app} embeddings saved -> {output_path}")
        successful.append(app)
    except subprocess.CalledProcessError as e:
        print(f"❌ {app} failed with exit code {e.returncode}")
        failed.append(app)

if __name__ == "__main__":
    for app, json_file in SEMANTIC_JSONS.items():
        run_embedding_extraction(app, json_file)

    print("\n=== Summary ===")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
