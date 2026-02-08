import subprocess
from pathlib import Path
import sys
import os

# 四个应用对应的语义 JSON
SEMANTIC_JSONS = {
    "acmeair": "data/processed/semantic/acmeair_semantic.json",
    "daytrader": "data/processed/semantic/daytrader_semantic.json",
    "plants": "data/processed/semantic/plants_semantic.json",
    "jpetstore": "data/processed/semantic/jpetstore_semantic.json",
}

OUTPUT_DIR = Path("data/processed/embedding")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

successful = []
failed = []

def run_embedding_extraction(
    app: str,
    input_json: str,
    *,
    no_arch_stopwords: bool = False,
    extra_arch_stopwords: str = "",
    no_domain_focus: bool = False,
    extra_domain_stopwords: str = "",
    distill_mode: str = "default",
):
    input_path = Path(input_json).resolve()
    output_path = OUTPUT_DIR / f"{app}.pt"  # 改为 .pt

    # Use current interpreter (venv) to avoid missing deps
    cmd = [
        sys.executable,
        "scripts/semantic/extract_embeddings.py",
        "--input",
        str(input_path),
        "--output",
        str(output_path),
        "--distill_mode",
        str(distill_mode),
    ]

    if no_arch_stopwords:
        cmd.append("--no-arch-stopwords")
    if extra_arch_stopwords:
        cmd.extend(["--extra-arch-stopwords", extra_arch_stopwords])

    if no_domain_focus:
        cmd.append("--no-domain-focus")
    if extra_domain_stopwords:
        cmd.extend(["--extra-domain-stopwords", extra_domain_stopwords])

    print(f"\n🚀 Running embedding extraction for {app} ...")
    try:
        subprocess.run(cmd, check=True)
        print(f"✅ {app} embeddings saved -> {output_path}")
        successful.append(app)
    except subprocess.CalledProcessError as e:
        print(f"❌ {app} failed with exit code {e.returncode}")
        failed.append(app)

if __name__ == "__main__":
    # Environment-driven batch flags (PowerShell friendly):
    #   $env:MM_DISTILL_MODE='legacy'
    #   $env:MM_NO_ARCH_STOPWORDS=1
    #   $env:MM_EXTRA_ARCH_STOPWORDS='foo,bar'
    #   $env:MM_NO_DOMAIN_FOCUS=1
    #   $env:MM_EXTRA_DOMAIN_STOPWORDS='impl,service'
    distill_mode = os.environ.get("MM_DISTILL_MODE", "default").strip().lower() or "default"
    no_arch = os.environ.get("MM_NO_ARCH_STOPWORDS", "").strip().lower() in {"1", "true", "yes"}
    extra_arch = os.environ.get("MM_EXTRA_ARCH_STOPWORDS", "").strip()
    no_domain = os.environ.get("MM_NO_DOMAIN_FOCUS", "").strip().lower() in {"1", "true", "yes"}
    extra_domain = os.environ.get("MM_EXTRA_DOMAIN_STOPWORDS", "").strip()

    for app, json_file in SEMANTIC_JSONS.items():
        run_embedding_extraction(
            app,
            json_file,
            no_arch_stopwords=no_arch,
            extra_arch_stopwords=extra_arch,
            no_domain_focus=no_domain,
            extra_domain_stopwords=extra_domain,
            distill_mode=distill_mode,
        )

    print("\n=== Summary ===")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
