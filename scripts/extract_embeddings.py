import json
from pathlib import Path
import argparse
from embedding_manager import EmbeddingManager

# -------------------- 主函数 --------------------
def run(input_file: Path, output_file: Path, em: EmbeddingManager):
    if not input_file.exists():
        print(f"⚠️ {input_file} not found, skip")
        return

    with open(input_file, "r", encoding="utf-8") as f:
        items = json.load(f)

    results = []
    for item in items:
        # 拼接文本作为输入
        text = " ".join([
            item.get("class", ""),
            item.get("method_name", ""),  # 注意 semantic JSON 用 method_name
            " ".join(item.get("variables", [])),
            item.get("comment", ""),
            item.get("body", "")
        ]).strip()

        if not text:
            continue

        vec = em.encode(text)
        results.append({
            "class": item.get("class"),
            "method": item.get("method_name"),
            "embedding": vec
        })

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"✅ Saved embeddings -> {output_file}")

# -------------------- 命令行入口 --------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate embeddings from semantic JSON")
    parser.add_argument("--input", required=True, help="Path to semantic JSON file")
    parser.add_argument("--output", required=True, help="Path to save embeddings JSON")
    parser.add_argument("--model", default="text-embedding-3-large", help="Embedding model name")
    args = parser.parse_args()

    em = EmbeddingManager(model=args.model)
    run(Path(args.input), Path(args.output), em)
