import json
from pathlib import Path
import argparse
import torch
import numpy as np
from scripts.semantic.embedding_manager import EmbeddingManager

# -------------------- 主函数 --------------------
def run(input_file: Path, output_file: Path, em: EmbeddingManager):
    if not input_file.exists():
        print(f"⚠️ {input_file} not found, skip")
        return

    with open(input_file, "r", encoding="utf-8") as f:
        items = json.load(f)

    results = []
    seen_texts = set()  # 用于去重（基于拼接后的文本）

    for item in items:
        # 拼接文本作为输入
        text = " ".join([
            item.get("class", ""),
            item.get("method_name", ""),  # semantic JSON 用 method_name
            " ".join(item.get("variables", [])),
            item.get("comment", ""),
            item.get("body", "")
        ]).strip()

        if not text or text in seen_texts:
            continue
        seen_texts.add(text)

        vec = em.encode(text)  # tensor
        results.append({
            "class": item.get("class"),
            "method": item.get("method_name"),
            "embedding": vec  # 保留 tensor，不转 list
        })

    # 保存前再检查 embedding 是否有重复（双重保险）
    unique_results = []
    seen_vecs = set()
    for r in results:
        # flatten 保证是一维向量（避免 list of list 报错）
        key = tuple(np.array(r["embedding"]).flatten().tolist())
        if key not in seen_vecs:
            seen_vecs.add(key)
            unique_results.append(r)

    torch.save(unique_results, output_file)
    print(f"✅ Saved embeddings ({len(unique_results)} unique) -> {output_file}")

# -------------------- 命令行入口 --------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate embeddings from semantic JSON")
    parser.add_argument("--input", required=True, help="Path to semantic JSON file")
    parser.add_argument("--output", required=True, help="Path to save embeddings (.pt file)")
    parser.add_argument("--model", default="microsoft/graphcodebert-base", help="Embedding model name")
    args = parser.parse_args()

    em = EmbeddingManager()
    run(Path(args.input), Path(args.output), em)
