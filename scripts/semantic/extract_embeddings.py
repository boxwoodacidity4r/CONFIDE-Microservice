import json
from pathlib import Path
import argparse
import torch
import numpy as np
from embedding_manager import EmbeddingManager
import re
import math
import os

def robust_split_camel_case(text):
    # Split on camel case, underscores, and digits, and handle consecutive capitals
    text = re.sub(r'([a-z0-9])([A-Z])', r'\1 \2', text)
    text = re.sub(r'([A-Z]+)([A-Z][a-z])', r'\1 \2', text)
    text = re.sub(r'[_\-\d]+', ' ', text)
    return text


def run(
    input_file: Path,
    output_file: Path,
    em: EmbeddingManager,
    use_arch_stopwords: bool = True,
    extra_arch_stopwords: str = "",
    *,
    # Domain-Focus Filter (Phase1 semantic purification)
    use_domain_focus_filter: bool = True,
    extra_domain_stopwords: str = "",
    # NEW: distillation recipe switch
    distill_mode: str = "default",
):
    if not input_file.exists():
        print(f"[WARN] {input_file} not found, skip")
        return

    distill_mode = (distill_mode or "default").strip().lower()
    if distill_mode not in {"default", "legacy"}:
        raise ValueError(f"Unknown distill_mode: {distill_mode}. Expected default|legacy")

    # Legacy mode: recover semantic diversity (paper-friendly ablation)
    if distill_mode == "legacy":
        # keep architecture stopwords behavior as-is, but do NOT apply domain-focus hard list
        use_domain_focus_filter = False

    with open(input_file, "r", encoding="utf-8") as f:
        items = json.load(f)

    # --- Step 1: Compute global document frequency (DF) and IDF ---
    java_stopwords = {
        'public', 'private', 'protected', 'static', 'final', 'void', 'return',
        'if', 'else', 'for', 'while', 'new', 'try', 'catch', 'throw', 'throws',
        'class', 'interface', 'extends', 'implements', 'import', 'package',
        'string', 'int', 'boolean', 'list', 'map', 'set', 'arraylist', 'hashmap',
        'override', 'get', 'set', 'null', 'true', 'false', 'this', 'object'
    }

    # Add system/project-name token(s) to stopwords (e.g., acmeair/daytrader/jpetstore/plants)
    # This reduces low-discriminative tokens that appear in many classes/methods.
    sys_token = input_file.stem.lower()
    # common naming like 'acmeair_semantic' or 'acmeair'
    for cand in [sys_token, sys_token.replace('_semantic', ''), sys_token.replace('semantic', '')]:
        cand = cand.strip('_')
        if cand:
            java_stopwords.add(cand)

    # --- Architecture/domain hard stopwords ---
    arch_stopwords = set()
    if use_arch_stopwords:
        try:
            # prefer the merged helper if available
            from architecture_stopwords import get_all_arch_stopwords, ARCH_STOPWORDS
        except Exception:
            get_all_arch_stopwords = None
            ARCH_STOPWORDS = set()

        if get_all_arch_stopwords is not None:
            arch_stopwords |= set(get_all_arch_stopwords(include_domain_focus_hard=bool(use_domain_focus_filter)))
        else:
            arch_stopwords |= set(ARCH_STOPWORDS)

    if extra_arch_stopwords:
        for w in re.split(r"[\s,;]+", extra_arch_stopwords.strip()):
            if w:
                arch_stopwords.add(w.lower())

    # Extra Domain-Focus stopwords (user-provided)
    if extra_domain_stopwords:
        for w in re.split(r"[\s,;]+", extra_domain_stopwords.strip()):
            if w:
                arch_stopwords.add(w.lower())

    # combined stopwords
    stopwords = set(java_stopwords) | set(arch_stopwords)

    # --- Persist semantic purification parameters for auditability ---
    params_path = output_file.parent / (output_file.stem + "_semantic_params.json")
    try:
        params = {
            "input": str(input_file),
            "output": str(output_file),
            "distill_mode": str(distill_mode),
            "use_arch_stopwords": bool(use_arch_stopwords),
            "use_domain_focus_filter": bool(use_domain_focus_filter),
            "extra_arch_stopwords": extra_arch_stopwords,
            "extra_domain_stopwords": extra_domain_stopwords,
        }
        with open(params_path, "w", encoding="utf-8") as f:
            json.dump(params, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"[WARN] Failed to write params file: {params_path} ({e})")

    doc_freq = {}
    total_docs = 0
    for item in items:
        # Use improved camel case splitting for all text fields
        cls_name = robust_split_camel_case(item.get('class', '')).lower()
        m_name = robust_split_camel_case(item.get('method', '')).lower()
        raw_text = f"{item.get('comments', '')} {item.get('variables', '')}"
        raw_text = robust_split_camel_case(raw_text).lower()
        words = set(re.findall(r'[a-z]+', f"{cls_name} {m_name} {raw_text}"))
        for w in words:
            if w not in stopwords and len(w) > 2:
                doc_freq[w] = doc_freq.get(w, 0) + 1
        total_docs += 1

    # Compute IDF for each word
    idf = {w: math.log((1 + total_docs) / (1 + df)) + 1 for w, df in doc_freq.items()}

    # IDF-min drop threshold
    # - default mode: keep current stricter default (1.35)
    # - legacy mode: disable IDF-based dropping (threshold=0)
    idf_min_raw = os.environ.get('MM_SEM_IDF_MIN', '').strip()
    if distill_mode == "legacy":
        idf_threshold = 0.0
    else:
        idf_threshold = float(idf_min_raw) if idf_min_raw else 1.35

    results = []
    seen_texts = set()  
    def distill_business_logic(item):
        """
        

        Modes:
          - default: (cls*3 + method*2) + IDF filtering
          - legacy:  (cls*1 + method*1) + no IDF dropping
        """
        cls_name = robust_split_camel_case(item.get('class', '')).lower()
        m_name = robust_split_camel_case(item.get('method', '')).lower()
        raw_text = f"{item.get('comments', '')} {item.get('variables', '')}"
        raw_text = robust_split_camel_case(raw_text).lower()
        words = re.findall(r'[a-z]+', raw_text)

        # Filter: stopwords, length, and global IDF
        filtered_words = [
            w for w in words
            if w not in stopwords and len(w) > 2 and idf.get(w, 100) >= float(idf_threshold)
        ]

        if distill_mode == "legacy":
            cls_w, m_w = 1, 1
        else:
            cls_w, m_w = 3, 2

        distilled_text = (cls_name + " ") * int(cls_w) + (m_name + " ") * int(m_w) + " ".join(filtered_words)
        return distilled_text.strip()

    # 1) Compatibility with method_name/method fields
    for item in items:
        # Compatibility with both method_name and method keys
        method = item.get("method_name") or item.get("method") or ""
        # === Business semantic distillation pre-processing ===
        text = distill_business_logic(item)
        if not text or text in seen_texts:
            continue
        seen_texts.add(text)
        vec = em.encode(text)  # tensor
        results.append({
            "class": item.get("class"),
            "method": method,
            "embedding": vec
        })

    # Before saving, check for duplicate embeddings (extra safety)
    unique_results = []
    seen_vecs = set()
    for r in results:
        key = tuple(np.array(r["embedding"]).flatten().tolist())
        if key not in seen_vecs:
            seen_vecs.add(key)
            unique_results.append(r)

    torch.save(unique_results, output_file)
    print(f"[OK] Saved embeddings ({len(unique_results)} unique) -> {output_file}")

    # 2) Build class-level embeddings (aggregate method embeddings; use mean)
    class2vecs = {}
    for r in unique_results:
        cls = r["class"]
        if not cls:
            continue
        arr = np.array(r["embedding"]).reshape(-1)
        class2vecs.setdefault(cls, []).append(arr)
    class_embeddings = []
    for idx, (cls, vecs) in enumerate(class2vecs.items()):
        arr = np.stack(vecs, axis=0)
        # Option C: Mean + Max combination
        combined_vec = (arr.mean(axis=0) + arr.max(axis=0)) / 2
        class_embeddings.append({
            "class": cls,
            "embedding": combined_vec.tolist(),
            "embedding_idx": idx
        })
    # Output class-level embedding index
    class_emb_path = output_file.parent / (output_file.stem + "_class_embeddings.json")
    with open(class_emb_path, "w", encoding="utf-8") as f:
        json.dump(class_embeddings, f, ensure_ascii=False, indent=2)
    print(f"[OK] Saved class-level embeddings ({len(class_embeddings)}) -> {class_emb_path}")

# -------------------- CLI entrypoint --------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate embeddings from semantic JSON")
    parser.add_argument("--input", required=True, help="Path to semantic JSON file")
    parser.add_argument("--output", required=True, help="Path to save embeddings (.pt file)")
    parser.add_argument("--model", default="microsoft/graphcodebert-base", help="Embedding model name")

    # NEW: distillation mode
    parser.add_argument(
        "--distill_mode",
        choices=["default", "legacy"],
        default="default",
        help=(
            "Semantic text distillation recipe. "
            "default = domain-focus + IDF dropping + (cls*3, method*2); "
            "legacy = disable domain-focus hard list, IDF threshold=0, (cls*1, method*1)."
        ),
    )

    parser.add_argument("--use-arch-stopwords", action="store_true", default=True,
                        help="Enable aggressive software-architecture stopwords filtering (default: enabled).")
    parser.add_argument("--no-arch-stopwords", action="store_true", default=False,
                        help="Disable architecture stopwords filtering.")
    parser.add_argument("--extra-arch-stopwords", default="",
                        help="Extra architecture stopwords (comma/space separated).")

    # Domain-Focus Filter knobs
    parser.add_argument("--no-domain-focus", action="store_true", default=False,
                        help="Disable Domain-Focus Filter hard stopwords (default: enabled).")
    parser.add_argument("--extra-domain-stopwords", default="",
                        help="Extra domain-focus stopwords (comma/space separated).")

    args = parser.parse_args()

    em = EmbeddingManager()
    run(
        Path(args.input),
        Path(args.output),
        em,
        use_arch_stopwords=(not args.no_arch_stopwords),
        extra_arch_stopwords=args.extra_arch_stopwords,
        use_domain_focus_filter=(not args.no_domain_focus),
        extra_domain_stopwords=args.extra_domain_stopwords,
        distill_mode=str(args.distill_mode),
    )
