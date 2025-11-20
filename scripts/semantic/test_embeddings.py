import torch
from pathlib import Path

# 指定要测试的文件
EMBEDDING_FILE = Path("data/processed/embedding/acmeair.pt")

def main():
    if not EMBEDDING_FILE.exists():
        print(f"❌ {EMBEDDING_FILE} not found")
        return

    data = torch.load(EMBEDDING_FILE)

    print(f"✅ Loaded {len(data)} items from {EMBEDDING_FILE}")

    # 打印前 3 个方法的情况
    for i, item in enumerate(data[:3]):
        print(f"\n--- Item {i+1} ---")
        print(f"Class: {item['class']}")
        print(f"Method: {item['method']}")
        print(f"Embedding shape: {item['embedding'].shape}")
        print(f"First 5 values: {item['embedding'][:5]}")

if __name__ == "__main__":
    main()
