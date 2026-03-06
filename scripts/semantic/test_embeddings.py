import torch
from pathlib import Path

# Target file to test
EMBEDDING_FILE = Path("data/processed/embedding/acmeair.pt")

def main():
    if not EMBEDDING_FILE.exists():
        print(f"[FAIL] {EMBEDDING_FILE} not found")
        return

    data = torch.load(EMBEDDING_FILE)

    print(f"[OK] Loaded {len(data)} items from {EMBEDDING_FILE}")

    # Print the first 3 items
    for i, item in enumerate(data[:3]):
        print(f"\n--- Item {i+1} ---")
        print(f"Class: {item['class']}")
        print(f"Method: {item['method']}")
        print(f"Embedding shape: {item['embedding'].shape}")
        print(f"First 5 values: {item['embedding'][:5]}")

if __name__ == "__main__":
    main()
