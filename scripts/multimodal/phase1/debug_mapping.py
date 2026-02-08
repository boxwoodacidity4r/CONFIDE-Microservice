import json
import numpy as np
from pathlib import Path

# 设置系统
SYSTEM = "acmeair"
ROOT = Path(r"D:\multimodal_microservice_extraction")

# 1. 加载 class_order
order_path = ROOT / "data" / "processed" / "fusion" / f"{SYSTEM}_class_order.json"
with open(order_path, 'r') as f:
    class_order = json.load(f)
class_set = set(class_order)

# 2. 加载生成的 S_temp
temp_path = ROOT / "data" / "processed" / "temporal" / f"{SYSTEM}_S_temp.npy"
S_temp = np.load(temp_path)

# 3. 分析结果
active_indices = np.where(np.diag(S_temp) > 0)[0]
print(f"--- System: {SYSTEM} ---")
print(f"Total classes in order: {len(class_order)}")
print(f"Classes with temporal data: {len(active_indices)}")

print("\n[Matched Classes]")
for idx in active_indices:
    print(f" - {class_order[idx]}")

if len(active_indices) < 5:
    print("\n[WARNING] Very few matches! Check if your ENDPOINT_MAPS in build_S_temp.py matches these names exactly.")
