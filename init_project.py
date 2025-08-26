import os

# 项目目录结构
dirs = [
    "data/raw",        # 原始数据
    "data/processed",  # 处理后的数据
    "scripts",         # 数据提取脚本
    "models",          # 模型代码
    "results",         # 实验结果
    "logs",            # 日志文件
    "config"           # 配置文件
]

# 创建目录
for d in dirs:
    os.makedirs(d, exist_ok=True)
print(" 项目目录创建完成")

# 写 .gitignore
gitignore_content = """
# Python 缓存
__pycache__/
*.pyc
*.pyo
*.pyd

# 虚拟环境
venv/
.env/
.venv/
*.env

# 数据文件
*.npy
*.pt
*.csv
*.json
*.graphml

# 日志文件
logs/
*.log

# IDE 文件
.idea/
.vscode/
"""

with open(".gitignore", "w", encoding="utf-8") as f:
    f.write(gitignore_content.strip())
print(" .gitignore 文件创建完成")

# 写 requirements.txt
requirements = """
networkx
javalang
javaparser
torch
transformers
numpy
pandas
matplotlib
seaborn
scikit-learn
"""

with open("requirements.txt", "w", encoding="utf-8") as f:
    f.write(requirements.strip())
print(" requirements.txt 文件创建完成")

# 写 README.md
readme_content = """# Multimodal Microservice Extraction"""

## 📂 项目结构
