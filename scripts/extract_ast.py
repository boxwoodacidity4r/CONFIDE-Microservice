import os
import json
import subprocess
import networkx as nx

def run_javaparser(jar_path, java_src_dir, output_json):
    """
    调用 JavaParser AST 提取工具
    :param jar_path: javaparser 提取器 jar 文件路径
    :param java_src_dir: Java 源码目录
    :param output_json: AST 输出 JSON 文件
    """
    cmd = [
        "java", "-jar", jar_path,
        java_src_dir,
        output_json
    ]
    print(f"🚀 正在运行 JavaParser 提取 AST...")
    subprocess.run(cmd, check=True)
    print(f"✅ AST 提取完成，结果保存到 {output_json}")


def ast_json_to_graph(ast_json_path, output_graphml_path):
    """
    将 AST JSON 转换为 NetworkX Graph 并保存为 GraphML
    :param ast_json_path: AST JSON 文件路径
    :param output_graphml_path: 输出 GraphML 文件路径
    """
    with open(ast_json_path, "r", encoding="utf-8") as f:
        ast_data = json.load(f)

    G = nx.DiGraph()

    # TODO: 根据 JSON 结构添加节点和边
    # 假设 ast_data["nodes"] 是节点列表，["edges"] 是边列表
    for node in ast_data.get("nodes", []):
        node_id = node["id"]
        G.add_node(node_id, **node)

    for edge in ast_data.get("edges", []):
        src = edge["source"]
        tgt = edge["target"]
        G.add_edge(src, tgt, **edge)

    nx.write_graphml(G, output_graphml_path)
    print(f"✅ GraphML 文件已保存到 {output_graphml_path}")


if __name__ == "__main__":
    # === 路径配置 ===
    JAVAPARSER_JAR = "tools/javaparser-extractor.jar"  # 你需要准备的 JavaParser 提取器 jar
    ACMEAIR_SRC = "data/raw/acmeair"  # Acmeair 源码目录
    AST_JSON_OUT = "data/processed/acmeair_ast.json"
    AST_GRAPHML_OUT = "data/processed/acmeair_ast.graphml"

    # === 步骤 1: 调用 JavaParser 提取 AST 到 JSON ===
    run_javaparser(JAVAPARSER_JAR, ACMEAIR_SRC, AST_JSON_OUT)

    # === 步骤 2: 将 JSON 转换为 GraphML 图结构 ===
    ast_json_to_graph(AST_JSON_OUT, AST_GRAPHML_OUT)
