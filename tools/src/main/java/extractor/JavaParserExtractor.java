package extractor;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.github.javaparser.StaticJavaParser;
import com.github.javaparser.ast.CompilationUnit;
import com.github.javaparser.ast.Node;

import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.nio.file.*;
import java.util.*;

/**
 * 使用 JavaParser 提取 AST，并保存为 JSON 格式
 */
public class JavaParserExtractor {

    static class ASTNode {
        public int id;
        public String type;
        public String name;

        public ASTNode(int id, String type, String name) {
            this.id = id;
            this.type = type;
            this.name = name;
        }
    }

    static class ASTEdge {
        public int source;
        public int target;
        public String relation;

        public ASTEdge(int source, int target, String relation) {
            this.source = source;
            this.target = target;
            this.relation = relation;
        }
    }

    static class ASTGraph {
        public List<ASTNode> nodes = new ArrayList<>();
        public List<ASTEdge> edges = new ArrayList<>();
    }

    private static int nodeCounter = 0;
    private static Map<Node, Integer> nodeIds = new HashMap<>();
    private static ASTGraph graph = new ASTGraph();

    public static void main(String[] args) throws IOException {
        if (args.length < 2) {
            System.out.println("用法: java -jar javaparser-extractor.jar <源代码目录> <输出JSON文件>");
            return;
        }

        String srcDir = args[0];
        String outputJson = args[1];

        // 遍历所有 .java 文件
        Files.walk(Paths.get(srcDir))
                .filter(p -> p.toString().endsWith(".java"))
                .forEach(path -> {
                    try {
                        parseFile(path.toFile());
                    } catch (IOException e) {
                        e.printStackTrace();
                    }
                });

        // 输出 JSON
        ObjectMapper mapper = new ObjectMapper();
        mapper.writerWithDefaultPrettyPrinter().writeValue(new File(outputJson), graph);

        System.out.println("AST 提取完成，已保存到: " + outputJson);
    }

    private static void parseFile(File file) throws IOException {
        try (FileInputStream in = new FileInputStream(file)) {
            CompilationUnit cu = StaticJavaParser.parse(in);
            traverseAST(cu, null);
        }
    }

    private static void traverseAST(Node node, Node parent) {
        int id = getNodeId(node);
        String type = node.getClass().getSimpleName();
        String name = node.toString().split("\\s+")[0]; // 简单取前几个词作为节点名（可改进）

        graph.nodes.add(new ASTNode(id, type, name));

        if (parent != null) {
            int parentId = getNodeId(parent);
            graph.edges.add(new ASTEdge(parentId, id, "contains"));
        }

        for (Node child : node.getChildNodes()) {
            traverseAST(child, node);
        }
    }


    private static int getNodeId(Node node) {
        return nodeIds.computeIfAbsent(node, k -> ++nodeCounter);
    }
}
