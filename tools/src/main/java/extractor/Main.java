package extractor;

import extractor.ast.ASTExtractor;
import extractor.ast.JavaParserASTExtractor;
import extractor.callgraph.CallGraphExtractor;
import extractor.callgraph.JavaParserCallGraphExtractor;
import extractor.dependency.JavaParserDependencyGraphExtractor;
import extractor.semantic.JavaParserSemanticExtractor;
import extractor.semantic.SemanticExtractor;

import org.jgrapht.graph.DefaultDirectedGraph;
import org.jgrapht.graph.DefaultEdge;
import com.github.javaparser.ast.CompilationUnit;
import com.fasterxml.jackson.databind.ObjectMapper;

import java.io.File;
import java.nio.file.Files;
import java.util.ArrayList;
import java.util.List;
import java.util.HashMap;
import java.util.Map;

public class Main {
    public static void main(String[] args) throws Exception {
        if (args.length < 3) {
            System.err.println("Usage: java extractor.Main <extractorType> <srcPath> <outputPath>");
            System.exit(1);
        }

        String extractorType = args[0].toLowerCase();
        String srcPath = args[1];
        String outputPath = args[2];

        // 确保输出目录存在
        File outFile = new File(outputPath);
        outFile.getParentFile().mkdirs();

        ObjectMapper mapper = new ObjectMapper();

        switch (extractorType) {
            case "ast": {
                ASTExtractor extractor = new JavaParserASTExtractor(srcPath);
                List<Object> astList = new ArrayList<>();
                List<String> failedFiles = new ArrayList<>();

                Files.walk(new File(srcPath).toPath())
                        .filter(Files::isRegularFile)
                        .filter(p -> p.toString().endsWith(".java"))
                        .forEach(p -> {
                            try {
                                CompilationUnit cu = extractor.extractAST(p.toFile());
                                if (cu != null) {
                                    astList.add(JavaParserASTExtractor.nodeToMap(cu));
                                }
                            } catch (Exception e) {
                                failedFiles.add(p.toString() + " (" + e.getMessage() + ")");
                            }
                        });

                mapper.writerWithDefaultPrettyPrinter().writeValue(outFile, astList);

                System.out.println("✅ AST extraction finished. Total files: " + astList.size());
                if (!failedFiles.isEmpty()) {
                    System.err.println("⚠️ Failed to parse files: " + failedFiles);
                }
                break;
            }

            case "callgraph": {
                CallGraphExtractor extractor = new JavaParserCallGraphExtractor(srcPath);
                DefaultDirectedGraph<String, DefaultEdge> graph = extractor.extractCallGraph(new File(srcPath));

                Map<String, Object> json = convertGraphToJson(graph);
                mapper.writerWithDefaultPrettyPrinter().writeValue(outFile, json);

                System.out.println("✅ CallGraph extraction finished. Nodes: "
                        + graph.vertexSet().size() + " Edges: " + graph.edgeSet().size());
                break;
            }

            case "dependency": {
                JavaParserDependencyGraphExtractor extractor = new JavaParserDependencyGraphExtractor();
                DefaultDirectedGraph<String, DefaultEdge> depGraph = extractor.extract(new File(srcPath));

                Map<String, Object> json = convertGraphToJson(depGraph);
                mapper.writerWithDefaultPrettyPrinter().writeValue(outFile, json);

                System.out.println("✅ DependencyGraph extraction finished. Nodes: "
                        + depGraph.vertexSet().size() + " Edges: " + depGraph.edgeSet().size());
                break;
            }

            case "semantic": {
                SemanticExtractor extractor = new JavaParserSemanticExtractor();
                extractor.extract(srcPath, outputPath);
                System.out.println("✅ Semantic extraction finished. Output: " + outputPath);
                break;
            }


            default:
                System.err.println("Unknown extractor type: " + extractorType);
                System.exit(2);
        }
    }

    // 公共方法：把 JGraphT 图转成 JSON
    private static Map<String, Object> convertGraphToJson(DefaultDirectedGraph<String, DefaultEdge> graph) {
        Map<String, Object> json = new HashMap<>();
        json.put("nodes", graph.vertexSet());

        List<Map<String, String>> edges = new ArrayList<>();
        graph.edgeSet().forEach(e -> {
            String source = graph.getEdgeSource(e);
            String target = graph.getEdgeTarget(e);
            Map<String, String> edge = new HashMap<>();
            edge.put("source", source);
            edge.put("target", target);
            edges.add(edge);
        });

        json.put("edges", edges);
        return json;
    }
}
