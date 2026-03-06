package extractor.utils;

import com.github.javaparser.JavaParser;
import com.github.javaparser.ParseResult;
import com.github.javaparser.ast.CompilationUnit;
import com.github.javaparser.ast.Node;
//import com.github.javaparser.ast.NodeList;
import org.jgrapht.graph.DefaultDirectedGraph;
import org.jgrapht.graph.DefaultEdge;
import org.json.JSONArray;
import org.json.JSONObject;

import java.io.FileInputStream;
import java.io.IOException;
import java.nio.file.Path;
import java.util.List;

public class JsonExporter {

    // Recursively convert an AST node to JSON.
    public static JSONObject nodeToJson(Node node) {
        JSONObject json = new JSONObject();
        json.put("nodeType", node.getClass().getSimpleName());
        json.put("toString", node.toString());

        JSONArray children = new JSONArray();
        List<Node> childNodes = node.getChildNodes();
        for (Node child : childNodes) {
            children.put(nodeToJson(child));
        }
        json.put("children", children);

        return json;
    }

    // Convert a call graph to JSON.
    public static JSONObject callGraphToJson(DefaultDirectedGraph<String, DefaultEdge> graph) {
        JSONObject json = new JSONObject();
        JSONArray nodes = new JSONArray();
        JSONArray edges = new JSONArray();

        graph.vertexSet().forEach(nodes::put);
        graph.edgeSet().forEach(e -> {
            JSONObject edge = new JSONObject();
            edge.put("source", graph.getEdgeSource(e));
            edge.put("target", graph.getEdgeTarget(e));
            edges.put(edge);
        });

        json.put("nodes", nodes);
        json.put("edges", edges);
        return json;
    }

    // Parse a Java file from disk and return the CompilationUnit.
    public static CompilationUnit parseJavaFile(Path path) throws IOException {
        try (FileInputStream in = new FileInputStream(path.toFile())) {
            JavaParser parser = new JavaParser();
            ParseResult<CompilationUnit> result = parser.parse(in);
            if (result.isSuccessful() && result.getResult().isPresent()) {
                return result.getResult().get();
            } else {
                throw new IOException("Failed to parse Java file: " + path);
            }
        }
    }
}
