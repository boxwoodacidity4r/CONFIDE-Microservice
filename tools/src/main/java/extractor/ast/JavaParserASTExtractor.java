package extractor.ast;

import com.github.javaparser.JavaParser;
import com.github.javaparser.ast.CompilationUnit;
import com.github.javaparser.ast.Node;
import com.github.javaparser.symbolsolver.JavaSymbolSolver;
import com.github.javaparser.symbolsolver.resolution.typesolvers.CombinedTypeSolver;
import com.github.javaparser.symbolsolver.resolution.typesolvers.ReflectionTypeSolver;
import com.github.javaparser.symbolsolver.resolution.typesolvers.JavaParserTypeSolver;

import java.io.File;
import java.io.FileInputStream;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class JavaParserASTExtractor implements ASTExtractor {

    private final JavaParser parser;

    public JavaParserASTExtractor(String sourceRootPath) {
        CombinedTypeSolver typeSolver = new CombinedTypeSolver();
        typeSolver.add(new ReflectionTypeSolver());
        typeSolver.add(new JavaParserTypeSolver(new File(sourceRootPath)));
        this.parser = new JavaParser();
        this.parser.getParserConfiguration().setSymbolResolver(new JavaSymbolSolver(typeSolver));
    }

    @Override
    public CompilationUnit extractAST(File file) throws Exception {
        try (FileInputStream in = new FileInputStream(file)) {
            return parser.parse(in).getResult()
                    .orElseThrow(() -> new RuntimeException("解析失败: " + file.getName()));
        }
    }

    // =========================
    // 将 Node 转成 Map，用于 JSON 输出
    // =========================
    public static Map<String, Object> nodeToMap(Node node) {
        Map<String, Object> map = new HashMap<>();
        map.put("type", node.getClass().getSimpleName());
        map.put("range", node.getRange().map(r -> r.toString()).orElse(""));
        List<Object> children = new ArrayList<>();
        for (Node child : node.getChildNodes()) {
            children.add(nodeToMap(child));
        }
        map.put("children", children);
        return map;
    }
}
