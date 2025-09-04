package extractor.dependency;

import com.github.javaparser.JavaParser;
import com.github.javaparser.ParserConfiguration;
import com.github.javaparser.ast.CompilationUnit;
import com.github.javaparser.ast.ImportDeclaration;
import com.github.javaparser.ast.body.ClassOrInterfaceDeclaration;

import org.jgrapht.graph.DefaultDirectedGraph;
import org.jgrapht.graph.DefaultEdge;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.util.Optional;

/**
 * 简单的依赖图提取器：
 * 节点 = 类名
 * 边   = import 依赖关系 (A -> B)
 */
public class JavaParserDependencyGraphExtractor {

    private final JavaParser parser;

    public JavaParserDependencyGraphExtractor() {
        ParserConfiguration config = new ParserConfiguration();
        this.parser = new JavaParser(config);
    }

    public DefaultDirectedGraph<String, DefaultEdge> extract(File srcDir) throws IOException {
        DefaultDirectedGraph<String, DefaultEdge> graph =
                new DefaultDirectedGraph<>(DefaultEdge.class);

        Files.walk(srcDir.toPath())
                .filter(Files::isRegularFile)
                .filter(p -> p.toString().endsWith(".java"))
                .forEach(p -> {
                    try {
                        Optional<CompilationUnit> result = parser.parse(p).getResult();
                        if (result.isEmpty()) return;
                        CompilationUnit cu = result.get();

                        // 找到当前文件中的类名（只取第一个 public class/interface）
                        String className = cu.findFirst(ClassOrInterfaceDeclaration.class)
                                .map(ClassOrInterfaceDeclaration::getNameAsString)
                                .orElse(p.getFileName().toString());

                        graph.addVertex(className);

                        // 遍历 import，建立依赖边
                        for (ImportDeclaration imp : cu.getImports()) {
                            String dep = imp.getNameAsString(); // 例如 java.util.List
                            graph.addVertex(dep);
                            graph.addEdge(className, dep);
                        }
                    } catch (Exception e) {
                        System.err.println("Failed to parse file: " + p + " -> " + e.getMessage());
                    }
                });

        return graph;
    }
}
