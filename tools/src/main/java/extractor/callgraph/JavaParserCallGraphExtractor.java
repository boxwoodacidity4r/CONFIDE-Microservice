package extractor.callgraph;

import com.github.javaparser.JavaParser;
import com.github.javaparser.ast.CompilationUnit;
import com.github.javaparser.ast.body.MethodDeclaration;
import com.github.javaparser.ast.expr.MethodCallExpr;
import com.github.javaparser.symbolsolver.JavaSymbolSolver;
import com.github.javaparser.resolution.declarations.ResolvedMethodDeclaration;
import com.github.javaparser.symbolsolver.resolution.typesolvers.CombinedTypeSolver;
import com.github.javaparser.symbolsolver.resolution.typesolvers.JavaParserTypeSolver;
import com.github.javaparser.symbolsolver.resolution.typesolvers.ReflectionTypeSolver;
import org.jgrapht.graph.DefaultDirectedGraph;
import org.jgrapht.graph.DefaultEdge;

import java.io.File;
import java.io.FileInputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.stream.Stream;

public class JavaParserCallGraphExtractor implements CallGraphExtractor {

    private final JavaParser parser;

    public JavaParserCallGraphExtractor(String sourceRootPath) {
        CombinedTypeSolver typeSolver = new CombinedTypeSolver();
        typeSolver.add(new ReflectionTypeSolver());
        typeSolver.add(new JavaParserTypeSolver(new File(sourceRootPath)));

        this.parser = new JavaParser();
        this.parser.getParserConfiguration().setSymbolResolver(new JavaSymbolSolver(typeSolver));
    }

    @Override
    public DefaultDirectedGraph<String, DefaultEdge> extractCallGraph(File sourceRoot) throws Exception {
        DefaultDirectedGraph<String, DefaultEdge> graph = new DefaultDirectedGraph<>(DefaultEdge.class);

        try (Stream<Path> paths = Files.walk(sourceRoot.toPath())) {
            paths.filter(Files::isRegularFile)
                 .filter(p -> p.toString().endsWith(".java"))
                 .forEach(p -> {
                     try (FileInputStream in = new FileInputStream(p.toFile())) {
                         CompilationUnit cu = parser.parse(in).getResult().orElse(null);
                         if (cu == null) return;

                         cu.findAll(MethodDeclaration.class).forEach(method -> {
                             String caller = method.getDeclarationAsString(false, false, false);
                             graph.addVertex(caller);

                             method.findAll(MethodCallExpr.class).forEach(call -> {
                                 try {
                                     ResolvedMethodDeclaration resolved = call.resolve();
                                     String callee = resolved.getQualifiedSignature();
                                     graph.addVertex(callee);
                                     graph.addEdge(caller, callee);
                                 } catch (Exception ignored) {
                                     // 解析失败的调用忽略
                                 }
                             });
                         });
                     } catch (Exception e) {
                         e.printStackTrace();
                     }
                 });
        }
        return graph;
    }
}
