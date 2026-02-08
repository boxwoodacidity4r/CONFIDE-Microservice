package extractor.dependency;

import com.github.javaparser.JavaParser;
import com.github.javaparser.ParserConfiguration;
import com.github.javaparser.ast.CompilationUnit;
import com.github.javaparser.ast.ImportDeclaration;
import com.github.javaparser.ast.body.ClassOrInterfaceDeclaration;
import com.github.javaparser.symbolsolver.JavaSymbolSolver;
import com.github.javaparser.symbolsolver.resolution.typesolvers.CombinedTypeSolver;
import com.github.javaparser.symbolsolver.resolution.typesolvers.ReflectionTypeSolver;
import com.github.javaparser.symbolsolver.resolution.typesolvers.JavaParserTypeSolver;
import com.github.javaparser.symbolsolver.resolution.typesolvers.JarTypeSolver;

import org.jgrapht.graph.DefaultDirectedGraph;
import org.jgrapht.graph.DefaultEdge;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.util.List;
import java.util.Optional;

/**
 * 依赖图提取器，集成 SymbolSolver 支持 JAR 依赖
 */
public class JavaParserDependencyGraphExtractor {

    private final JavaParser parser;

    public JavaParserDependencyGraphExtractor(String sourceRootPath, List<File> dependencyJars) {
        CombinedTypeSolver typeSolver = new CombinedTypeSolver();
        typeSolver.add(new ReflectionTypeSolver());
        typeSolver.add(new JavaParserTypeSolver(new File(sourceRootPath)));
        if (dependencyJars != null) {
            for (File jar : dependencyJars) {
                try {
                    typeSolver.add(new JarTypeSolver(jar.getAbsolutePath()));
                    System.out.println("[Dependency] Added JarTypeSolver: " + jar.getAbsolutePath());
                } catch (Exception e) {
                    System.err.println("[Dependency] Failed to add JarTypeSolver for " + jar + ": " + e.getMessage());
                }
            }
        }
        ParserConfiguration config = new ParserConfiguration();
        config.setSymbolResolver(new JavaSymbolSolver(typeSolver));
        this.parser = new JavaParser(config);
    }

    // 复用 CallGraph 风格的业务类过滤：仅保留项目内部包前缀
    private boolean isBusinessClass(String cls) {
        if (cls == null) return false;
        String s = cls.trim();
        String[] prefixWhitelist = new String[] {
                "org.springframework.samples.jpetstore.",
                "com.ibm.websphere.samples.",
                "com.ibm.websphere.samples.pbw.",
                "com.ibm.websphere.",
                "com.acmeair.",
                "com.ibm.ws.samples.daytrader.",
                "org.apache.geronimo.samples.daytrader.",
                "daytrader.",
        };
        for (String p : prefixWhitelist) {
            if (s.startsWith(p)) return true;
        }
        return false;
    }

    private String dependencyKind(String fqn) {
        if (fqn == null) return "other";
        String lower = fqn.toLowerCase();
        if (lower.endsWith("service") || lower.endsWith("controller") || lower.endsWith("repository")) return "service";
        if (lower.endsWith("entity") || lower.endsWith("dto") || lower.endsWith("model") || lower.endsWith("pojo")) return "entity";
        return "other";
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

                        // 获取包名
                        String packageName = cu.getPackageDeclaration().map(pd -> pd.getNameAsString()).orElse("");
                        // 找到当前文件中的类名（只取第一个 public class/interface）
                        String className = cu.findFirst(ClassOrInterfaceDeclaration.class)
                                .map(ClassOrInterfaceDeclaration::getNameAsString)
                                .orElse(p.getFileName().toString());
                        String fullClassName = packageName.isEmpty() ? className : packageName + "." + className;
                        graph.addVertex(fullClassName);

                        // 遍历 import，建立依赖边（过滤基础设施依赖）
                        for (ImportDeclaration imp : cu.getImports()) {
                            String dep = imp.getNameAsString();
                            // 跳过 star import，避免 java.util.* 这种噪声
                            if (imp.isAsterisk()) continue;
                            if (!isBusinessClass(dep)) continue;

                            graph.addVertex(dep);
                            graph.addEdge(fullClassName, dep);

                            // 轻量标注：仅日志输出，数据格式不破坏
                            String kind = dependencyKind(dep);
                            if (!"other".equals(kind)) {
                                System.out.println("[Dependency] " + fullClassName + " -> " + dep + " (" + kind + ")");
                            }
                        }
                    } catch (Exception e) {
                        System.err.println("Failed to parse file: " + p + " -> " + e.getMessage());
                    }
                });

        return graph;
    }
}
