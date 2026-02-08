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
import com.github.javaparser.symbolsolver.resolution.typesolvers.JarTypeSolver;
import org.jgrapht.graph.DefaultDirectedWeightedGraph;
import org.jgrapht.graph.DefaultWeightedEdge;

import java.io.File;
import java.io.FileInputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.stream.Collectors;
import java.util.stream.Stream;
import java.nio.file.Paths;

public class JavaParserCallGraphExtractor implements CallGraphExtractor {

    private final JavaParser parser;

    public JavaParserCallGraphExtractor(String sourceRootPath, List<File> dependencyJars) {
        CombinedTypeSolver typeSolver = new CombinedTypeSolver();
        // JDK
        typeSolver.add(new ReflectionTypeSolver());
        // 项目源码
        typeSolver.add(new JavaParserTypeSolver(new File(sourceRootPath)));
        // 自动收集常见依赖目录下的所有 jar
        List<File> allJars = new ArrayList<>();
        if (dependencyJars != null) allJars.addAll(dependencyJars);
        // 自动扫描 lib、libs、target/lib、build/libs 等目录
        String[] jarDirs = {"lib", "libs", "target/lib", "build/libs"};
        for (String dir : jarDirs) {
            Path jarPath = Paths.get(sourceRootPath, dir);
            if (Files.exists(jarPath)) {
                try {
                    List<File> found = Files.walk(jarPath)
                        .filter(p -> p.toString().endsWith(".jar"))
                        .map(Path::toFile)
                        .collect(Collectors.toList());
                    allJars.addAll(found);
                } catch (Exception e) {
                    System.err.println("[CallGraph] Failed to scan jars in " + jarPath + ": " + e.getMessage());
                }
            }
        }
        // 外部依赖 JAR
        for (File jar : allJars) {
            try {
                typeSolver.add(new JarTypeSolver(jar.getAbsolutePath()));
                System.out.println("[SymbolSolver] Added JarTypeSolver: " + jar.getAbsolutePath());
            } catch (Exception e) {
                System.err.println("[CallGraph] Failed to add JarTypeSolver for " + jar + ": " + e.getMessage());
            }
        }
        JavaSymbolSolver solver = new JavaSymbolSolver(typeSolver);
        this.parser = new JavaParser();
        this.parser.getParserConfiguration().setSymbolResolver(solver);
    }

    // 科研代码，用于微服务提取实验，重点保证正确性和稳定性
    @SuppressWarnings("unused")
    private String formatMethod(ResolvedMethodDeclaration m) {
        String cls = m.declaringType().getQualifiedName();
        StringBuilder sb = new StringBuilder(cls)
                .append(".")
                .append(m.getName())
                .append("(");
        for (int i = 0; i < m.getNumberOfParams(); i++) {
            if (i > 0) sb.append(",");
            sb.append(m.getParam(i).getType().describe());
        }
        sb.append(")");
        return sb.toString();
    }

    // 判断是否业务类（仅用于 caller）
    private boolean isBusinessClass(String cls) {
        if (cls == null) return false;
        String s = cls.trim();
        String lower = s.toLowerCase();
        // 应用代码白名单前缀：对这些前缀一律认为是业务类
        String[] prefixWhitelist = new String[] {
            "org.springframework.samples.jpetstore.",
            "com.ibm.websphere.samples.",      // plantsbywebsphere 之类
            "com.ibm.websphere.samples.pbw.",  // 更具体
            "com.ibm.websphere.",
            "com.acmeair.",
            "com.ibm.ws.samples.daytrader.",
            "org.apache.geronimo.samples.daytrader.",
            "daytrader.",
        };
        for (String prefix : prefixWhitelist) {
            if (s.startsWith(prefix)) {
                return true;
            }
        }
        // 框架/标准库黑名单前缀（保持原有过滤）
        String[] prefixBlacklist = new String[] {
            "java.", "javax.", "jakarta.",
            "org.springframework.", "org.apache.",
            "com.fasterxml.", "ch.qos.logback."
        };
        for (String prefix : prefixBlacklist) {
            if (lower.startsWith(prefix)) return false;
        }
        // 关键词黑名单
        String[] keywordBlacklist = new String[] {
            "test", "mock", "jmeter", "benchmark", "example"
        };
        for (String kw : keywordBlacklist) {
            if (lower.contains(kw)) return false;
        }
        return true;
    }

    /**
     * 归一化方法/类名：
     * - 去掉泛型 <...>
     * - 去掉数组/内部类符号等噪声
     */
    private static String normalizeFqn(String s) {
        if (s == null) return "";
        String x = s.trim();
        // strip generics: List<com.x.A> -> List
        x = x.replaceAll("<[^>]*>", "");
        // strip array suffix
        x = x.replace("[]", "");
        // collapse whitespace
        x = x.replaceAll("\\s+", "");
        return x;
    }

    private static String extractDeclaringTypeFromMethodFqn(String methodFqn) {
        if (methodFqn == null) return "";
        int idx = methodFqn.lastIndexOf('.');
        if (idx <= 0) return "";
        return methodFqn.substring(0, idx);
    }

    /**
     * 超级节点阈值：inDegree > totalBusinessClasses * ratio
     */
    private static final double SUPER_NODE_INDEGREE_RATIO = 0.20;

    private boolean isProjectInternalCallee(String fqn) {
        if (fqn == null) return false;
        String s = fqn.trim();
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

    /**
     * 对无法 resolve 的 MethodCallExpr，做一个“保守的语法级”callee 归一化：
     * - 如果能拿到 scope 的 ResolvedType，则用 scopeType.qualifiedName + "." + methodName
     * - 否则返回空字符串（不入图），避免 methodName-only 节点爆炸
     */
    private String fallbackCallee(MethodCallExpr call) {
        try {
            if (call.getScope().isPresent()) {
                try {
                    String typeDesc = call.getScope().get().calculateResolvedType().describe();
                    typeDesc = normalizeFqn(typeDesc);
                    if (typeDesc.contains(".")) {
                        String fqn = typeDesc + "." + call.getNameAsString();
                        // 只保留项目内部 callee，避免基础设施噪声补边
                        return isProjectInternalCallee(fqn) ? fqn : "";
                    }
                } catch (Exception ignored) {
                }
            }
        } catch (Exception ignored) {
        }
        return "";
    }

    @Override
    public org.jgrapht.graph.DefaultDirectedGraph<String, org.jgrapht.graph.DefaultEdge> extractCallGraph(File sourceRoot) throws Exception {
        // 内部使用加权图做统计/惩罚，最终转换为无权图以兼容现有接口
        DefaultDirectedWeightedGraph<String, DefaultWeightedEdge> wGraph =
                new DefaultDirectedWeightedGraph<>(DefaultWeightedEdge.class);

        // 用于后处理：统计 caller->callee 的调用频次
        Map<String, Map<String, Integer>> edgeCounts = new HashMap<>();
        Set<String> businessClasses = new HashSet<>();

        try (Stream<Path> paths = Files.walk(sourceRoot.toPath())) {
            paths.filter(Files::isRegularFile)
                    .filter(p -> p.toString().endsWith(".java"))
                    .filter(p -> !p.toString().toLowerCase().contains("/test/"))
                    .forEach(p -> {
                        try (FileInputStream in = new FileInputStream(p.toFile())) {
                            CompilationUnit cu = parser.parse(in).getResult().orElse(null);
                            if (cu == null) return;
                            cu.findAll(MethodDeclaration.class).forEach(method -> {
                                final String[] callerHolder = new String[1];
                                final String[] classNameHolder = new String[1];
                                try {
                                    ResolvedMethodDeclaration resolvedMethod = method.resolve();
                                    callerHolder[0] = normalizeFqn(resolvedMethod.getQualifiedName());
                                    classNameHolder[0] = normalizeFqn(resolvedMethod.declaringType().getQualifiedName());
                                } catch (Exception e) {
                                    callerHolder[0] = normalizeFqn(method.getDeclarationAsString(false, false, false));
                                    classNameHolder[0] = normalizeFqn(cu.getPrimaryTypeName().orElse(""));
                                    System.err.println("[CallGraph] Method resolve failed: " + method.getName() + " - " + e.getMessage());
                                }

                                String caller = callerHolder[0];
                                String className = classNameHolder[0];
                                if (!isBusinessClass(className)) return;

                                businessClasses.add(className);
                                if (!wGraph.containsVertex(caller)) wGraph.addVertex(caller);

                                method.findAll(MethodCallExpr.class).forEach(call -> {
                                    // 1) 优先走 SymbolSolver resolve
                                    try {
                                        ResolvedMethodDeclaration calleeResolved = call.resolve();
                                        String callee = normalizeFqn(calleeResolved.getQualifiedName());
                                        if (callee.isEmpty()) return;
                                        if (!wGraph.containsVertex(callee)) wGraph.addVertex(callee);

                                        edgeCounts.computeIfAbsent(caller, k -> new HashMap<>())
                                                .merge(callee, 1, Integer::sum);
                                        return;
                                    } catch (Exception e) {
                                        // 2) 语法级 fallback：尽量恢复边，但不引入大量噪声
                                        String callee = fallbackCallee(call);
                                        if (callee == null || callee.isEmpty()) return;
                                        callee = normalizeFqn(callee);

                                        if (!wGraph.containsVertex(callee)) wGraph.addVertex(callee);
                                        edgeCounts.computeIfAbsent(caller, k -> new HashMap<>())
                                                .merge(callee, 1, Integer::sum);

                                        // 降噪：fallback 已限定为 project-internal，这里打印更有意义
                                        System.err.println("[CallGraph] Callee resolve failed, fallback=" + callee + " | expr=" + call + " - " + e.getMessage());
                                    }
                                });
                            });
                        } catch (Exception e) {
                            System.err.println("[CallGraph] File parse failed: " + p + " - " + e.getMessage());
                        }
                    });
        }

        // 先把计数边写入加权图（调用频次作为基础权重）
        for (Map.Entry<String, Map<String, Integer>> e1 : edgeCounts.entrySet()) {
            String caller = e1.getKey();
            for (Map.Entry<String, Integer> e2 : e1.getValue().entrySet()) {
                String callee = e2.getKey();
                int cnt = e2.getValue();
                if (!wGraph.containsVertex(caller)) wGraph.addVertex(caller);
                if (!wGraph.containsVertex(callee)) wGraph.addVertex(callee);
                DefaultWeightedEdge edge = wGraph.getEdge(caller, callee);
                if (edge == null) {
                    edge = wGraph.addEdge(caller, callee);
                    if (edge != null) wGraph.setEdgeWeight(edge, cnt);
                } else {
                    wGraph.setEdgeWeight(edge, wGraph.getEdgeWeight(edge) + cnt);
                }
            }
        }

        // 双向调用增强：A->B 且 B->A，则两条边权重 *2
        for (DefaultWeightedEdge e : new ArrayList<>(wGraph.edgeSet())) {
            String s = wGraph.getEdgeSource(e);
            String t = wGraph.getEdgeTarget(e);
            if (wGraph.containsEdge(t, s)) {
                wGraph.setEdgeWeight(e, wGraph.getEdgeWeight(e) * 2.0);
            }
        }

        // 超级节点惩罚：对入度过高（被过多业务类调用）的节点，削弱进入它的边权重
        int totalBusiness = businessClasses.size();
        int indegreeThreshold = (int) Math.ceil(totalBusiness * SUPER_NODE_INDEGREE_RATIO);
        if (indegreeThreshold < 1) indegreeThreshold = 1;

        // 统计入度（按“业务类调用者”去重）
        Map<String, Set<String>> indegreeCallers = new HashMap<>();
        for (DefaultWeightedEdge e : wGraph.edgeSet()) {
            String callerMethod = wGraph.getEdgeSource(e);
            String calleeMethod = wGraph.getEdgeTarget(e);
            String callerCls = extractDeclaringTypeFromMethodFqn(callerMethod);
            String calleeCls = extractDeclaringTypeFromMethodFqn(calleeMethod);
            if (callerCls.isEmpty() || calleeCls.isEmpty()) continue;
            if (!isBusinessClass(callerCls)) continue;
            indegreeCallers.computeIfAbsent(calleeCls, k -> new HashSet<>()).add(callerCls);
        }

        Set<String> superClasses = new HashSet<>();
        for (Map.Entry<String, Set<String>> e : indegreeCallers.entrySet()) {
            if (e.getValue().size() >= indegreeThreshold) {
                superClasses.add(e.getKey());
            }
        }
        if (!superClasses.isEmpty()) {
            System.out.println("[CallGraph] Super-node classes (inDegree callers >= " + indegreeThreshold + "): " + superClasses.size());
        }

        // 对指向超级类的方法节点的边做惩罚（不直接剪枝，避免图断裂过多）
        final double SUPER_EDGE_PENALTY = 0.2; // 削弱到 20%
        for (DefaultWeightedEdge e : wGraph.edgeSet()) {
            String calleeMethod = wGraph.getEdgeTarget(e);
            String calleeCls = extractDeclaringTypeFromMethodFqn(calleeMethod);
            if (superClasses.contains(calleeCls)) {
                wGraph.setEdgeWeight(e, wGraph.getEdgeWeight(e) * SUPER_EDGE_PENALTY);
            }
        }

        // 在函数末尾：将 wGraph 转换为 DefaultDirectedGraph
        org.jgrapht.graph.DefaultDirectedGraph<String, org.jgrapht.graph.DefaultEdge> graph =
                new org.jgrapht.graph.DefaultDirectedGraph<>(org.jgrapht.graph.DefaultEdge.class);
        for (String v : wGraph.vertexSet()) {
            graph.addVertex(v);
        }
        for (DefaultWeightedEdge e : wGraph.edgeSet()) {
            String s = wGraph.getEdgeSource(e);
            String t = wGraph.getEdgeTarget(e);
            if (!graph.containsEdge(s, t)) {
                graph.addEdge(s, t);
            }
        }
        return graph;
    }
}
