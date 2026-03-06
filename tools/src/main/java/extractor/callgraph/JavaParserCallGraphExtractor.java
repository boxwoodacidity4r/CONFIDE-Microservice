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
        // Project source code
        typeSolver.add(new JavaParserTypeSolver(new File(sourceRootPath)));
        // Collect all jars from common dependency directories.
        List<File> allJars = new ArrayList<>();
        if (dependencyJars != null) allJars.addAll(dependencyJars);
        // Auto-scan lib/libs/target/lib/build/libs under the source root.
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
        // External dependency JARs
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

    // Research-oriented extractor used for microservice extraction experiments.
    // Prioritizes correctness and stability.
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

    // Decide whether a class should be treated as a business class (caller-side filter only).
    private boolean isBusinessClass(String cls) {
        if (cls == null) return false;
        String s = cls.trim();
        String lower = s.toLowerCase();
        // Application package allow-list prefixes.
        String[] prefixWhitelist = new String[] {
            "org.springframework.samples.jpetstore.",
            "com.ibm.websphere.samples.",      // plantsbywebsphere, etc.
            "com.ibm.websphere.samples.pbw.",  // more specific
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
        // Framework / standard library deny-list prefixes.
        String[] prefixBlacklist = new String[] {
            "java.", "javax.", "jakarta.",
            "org.springframework.", "org.apache.",
            "com.fasterxml.", "ch.qos.logback."
        };
        for (String prefix : prefixBlacklist) {
            if (lower.startsWith(prefix)) return false;
        }
        // Keyword deny-list.
        String[] keywordBlacklist = new String[] {
            "test", "mock", "jmeter", "benchmark", "example"
        };
        for (String kw : keywordBlacklist) {
            if (lower.contains(kw)) return false;
        }
        return true;
    }

    /**
     * Normalize method/class FQN:
     * - remove generics <...>
     * - remove array suffixes and other common noise
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
     * Super-node threshold: inDegree > totalBusinessClasses * ratio
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
     * For MethodCallExpr that cannot be resolved, apply a conservative syntax-level callee normalization:
     * - if scope resolved type is available: scopeType.qualifiedName + "." + methodName
     * - otherwise return empty (skip) to avoid methodName-only node explosion
     */
    private String fallbackCallee(MethodCallExpr call) {
        try {
            if (call.getScope().isPresent()) {
                try {
                    String typeDesc = call.getScope().get().calculateResolvedType().describe();
                    typeDesc = normalizeFqn(typeDesc);
                    if (typeDesc.contains(".")) {
                        String fqn = typeDesc + "." + call.getNameAsString();
                        // Keep only project-internal callees to avoid adding infrastructure noise edges.
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
        // Internally use a weighted graph for counting/penalization, then convert back to an
        // unweighted graph to match the existing interface.

        DefaultDirectedWeightedGraph<String, DefaultWeightedEdge> wGraph =
                new DefaultDirectedWeightedGraph<>(DefaultWeightedEdge.class);

        // Used for post-processing: count caller->callee call frequencies
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
                                    // 1) Prefer SymbolSolver resolution
                                    try {
                                        ResolvedMethodDeclaration calleeResolved = call.resolve();
                                        String callee = normalizeFqn(calleeResolved.getQualifiedName());
                                        if (callee.isEmpty()) return;
                                        if (!wGraph.containsVertex(callee)) wGraph.addVertex(callee);

                                        edgeCounts.computeIfAbsent(caller, k -> new HashMap<>())
                                                .merge(callee, 1, Integer::sum);
                                        return;
                                    } catch (Exception e) {
                                        // 2) Syntax-level fallback: try to recover the edge without introducing too much noise
                                        String callee = fallbackCallee(call);
                                        if (callee == null || callee.isEmpty()) return;
                                        callee = normalizeFqn(callee);

                                        if (!wGraph.containsVertex(callee)) wGraph.addVertex(callee);
                                        edgeCounts.computeIfAbsent(caller, k -> new HashMap<>())
                                                .merge(callee, 1, Integer::sum);

                                        // Noise reduction: fallback is limited to project-internal, so this log is more meaningful
                                        System.err.println("[CallGraph] Callee resolve failed, fallback=" + callee + " | expr=" + call + " - " + e.getMessage());
                                    }
                                });
                            });
                        } catch (Exception e) {
                            System.err.println("[CallGraph] File parse failed: " + p + " - " + e.getMessage());
                        }
                    });
        }

        // First, write the counted edges into the weighted graph (call frequency as base weight)
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

        // Bidirectional call enhancement: if A->B and B->A, then double the weight of both edges
        for (DefaultWeightedEdge e : new ArrayList<>(wGraph.edgeSet())) {
            String s = wGraph.getEdgeSource(e);
            String t = wGraph.getEdgeTarget(e);
            if (wGraph.containsEdge(t, s)) {
                wGraph.setEdgeWeight(e, wGraph.getEdgeWeight(e) * 2.0);
            }
        }

        // Super-node penalization: weaken the edge weights leading to nodes with excessively high in-degree (called by too many business classes)
        int totalBusiness = businessClasses.size();
        int indegreeThreshold = (int) Math.ceil(totalBusiness * SUPER_NODE_INDEGREE_RATIO);
        if (indegreeThreshold < 1) indegreeThreshold = 1;

        // Count in-degrees (deduplicated by "business class caller")
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

        // Penalize edges pointing to super-class method nodes (do not directly prune to avoid excessive graph fragmentation)
        final double SUPER_EDGE_PENALTY = 0.2; // Weaken to 20%
        for (DefaultWeightedEdge e : wGraph.edgeSet()) {
            String calleeMethod = wGraph.getEdgeTarget(e);
            String calleeCls = extractDeclaringTypeFromMethodFqn(calleeMethod);
            if (superClasses.contains(calleeCls)) {
                wGraph.setEdgeWeight(e, wGraph.getEdgeWeight(e) * SUPER_EDGE_PENALTY);
            }
        }

        // At the end of the function: convert wGraph to DefaultDirectedGraph
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
