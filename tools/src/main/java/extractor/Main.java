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
import java.nio.file.Path;
import java.util.LinkedHashSet;
import java.util.Set;
import java.util.stream.Stream;

public class Main {
    private static List<File> collectDependencyJars(String srcPath) {
        // Strategy:
        // 1) Locate a stable "project root" (closest pom.xml upward; fallback to srcPath)
        // 2) Recursively scan for jars under a whitelist of directory patterns that commonly hold deps
        //    (target/dependency, WEB-INF/lib, liberty/wlp, db2jars, etc.)
        // This avoids the "0 jar" issue on multi-module projects like daytrader7.
        Set<String> uniq = new LinkedHashSet<>();
        try {
            File src = new File(srcPath).getCanonicalFile();

            File projectRoot = findNearestPomRoot(src, 8);
            if (projectRoot == null) projectRoot = src;

            File moduleRoot = guessModuleRoot(src, projectRoot);

            // DEBUG: show roots and intended scan locations
            System.out.println("[Main] srcPath=" + src.getPath());
            System.out.println("[Main] projectRoot=" + projectRoot.getPath());
            System.out.println("[Main] moduleRoot=" + moduleRoot.getPath());

            scanJarWhitelist(moduleRoot.toPath(), uniq);
            scanJarWhitelist(new File(moduleRoot, "build").toPath(), uniq);

            if (!moduleRoot.getCanonicalPath().equals(projectRoot.getCanonicalPath())) {
                scanJarWhitelist(projectRoot.toPath(), uniq);
                scanJarWhitelist(new File(projectRoot, "build").toPath(), uniq);
            }

            if (uniq.isEmpty()) {
                System.out.println("[Main] No jars found after scan. moduleRoot/build exists=" + new File(moduleRoot, "build").exists());
            }

        } catch (Exception e) {
            System.err.println("[Main] collectDependencyJars failed: " + e.getMessage());
        }

        List<File> jars = new ArrayList<>();
        for (String p : uniq) {
            jars.add(new File(p));
        }
        return jars;
    }

    // Find closest parent folder that contains pom.xml (bounded).
    private static File findNearestPomRoot(File start, int maxUp) {
        File p = start;
        for (int i = 0; i < maxUp && p != null; i++) {
            File pom = new File(p, "pom.xml");
            if (pom.exists()) return p;
            p = p.getParentFile();
        }
        return null;
    }

    // Heuristic: if src is deep (e.g., src/main/java), lift to a reasonable module root.
    // For Maven projects, keep within projectRoot (pom.xml root). For Gradle/no-pom, allow lifting freely.
    private static File guessModuleRoot(File src, File projectRoot) {
        try {
            boolean constrainToProjectRoot = projectRoot != null && new File(projectRoot, "pom.xml").exists();

            File cur = src;
            for (int i = 0; i < 10 && cur != null; i++) {
                String name = cur.getName().toLowerCase();
                if (name.equals("java")
                        && cur.getParentFile() != null
                        && cur.getParentFile().getName().equalsIgnoreCase("main")) {
                    File srcDir = cur.getParentFile().getParentFile(); // .../src
                    if (srcDir != null && srcDir.getParentFile() != null) {
                        File candidate = srcDir.getParentFile(); // <module>
                        if (!constrainToProjectRoot) {
                            return candidate;
                        }
                        if (projectRoot != null && candidate.getCanonicalPath().startsWith(projectRoot.getCanonicalPath())) {
                            return candidate;
                        }
                    }
                }
                cur = cur.getParentFile();
                if (constrainToProjectRoot && projectRoot != null && cur != null && cur.getCanonicalPath().equals(projectRoot.getCanonicalPath())) break;
            }
        } catch (Exception ignored) {
        }
        return projectRoot != null ? projectRoot : src;
    }

    private static void scanJarWhitelist(Path root, Set<String> uniq) {
        if (root == null) return;
        try {
            if (!Files.exists(root)) return;

            // Paths are matched by substring on normalized '/' string for portability.
            String[] includeFragments = new String[] {
                    "/target/dependency/",
                    "/target/lib/",
                    "/target/",
                    "/lib/",
                    "/libs/",
                    "/web-inf/lib/",
                    "/build/libs/",
                    "/build/dependency-jars/", // gradle dependency copy output (acmeair)
                    "/db2jars/",
                    "/liberty/",
                    "/wlp/",              // liberty runtime jars
                    "/usr/"               // liberty shared server/user dirs
            };

            try (Stream<Path> s = Files.walk(root)) {
                s.filter(Files::isRegularFile)
                        .filter(p -> p.toString().toLowerCase().endsWith(".jar"))
                        .forEach(p -> {
                            try {
                                String norm = p.toAbsolutePath().normalize().toString().replace('\\', '/');
                                String lower = norm.toLowerCase();
                                for (String frag : includeFragments) {
                                    if (lower.contains(frag)) {
                                        uniq.add(p.toFile().getCanonicalPath());
                                        return;
                                    }
                                }
                                // Also accept maven's local repo jars if you copied them under the project (rare), ignore otherwise.
                            } catch (Exception ignored) {
                            }
                        });
            }
        } catch (Exception e) {
            System.err.println("[Main] scanJarWhitelist failed under " + root + ": " + e.getMessage());
        }
    }

    public static void main(String[] args) throws Exception {
        if (args.length < 3) {
            System.err.println("Usage: java extractor.Main <extractorType> <srcPath> <outputPath>");
            System.exit(1);
        }

        String extractorType = args[0].toLowerCase();
        String srcPath = args[1];
        String outputPath = args[2];

        // Ensure the output directory exists.
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

                System.out.println("[OK] AST extraction finished. Total files: " + astList.size());
                if (!failedFiles.isEmpty()) {
                    System.err.println("[WARN] Failed to parse files: " + failedFiles);
                }
                break;
            }

            case "callgraph": {
                List<File> dependencyJars = collectDependencyJars(srcPath);
                System.out.println("Loaded dependency JARs:");
                for (File jar : dependencyJars) {
                    System.out.println("  " + jar.getAbsolutePath());
                }
                File jarLogFile = new File(outFile.getParentFile(), "loaded_jars.txt");
                try (java.io.PrintWriter jarLogWriter = new java.io.PrintWriter(jarLogFile, "UTF-8")) {
                    jarLogWriter.println("Loaded dependency JARs:");
                    for (File jar : dependencyJars) {
                        jarLogWriter.println(jar.getAbsolutePath());
                    }
                } catch (Exception e) {
                    System.err.println("Failed to write loaded_jars.txt: " + e.getMessage());
                }

                CallGraphExtractor extractor = new JavaParserCallGraphExtractor(srcPath, dependencyJars);
                DefaultDirectedGraph<String, DefaultEdge> graph = extractor.extractCallGraph(new File(srcPath));

                Map<String, Object> json = convertGraphToJson(graph);
                mapper.writerWithDefaultPrettyPrinter().writeValue(outFile, json);

                System.out.println("[OK] CallGraph extraction finished. Nodes: "
                        + graph.vertexSet().size() + " Edges: " + graph.edgeSet().size());
                System.out.println("Dependency JAR count: " + dependencyJars.size());
                break;
            }

            case "dependency": {
                List<File> dependencyJars = collectDependencyJars(srcPath);
                JavaParserDependencyGraphExtractor extractor = new JavaParserDependencyGraphExtractor(srcPath, dependencyJars);
                DefaultDirectedGraph<String, DefaultEdge> depGraph = extractor.extract(new File(srcPath));

                Map<String, Object> json = convertGraphToJson(depGraph);
                mapper.writerWithDefaultPrettyPrinter().writeValue(outFile, json);

                System.out.println("[OK] DependencyGraph extraction finished. Nodes: "
                        + depGraph.vertexSet().size() + " Edges: " + depGraph.edgeSet().size());
                System.out.println("Dependency JAR count: " + dependencyJars.size());
                break;
            }

            case "semantic": {
                List<File> dependencyJars = collectDependencyJars(srcPath);
                SemanticExtractor extractor = new JavaParserSemanticExtractor(srcPath, dependencyJars);
                extractor.extract(srcPath, outputPath);
                System.out.println("[OK] Semantic extraction finished. Output: " + outputPath);
                System.out.println("Dependency JAR count: " + dependencyJars.size());
                break;
            }


            default:
                System.err.println("Unknown extractor type: " + extractorType);
                System.exit(2);
        }
    }

    // Shared helper: convert a JGraphT graph to a JSON-serializable structure.
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
