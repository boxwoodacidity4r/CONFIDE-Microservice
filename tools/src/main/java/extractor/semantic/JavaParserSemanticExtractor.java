package extractor.semantic;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.github.javaparser.JavaParser;
import com.github.javaparser.ParserConfiguration;
import com.github.javaparser.ParseProblemException;
import com.github.javaparser.Position;
import com.github.javaparser.Range;
import com.github.javaparser.ast.CompilationUnit;
import com.github.javaparser.ast.body.*;
import com.github.javaparser.ast.comments.Comment;
import com.github.javaparser.ast.PackageDeclaration;
import com.github.javaparser.symbolsolver.JavaSymbolSolver;
import com.github.javaparser.symbolsolver.resolution.typesolvers.CombinedTypeSolver;
import com.github.javaparser.symbolsolver.resolution.typesolvers.ReflectionTypeSolver;
import com.github.javaparser.symbolsolver.resolution.typesolvers.JavaParserTypeSolver;
import com.github.javaparser.symbolsolver.resolution.typesolvers.JarTypeSolver;

import java.io.File;
import java.io.IOException;
import java.nio.file.*;
import java.util.*;

public class JavaParserSemanticExtractor implements SemanticExtractor {
    private final JavaParser parser;
    private final ObjectMapper mapper = new ObjectMapper();

    public JavaParserSemanticExtractor(String sourceRootPath, List<File> dependencyJars) {
        CombinedTypeSolver typeSolver = new CombinedTypeSolver();
        typeSolver.add(new ReflectionTypeSolver());
        typeSolver.add(new JavaParserTypeSolver(new File(sourceRootPath)));
        if (dependencyJars != null) {
            for (File jar : dependencyJars) {
                try {
                    typeSolver.add(new JarTypeSolver(jar.getAbsolutePath()));
                    System.out.println("[Semantic] Added JarTypeSolver: " + jar.getAbsolutePath());
                } catch (Exception e) {
                    System.err.println("[Semantic] Failed to add JarTypeSolver for " + jar + ": " + e.getMessage());
                }
            }
        }
        ParserConfiguration config = new ParserConfiguration();
        config.setSymbolResolver(new JavaSymbolSolver(typeSolver));
        this.parser = new JavaParser(config);
    }

    @Override
    public void extract(String projectPath, String outputPath) throws IOException {
        List<Map<String, Object>> methodsInfo = new ArrayList<>();
        Set<String> parseFailed = new HashSet<>();
        int totalFiles = 0;
        int parsedFiles = 0;
        int filesWithMethods = 0;

        Path root = Paths.get(projectPath);
        List<Path> javaFiles = new ArrayList<>();
        Files.walk(root)
                .filter(p -> p.toString().endsWith(".java"))
                .forEach(javaFiles::add);

        totalFiles = javaFiles.size();

        for (Path file : javaFiles) {
            CompilationUnit cu = null;
            boolean parseError = false;
            try {
                cu = parser.parse(file).getResult().orElse(null);
            } catch (ParseProblemException e) {
                parseFailed.add(root.relativize(file).toString() + " (ParseProblemException: " + e.getMessage() + ")");
                parseError = true;
            } catch (Exception e) {
                parseFailed.add(root.relativize(file).toString() + " (Exception: " + e.getMessage() + ")");
                parseError = true;
            }
            if (cu == null) {
                if (!parseError) {
                    parseFailed.add(root.relativize(file).toString() + " (null CompilationUnit)");
                }
                continue;
            }
            parsedFiles++;

            String pkg = cu.getPackageDeclaration().map(PackageDeclaration::getNameAsString).orElse("");
            String sourceFile = root.relativize(file).toString();

            // Recursively collect all type declarations (classes/interfaces/enums), including inner types.
            List<TypeDeclaration<?>> allTypes = new ArrayList<>();
            cu.findAll(TypeDeclaration.class).forEach(allTypes::add);

            int methodCountInFile = 0;
            Set<String> seenMethods = new HashSet<>(); // used for deduplication

            for (TypeDeclaration<?> type : allTypes) {
                String className = getFQN(type, pkg, sourceFile);
                if (className == null || className.isEmpty()) {
                    className = "Unresolved::" + sourceFile.replace(File.separatorChar, '.');
                }

                if (type.getMembers() != null) {
                    for (BodyDeclaration<?> member : type.getMembers()) {
                        if (member.isMethodDeclaration()) {
                            MethodDeclaration method = member.asMethodDeclaration();
                            String methodKey = getMethodKey(method);
                            if (seenMethods.add(methodKey)) {
                                Map<String, Object> methodInfo = extractMethodInfo(method, className, pkg, sourceFile);
                                methodsInfo.add(methodInfo);
                                methodCountInFile++;
                            }
                        }
                    }
                }
            }

            // Fallback strategy: if the file contains methods but they cannot be attributed to any class.
            // Only extract methods that have not been attributed yet.
            List<MethodDeclaration> orphanMethods = cu.findAll(MethodDeclaration.class);
            for (MethodDeclaration method : orphanMethods) {
                String methodKey = getMethodKey(method);
                if (!seenMethods.contains(methodKey)) {
                    Map<String, Object> methodInfo = extractMethodInfo(
                            method,
                            "Unresolved::" + sourceFile.replace(File.separatorChar, '.'),
                            pkg,
                            sourceFile
                    );
                    methodsInfo.add(methodInfo);
                    methodCountInFile++;
                    seenMethods.add(methodKey);
                }
            }

            if (methodCountInFile > 0) filesWithMethods++;
        }

        mapper.writerWithDefaultPrettyPrinter().writeValue(new File(outputPath), methodsInfo);

        // Print summary statistics
        System.out.println("Semantic Extraction Summary:");
        System.out.println("- Total Java files: " + totalFiles);
        System.out.println("- Parsed successfully: " + parsedFiles);
        System.out.println("- Files with methods: " + filesWithMethods);
        System.out.println("- Parse failed: " + parseFailed.size());
        if (!parseFailed.isEmpty()) {
            System.out.println("Parse failed files:");
            parseFailed.forEach(System.out::println);
        }
        System.out.println("[OK] Semantic features saved to " + outputPath);
    }

    // Get fully-qualified name (FQN)
    private String getFQN(TypeDeclaration<?> type, String pkg, String sourceFile) {
        Deque<String> names = new ArrayDeque<>();
        TypeDeclaration<?> current = type;
        while (current != null) {
            names.addFirst(current.getNameAsString());
            if (current.getParentNode().isPresent() && current.getParentNode().get() instanceof TypeDeclaration) {
                current = (TypeDeclaration<?>) current.getParentNode().get();
            } else {
                break;
            }
        }
        String fqn = String.join(".", names);
        if (pkg != null && !pkg.isEmpty()) {
            fqn = pkg + "." + fqn;
        }
        if (fqn == null || fqn.isEmpty()) {
            fqn = "Unresolved::" + sourceFile.replace(File.separatorChar, '.');
        }
        return fqn;
    }

    // Method unique identifier (based on Range)
    private String getMethodKey(MethodDeclaration method) {
        Optional<Range> range = method.getRange();
        return method.getNameAsString() + ":" +
                range.map(r -> r.begin.line + "-" + r.end.line).orElse("unknown");
    }

    // Extract method semantic information
    private Map<String, Object> extractMethodInfo(MethodDeclaration method, String className, String pkg, String sourceFile) {
        Map<String, Object> methodInfo = new LinkedHashMap<>();
        methodInfo.put("class", className);
        methodInfo.put("method", method.getNameAsString());
        methodInfo.put("body", method.getBody().map(Object::toString).orElse(""));
        List<String> variables = new ArrayList<>();
        method.findAll(VariableDeclarator.class).forEach(v -> variables.add(v.getNameAsString()));
        methodInfo.put("variables", variables);
        String comment = method.getComment().map(Comment::getContent).orElse("").trim();
        methodInfo.put("comments", comment);
        methodInfo.put("sourceFile", sourceFile);
        methodInfo.put("package", pkg);
        // Line number range
        Optional<Range> range = method.getRange();
        methodInfo.put("beginLine", range.map(r -> r.begin.line).orElse(-1));
        methodInfo.put("endLine", range.map(r -> r.end.line).orElse(-1));
        return methodInfo;
    }
}