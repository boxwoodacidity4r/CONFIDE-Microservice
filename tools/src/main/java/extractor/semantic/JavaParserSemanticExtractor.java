package extractor.semantic;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.github.javaparser.JavaParser;
import com.github.javaparser.ast.CompilationUnit;
import com.github.javaparser.ast.body.ClassOrInterfaceDeclaration;
import com.github.javaparser.ast.body.MethodDeclaration;
import com.github.javaparser.ast.body.VariableDeclarator;
import com.github.javaparser.ast.comments.Comment;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.*;

public class JavaParserSemanticExtractor implements SemanticExtractor {

    private final JavaParser parser = new JavaParser();
    private final ObjectMapper mapper = new ObjectMapper();

    @Override
    public void extract(String projectPath, String outputPath) throws IOException {
        List<Map<String, Object>> methodsInfo = new ArrayList<>();

        Files.walk(Path.of(projectPath))
                .filter(p -> p.toString().endsWith(".java"))
                .forEach(file -> {
                    try {
                        CompilationUnit cu = parser.parse(file).getResult().orElse(null);
                        if (cu == null) return;

                        // 找到类名（可能有多个 class/interface）
                        cu.findAll(ClassOrInterfaceDeclaration.class).forEach(clazz -> {
                            String className = clazz.getNameAsString();

                            clazz.findAll(MethodDeclaration.class).forEach(method -> {
                                Map<String, Object> methodInfo = new LinkedHashMap<>();
                                methodInfo.put("class", className);
                                methodInfo.put("method", method.getNameAsString());

                                // 方法体源码
                                String body = method.getBody().map(Object::toString).orElse("");
                                methodInfo.put("body", body);

                                // 局部变量
                                List<String> variables = new ArrayList<>();
                                method.findAll(VariableDeclarator.class).forEach(v ->
                                        variables.add(v.getNameAsString())
                                );
                                methodInfo.put("variables", variables);

                                // 方法注释（包括 javadoc）
                                String comment = method.getComment().map(Comment::getContent).orElse("").trim();
                                methodInfo.put("comments", comment);

                                methodsInfo.add(methodInfo);
                            });
                        });

                    } catch (Exception e) {
                        System.err.println("⚠️ Failed to parse: " + file + " due to " + e.getMessage());
                    }
                });

        // 输出 JSON
        mapper.writerWithDefaultPrettyPrinter().writeValue(new File(outputPath), methodsInfo);
        System.out.println("✅ Semantic features saved to " + outputPath);
    }
}
