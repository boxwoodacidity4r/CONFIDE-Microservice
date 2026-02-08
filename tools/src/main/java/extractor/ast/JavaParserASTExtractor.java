package extractor.ast;

import com.github.javaparser.JavaParser;
import com.github.javaparser.ast.CompilationUnit;
import com.github.javaparser.ast.Node;
import com.github.javaparser.symbolsolver.JavaSymbolSolver;
import com.github.javaparser.symbolsolver.resolution.typesolvers.CombinedTypeSolver;
import com.github.javaparser.symbolsolver.resolution.typesolvers.ReflectionTypeSolver;
import com.github.javaparser.symbolsolver.resolution.typesolvers.JavaParserTypeSolver;
import com.github.javaparser.symbolsolver.resolution.typesolvers.JarTypeSolver;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileWriter;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;
import java.util.stream.Stream;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;

public class JavaParserASTExtractor implements ASTExtractor {

    private final JavaParser parser;

    public JavaParserASTExtractor(String sourceRootPath) {
        this(sourceRootPath, null);
    }

    public JavaParserASTExtractor(String sourceRootPath, List<File> dependencyJars) {
        CombinedTypeSolver typeSolver = new CombinedTypeSolver();
        typeSolver.add(new ReflectionTypeSolver());
        typeSolver.add(new JavaParserTypeSolver(new File(sourceRootPath)));
        if (dependencyJars != null) {
            for (File jar : dependencyJars) {
                try {
                    typeSolver.add(new JarTypeSolver(jar.getAbsolutePath()));
                    System.out.println("[AST] Added JarTypeSolver: " + jar.getAbsolutePath());
                } catch (Exception e) {
                    System.err.println("[AST] Failed to add JarTypeSolver for " + jar + ": " + e.getMessage());
                }
            }
        }
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

    private static boolean isProjectInternal(String fqn) {
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

    private static boolean isNoisyRootBase(String fqnOrSimpleName) {
        if (fqnOrSimpleName == null) return false;
        String s = fqnOrSimpleName.trim();
        String simple = s.contains(".") ? s.substring(s.lastIndexOf('.') + 1) : s;
        String lower = simple.toLowerCase();
        // 常见“单根基类”模式：EntityBase/Base*/Abstract*
        if (lower.equals("entitybase")) return true;
        if (lower.startsWith("base") || lower.endsWith("base")) return true;
        if (lower.startsWith("abstract")) return true;
        return false;
    }

    // =========================
    // 提取所有类的全限定名、父类、接口信息，输出为 {class: {bases: [...], interfaces: [...]}} 
    // =========================
    public static Map<String, Map<String, List<String>>> extractClassHierarchy(CompilationUnit cu) {
        Map<String, Map<String, List<String>>> result = new HashMap<>();
        String packageName = cu.getPackageDeclaration().map(pd -> pd.getNameAsString()).orElse("");
        cu.findAll(com.github.javaparser.ast.body.ClassOrInterfaceDeclaration.class).forEach(cls -> {
            String className = cls.getNameAsString();
            String fullName = packageName.isEmpty() ? className : packageName + "." + className;
            List<String> bases = new ArrayList<>();
            cls.getExtendedTypes().forEach(t -> {
                String extName = t.getNameAsString();
                String extFull = packageName.isEmpty() ? extName : packageName + "." + extName;
                // 过滤框架/单根基类噪声
                if (!isProjectInternal(extFull)) return;
                if (isNoisyRootBase(extFull) || isNoisyRootBase(extName)) return;
                bases.add(extFull);
            });
            List<String> interfaces = new ArrayList<>();
            cls.getImplementedTypes().forEach(t -> {
                String implName = t.getNameAsString();
                String implFull = packageName.isEmpty() ? implName : packageName + "." + implName;
                if (!isProjectInternal(implFull)) return;
                interfaces.add(implFull);
            });
            Map<String, List<String>> info = new HashMap<>();
            info.put("bases", bases);
            info.put("interfaces", interfaces);
            result.put(fullName, info);
        });
        return result;
    }

    /**
     * 批量导出AST继承/接口信息，输出格式为 {FQN: {bases: [...], interfaces: [...]}}
     * @param sourceDir 源代码根目录
     * @param outputFile 输出JSON文件路径（注意：是文件，不是目录）
     */
    public static void exportBatchAST(String sourceDir, String outputFile, List<File> dependencyJars) throws Exception {
        JavaParserASTExtractor extractor = new JavaParserASTExtractor(sourceDir, dependencyJars);
        Gson gson = new GsonBuilder().disableHtmlEscaping().setPrettyPrinting().create();

        Map<String, Map<String, List<String>>> all = new HashMap<>();

        try (Stream<Path> paths = Files.walk(Paths.get(sourceDir))) {
            List<Path> javaFiles = paths.filter(Files::isRegularFile)
                    .filter(p -> p.toString().endsWith(".java"))
                    .collect(Collectors.toList());

            for (Path javaPath : javaFiles) {
                File javaFile = javaPath.toFile();
                try {
                    CompilationUnit cu = extractor.extractAST(javaFile);
                    Map<String, Map<String, List<String>>> local = extractClassHierarchy(cu);
                    for (Map.Entry<String, Map<String, List<String>>> e : local.entrySet()) {
                        // 后遍历的覆盖先前的，相同FQN类以最后一次为准
                        all.put(e.getKey(), e.getValue());
                    }
                } catch (Exception e) {
                    System.err.println("[AST] Failed to parse " + javaFile.getAbsolutePath() + ": " + e.getMessage());
                }
            }
        }

        File outFile = new File(outputFile);
        if (outFile.getParentFile() != null) {
            outFile.getParentFile().mkdirs();
        }
        try (FileWriter writer = new FileWriter(outFile)) {
            gson.toJson(all, writer);
        }
        System.out.println("[AST] Exported " + all.size() + " classes to " + outputFile);
    }

    /**
     * 严格兼容ASTBatchExporter格式，仅保留type、name、children、range字段
     */
    public static Map<String, Object> nodeToASTBatchExporterFormat(Node node) {
        Map<String, Object> map = new HashMap<>();
        map.put("type", node.getClass().getSimpleName());
        // name字段（如有）
        if (node instanceof com.github.javaparser.ast.body.ClassOrInterfaceDeclaration) {
            map.put("name", ((com.github.javaparser.ast.body.ClassOrInterfaceDeclaration) node).getNameAsString());
        } else if (node instanceof com.github.javaparser.ast.body.MethodDeclaration) {
            map.put("name", ((com.github.javaparser.ast.body.MethodDeclaration) node).getNameAsString());
        } else if (node instanceof com.github.javaparser.ast.body.FieldDeclaration) {
            map.put("name", node.toString()); // 可自定义字段名提取
        }
        map.put("range", node.getRange().map(r -> r.toString()).orElse(""));
        List<Object> children = new ArrayList<>();
        for (Node child : node.getChildNodes()) {
            children.add(nodeToASTBatchExporterFormat(child));
        }
        map.put("children", children);
        return map;
    }

    // 可选：main方法，支持命令行批量导出
    public static void main(String[] args) throws Exception {
        if (args.length < 2) {
            System.out.println("用法: JavaParserASTExtractor <sourceDir> <outputFile> [dependencyJar1,dependencyJar2,...]");
            return;
        }
        String sourceDir = args[0];
        String outputFile = args[1];
        List<File> jars = null;
        if (args.length > 2) {
            jars = Stream.of(args[2].split(",")).map(File::new).collect(Collectors.toList());
        }
        exportBatchAST(sourceDir, outputFile, jars);
    }
}
