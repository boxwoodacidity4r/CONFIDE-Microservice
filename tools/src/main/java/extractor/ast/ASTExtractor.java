package extractor.ast;

import com.github.javaparser.ast.CompilationUnit;
import java.io.File;

/**
 * ASTExtractor 接口定义
 * 
 * 实现类需要提供从单个 Java 文件中提取 AST（抽象语法树）的能力
 */
public interface ASTExtractor {

    /**
     * 提取指定 Java 文件的 AST
     *
     * @param file 待解析的 Java 文件
     * @return CompilationUnit 表示 AST
     * @throws Exception 如果解析失败抛出异常
     */
    CompilationUnit extractAST(File file) throws Exception;
}
