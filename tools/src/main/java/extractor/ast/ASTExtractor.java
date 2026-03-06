package extractor.ast;

import com.github.javaparser.ast.CompilationUnit;
import java.io.File;

/**
 * ASTExtractor interface.
 * 
 * Implementations should provide the capability to extract an AST (Abstract Syntax Tree)
 * from a single Java source file.
 */
public interface ASTExtractor {

    /**
     * Extract the AST of the given Java file.
     *
     * @param file Java source file to parse
     * @return CompilationUnit representing the AST
     * @throws Exception if parsing fails
     */
    CompilationUnit extractAST(File file) throws Exception;
}
