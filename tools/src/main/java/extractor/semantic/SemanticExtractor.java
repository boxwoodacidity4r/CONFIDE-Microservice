package extractor.semantic;

import java.io.IOException;

public interface SemanticExtractor {
    void extract(String projectPath, String outputPath) throws IOException;
}
