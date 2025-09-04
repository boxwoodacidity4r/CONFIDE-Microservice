package extractor.dependency;

import java.io.File;
import java.io.IOException;

public interface DependencyGraphExtractor {
    void extract(File projectDir, File outputFile) throws IOException;
}
