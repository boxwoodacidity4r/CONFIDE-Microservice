package extractor.callgraph;

import org.jgrapht.graph.DefaultDirectedGraph;
import org.jgrapht.graph.DefaultEdge;
import java.io.File;

/**
 * CallGraphExtractor interface.
 *
 * Implementations should provide the capability to build a function/method call graph
 * from a source code directory.
 */
public interface CallGraphExtractor {

    /**
     * Extract the call graph from the given source root directory.
     *
     * @param sourceRoot source root directory
     * @return call graph as a directed graph
     * @throws Exception if parsing or graph construction fails
     */
    DefaultDirectedGraph<String, DefaultEdge> extractCallGraph(File sourceRoot) throws Exception;
}
