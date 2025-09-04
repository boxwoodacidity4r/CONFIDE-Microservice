package extractor.callgraph;

import org.jgrapht.graph.DefaultDirectedGraph;
import org.jgrapht.graph.DefaultEdge;
import java.io.File;

/**
 * CallGraphExtractor 接口定义
 *
 * 实现类需要提供从源码目录中构建函数调用图的能力
 */
public interface CallGraphExtractor {

    /**
     * 提取指定源码目录的函数调用图
     *
     * @param sourceRoot 源码根目录
     * @return DefaultDirectedGraph<String, DefaultEdge> 表示调用图
     * @throws Exception 如果解析或构建调用图失败
     */
    DefaultDirectedGraph<String, DefaultEdge> extractCallGraph(File sourceRoot) throws Exception;
}
