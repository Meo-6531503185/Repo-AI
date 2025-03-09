import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;
import java.util.*;

class GraphTest {

    @Test
    void testAddEdge() {
        Graph graph = new Graph();
        graph.addEdge(0, 1);
        assertEquals("DFS traversal starting from node 0: 0 1 ", graph.dfs(0));
    }

    @Test
    void testDfs() {
        Graph graph = new Graph();
        graph.addEdge(0, 1);
        graph.addEdge(0, 2);
        graph.addEdge(1, 3);
        graph.addEdge(1, 4);
        graph.addEdge(2, 5);
        graph.addEdge(2, 6);
        assertEquals("DFS traversal starting from node 0: 0 1 3 4 2 5 6 ", graph.dfs(0));
    }

    @Test
    void testDfsSingleNode() {
        Graph graph = new Graph();
        graph.addEdge(0, 0); 
        assertEquals("DFS traversal starting from node 0: 0 ", graph.dfs(0));
    }

    @Test
    void testBfs() {
        Graph graph = new Graph();
        graph.addEdge(0, 1);
        graph.addEdge(0, 2);
        graph.addEdge(1, 3);
        graph.addEdge(1, 4);
        graph.addEdge(2, 5);
        graph.addEdge(2, 6);
        assertEquals("BFS traversal starting from node 0: 0 1 2 3 4 5 6 ", graph.bfs(0));
    }

    @Test
    void testBfsSingleNode() {
        Graph graph = new Graph();
        graph.addEdge(0, 0);
        assertEquals("BFS traversal starting from node 0: 0 ", graph.bfs(0));
    }
}
