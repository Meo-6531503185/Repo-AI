import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;
import java.util.List;

class GraphTest {

    @Test
    void testConstructor() {
        Graph graph = new Graph(5);
        assertEquals(5, graph.adjacencyList.size());
        for (List<Integer> list : graph.adjacencyList) {
            assertTrue(list.isEmpty());
        }
    }

    @Test
    void testAddEdge() {
        Graph graph = new Graph(3);
        graph.addEdge(0, 1);
        graph.addEdge(1, 2);
        assertEquals(1, graph.adjacencyList.get(0).size());
        assertEquals(1, graph.adjacencyList.get(1).size());
        assertTrue(graph.adjacencyList.get(0).contains(1));
        assertTrue(graph.adjacencyList.get(1).contains(2));
    }

    @Test
    void testBfsEmptyGraph() {
        Graph graph = new Graph(0);
        graph.bfs(0);
        // No exceptions should be thrown
    }

    @Test
    void testBfsStartingFromZero() {
        Graph graph = new Graph(7);
        graph.addEdge(0, 1);
        graph.addEdge(0, 2);
        graph.addEdge(1, 3);
        graph.addEdge(1, 4);
        graph.addEdge(2, 5);
        graph.addEdge(2, 6);
        // Since bfs prints to the console, we can't directly assert its output.
        // Instead, we can visually inspect the console output or redirect it for testing.
        graph.bfs(0);
    }

    @Test
    void testBfsStartingFromNonZero() {
        Graph graph = new Graph(7);
        graph.addEdge(0, 1);
        graph.addEdge(0, 2);
        graph.addEdge(1, 3);
        graph.addEdge(1, 4);
        graph.addEdge(2, 5);
        graph.addEdge(2, 6);
        graph.bfs(2);
    }

    @Test
    void testBfsWithDisconnectedComponent() {
        Graph graph = new Graph(8);
        graph.addEdge(0, 1);
        graph.addEdge(0, 2);
        graph.addEdge(1, 3);
        graph.addEdge(1, 4);
        graph.addEdge(2, 5);
        graph.addEdge(2, 6);
        graph.addEdge(7, 7); // Disconnected component
        graph.bfs(0);
    }
}
