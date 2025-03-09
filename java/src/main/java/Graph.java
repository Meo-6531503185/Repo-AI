import java.util.*;
public class Graph {
    private Map<Integer, List<Integer>> adjacencyList;
    public Graph() {
        adjacencyList = new HashMap<>();
    }
    public void addEdge(int source, int destination) {
        adjacencyList.putIfAbsent(source, new ArrayList<>());
        adjacencyList.putIfAbsent(destination, new ArrayList<>());
        adjacencyList.get(source).add(destination);
    }
    public String dfs(int start) {
        Set<Integer> visited = new HashSet<>();
        StringBuilder sb = new StringBuilder("DFS traversal starting from node " + start + ": ");
        dfsHelper(start, visited, sb);
        return sb.toString();
    }
    private void dfsHelper(int node, Set<Integer> visited, StringBuilder sb) {
        if (visited.contains(node)) {
            return;
        }
        sb.append(node).append(" ");
        visited.add(node);
        List<Integer> neighbors = adjacencyList.getOrDefault(node, Collections.emptyList());
        for (int neighbor : neighbors) {
            dfsHelper(neighbor, visited, sb);
        }
    }
    public String bfs(int start) {
        Set<Integer> visited = new HashSet<>();
        Queue<Integer> queue = new LinkedList<>();
        visited.add(start);
        queue.offer(start);
        StringBuilder sb = new StringBuilder("BFS traversal starting from node " + start + ": ");
        while (!queue.isEmpty()) {
            int currentNode = queue.poll();
            sb.append(currentNode).append(" ");
            List<Integer> neighbors = adjacencyList.getOrDefault(currentNode, Collections.emptyList());
            for (int neighbor : neighbors) {
                if (!visited.contains(neighbor)) {
                    visited.add(neighbor);
                    queue.offer(neighbor);
                }
            }
        }
        return sb.toString();
    }
    public static void main(String[] args) {
        Graph graph = new Graph();
        graph.addEdge(0, 1);
        graph.addEdge(0, 2);
        graph.addEdge(1, 3);
        graph.addEdge(1, 4);
        graph.addEdge(2, 5);
        graph.addEdge(2, 6);
        System.out.println();
        System.out.println(graph.dfs(0));
        System.out.println();
        System.out.println(graph.bfs(0));
    }
}
