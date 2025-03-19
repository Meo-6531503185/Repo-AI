import java.util.*;
class NQueens {
    public static void solveNQueens(int n) {
        List<List<String>> solutions = new ArrayList<>();
        char[][] board = new char[n][n];
        for (int i = 0; i < n; i++) {
            Arrays.fill(board[i], '.');
        }
        backtrack(board, 0, solutions);
        printSolutions(solutions);
    }

    private static void backtrack(char[][] board, int row, List<List<String>> solutions) {
        if (row == board.length) {
            solutions.add(constructBoard(board));
            return;
        }
        for (int col = 0; col < board.length; col++) {
            if (isValid(board, row, col)) {
                board[row][col] = 'Q';
                backtrack(board, row + 1, solutions);
                board[row][col] = '.';
            }
        }
    }

    private static boolean isValid(char[][] board, int row, int col) {
        int n = board.length;
        for (int i = 0; i < row; i++) {
            if (board[i][col] == 'Q') {
                return false;
            }
        }
        for (int i = row - 1, j = col - 1; i >= 0 && j >= 0; i--, j--) {
            if (board[i][j] == 'Q') {
                return false;
            }
        }
        for (int i = row - 1, j = col + 1; i >= 0 && j < n; i--, j++) {
            if (board[i][j] == 'Q') {
                return false;
            }
        }
        return true;
    }

    private static List<String> constructBoard(char[][] board) {
        List<String> solution = new ArrayList<>();
        for (char[] row : board) {
            solution.add(new String(row));
        }
        return solution;
    }

    private static void printSolutions(List<List<String>> solutions) {
        for (List<String> solution : solutions) {
            System.out.println(solution);
        }
    }

    public static void main(String[] args) {
        solveNQueens(4);
    }
}
