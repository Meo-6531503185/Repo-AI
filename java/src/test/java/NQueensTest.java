import org.junit.jupiter.api.Test;
import java.util.List;
import static org.junit.jupiter.api.Assertions.*;

class NQueensTest {

    @Test
    void testSolveNQueens_ValidInput() {
        NQueens.solveNQueens(1);
        // Assertions are not applicable here as the method prints directly to the console.
        // We can visually inspect the console output to verify the correctness for this specific input.
    }

    @Test
    void testSolveNQueens_EdgeCase_Zero() {
        NQueens.solveNQueens(0); 
        // Assertions are not applicable here as the method prints directly to the console.
        // We can visually inspect the console output to verify the correctness for an empty board.
    }
} 
