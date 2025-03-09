import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

class KnapsackTest {

    @Test
    void testKnapsack_EmptyItems() {
        int[] weights = {};
        int[] values = {};
        int capacity = 10;
        int expectedValue = 0;
        int actualValue = Knapsack.knapsack(weights, values, capacity);
        assertEquals(expectedValue, actualValue);
    }

    @Test
    void testKnapsack_ZeroCapacity() {
        int[] weights = {1, 2, 3};
        int[] values = {10, 20, 30};
        int capacity = 0;
        int expectedValue = 0;
        int actualValue = Knapsack.knapsack(weights, values, capacity);
        assertEquals(expectedValue, actualValue);
    }

    @Test
    void testKnapsack_ValidInput() {
        int[] weights = {10, 20, 30};
        int[] values = {60, 100, 120};
        int capacity = 50;
        int expectedValue = 220;
        int actualValue = Knapsack.knapsack(weights, values, capacity);
        assertEquals(expectedValue, actualValue);
    }

    @Test
    void testKnapsack_EdgeCase() {
        int[] weights = {50, 49, 48, 51};
        int[] values = {100, 99, 98, 101};
        int capacity = 100;
        int expectedValue = 200;
        int actualValue = Knapsack.knapsack(weights, values, capacity);
        assertEquals(expectedValue, actualValue);
    }
}
