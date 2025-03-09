import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;
class FibonacciTest {
    @Test
    void testBubbleSortEmptyArray() {
        int[] arr = {};
        Fibonacci.bubbleSort(arr);
        assertEquals(0, arr.length);
    }
    @Test
    void testBubbleSortSortedArray() {
        int[] arr = {1, 2, 3, 4, 5};
        Fibonacci.bubbleSort(arr);
        assertArrayEquals(new int[]{1, 2, 3, 4, 5}, arr);
    }
    @Test
    void testBubbleSortReverseSortedArray() {
        int[] arr = {5, 4, 3, 2, 1};
        Fibonacci.bubbleSort(arr);
        assertArrayEquals(new int[]{1, 2, 3, 4, 5}, arr);
    }
    @Test
    void testBubbleSortWithDuplicates() {
        int[] arr = {3, 1, 4, 1, 5, 9, 2, 6, 5};
        Fibonacci.bubbleSort(arr);
        assertArrayEquals(new int[]{1, 1, 2, 3, 4, 5, 5, 6, 9}, arr);
    }
} 
