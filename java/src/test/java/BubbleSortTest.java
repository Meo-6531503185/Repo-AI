import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;
import java.util.Arrays;

class BubbleSortTest {

    @Test
    void testBubbleSortWithPositiveNumbers() {
        int[] arr = {64, 34, 25, 12, 22, 11, 90};
        int[] expected = {11, 12, 22, 25, 34, 64, 90};
        BubbleSort.bubbleSort(arr);
        assertArrayEquals(expected, arr);
    }

    @Test
    void testBubbleSortWithNegativeNumbers() {
        int[] arr = {-5, -10, -2, -8, -1};
        int[] expected = {-10, -8, -5, -2, -1};
        BubbleSort.bubbleSort(arr);
        assertArrayEquals(expected, arr);
    }

    @Test
    void testBubbleSortWithMixedNumbers() {
        int[] arr = {10, -5, 0, 5, -10};
        int[] expected = {-10, -5, 0, 5, 10};
        BubbleSort.bubbleSort(arr);
        assertArrayEquals(expected, arr);
    }

    @Test
    void testBubbleSortWithDuplicateNumbers() {
        int[] arr = {3, 5, 2, 5, 1, 3};
        int[] expected = {1, 2, 3, 3, 5, 5};
        BubbleSort.bubbleSort(arr);
        assertArrayEquals(expected, arr);
    }

    @Test
    void testBubbleSortWithEmptyArray() {
        int[] arr = {};
        int[] expected = {};
        BubbleSort.bubbleSort(arr);
        assertArrayEquals(expected, arr);
    }

    @Test
    void testBubbleSortWithSingleElementArray() {
        int[] arr = {5};
        int[] expected = {5};
        BubbleSort.bubbleSort(arr);
        assertArrayEquals(expected, arr);
    }

    @Test
    void testBubbleSortWithSortedArray() {
        int[] arr = {1, 2, 3, 4, 5};
        int[] expected = {1, 2, 3, 4, 5};
        BubbleSort.bubbleSort(arr);
        assertArrayEquals(expected, arr);
    }

    @Test
    void testBubbleSortWithReverseSortedArray() {
        int[] arr = {5, 4, 3, 2, 1};
        int[] expected = {1, 2, 3, 4, 5};
        BubbleSort.bubbleSort(arr);
        assertArrayEquals(expected, arr);
    }
}
