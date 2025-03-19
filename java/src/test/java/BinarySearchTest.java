import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;
class BinarySearchTest {
    @Test
    void testInsertionSortEmptyArray() {
        int[] arr = {};
        BinarySearch.insertionSort(arr);
        assertEquals(0, arr.length);
    }
    @Test
    void testInsertionSortSortedArray() {
        int[] arr = {1, 2, 3, 4, 5};
        BinarySearch.insertionSort(arr);
        assertArrayEquals(new int[]{1, 2, 3, 4, 5}, arr);
    }
    @Test
    void testInsertionSortReverseSortedArray() {
        int[] arr = {5, 4, 3, 2, 1};
        BinarySearch.insertionSort(arr);
        assertArrayEquals(new int[]{1, 2, 3, 4, 5}, arr);
    }
    @Test
    void testInsertionSortDuplicateElements() {
        int[] arr = {2, 1, 3, 2, 4, 1, 5};
        BinarySearch.insertionSort(arr);
        assertArrayEquals(new int[]{1, 1, 2, 2, 3, 4, 5}, arr);
    }
}
