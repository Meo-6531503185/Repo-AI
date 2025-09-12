import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

class BinarySearchTest {
    @Test
    void testMergeSort_EmptyArray() {
        int[] arr = {};
        BinarySearch.mergeSort(arr);
        assertArrayEquals(new int[] {}, arr);
    }

    @Test
    void testMergeSort_SingleElementArray() {
        int[] arr = { 1 };
        BinarySearch.mergeSort(arr);
        assertArrayEquals(new int[] { 1 }, arr);
    }

    @Test
    void testMergeSort_SortedArray() {
        int[] arr = { 1, 2, 3, 4, 5 };
        BinarySearch.mergeSort(arr);
        assertArrayEquals(new int[] { 1, 2, 3, 4, 5 }, arr);
    }

    @Test
    void testMergeSort_ReverseSortedArray() {
        int[] arr = { 5, 4, 3, 2, 1 };
        BinarySearch.mergeSort(arr);
        assertArrayEquals(new int[] { 1, 2, 3, 4, 5 }, arr);
    }

    @Test
    void testMergeSort_UnsortedArray() {
        int[] arr = { 3, 1, 4, 2, 5 };
        BinarySearch.mergeSort(arr);
        assertArrayEquals(new int[] { 1, 2, 3, 4, 5 }, arr);
    }

    @Test
    void testMergeSort_ArrayWithDuplicates() {
        int[] arr = { 2, 5, 1, 3, 2, 4, 5 };
        BinarySearch.mergeSort(arr);
        assertArrayEquals(new int[] { 1, 2, 2, 3, 4, 5, 5 }, arr);
    }
}
