import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

import java.util.Arrays;

public class BinarySearchTest {

    @Test
    void insertionSort_emptyArray() {
        int[] arr = {};
        BinarySearch.insertionSort(arr);
        assertArrayEquals(new int[]{}, arr);
    }

    @Test
    void insertionSort_singleElementArray() {
        int[] arr = {5};
        BinarySearch.insertionSort(arr);
        assertArrayEquals(new int[]{5}, arr);
    }

    @Test
    void insertionSort_alreadySortedArray() {
        int[] arr = {1, 2, 3, 4, 5};
        BinarySearch.insertionSort(arr);
        assertArrayEquals(new int[]{1, 2, 3, 4, 5}, arr);
    }

    @Test
    void insertionSort_reverseSortedArray() {
        int[] arr = {5, 4, 3, 2, 1};
        BinarySearch.insertionSort(arr);
        assertArrayEquals(new int[]{1, 2, 3, 4, 5}, arr);
    }


    @Test
    void insertionSort_unsortedArray() {
        int[] arr = {11, 12, 22, 25, 34, 64, 90, 23, 5, 78, 2};
        BinarySearch.insertionSort(arr);
        assertArrayEquals(new int[]{2, 5, 11, 12, 22, 23, 25, 34, 64, 78, 90}, arr);
    }

    @Test
    void insertionSort_duplicateElements() {
        int[] arr = {5, 2, 8, 2, 5, 1};
        BinarySearch.insertionSort(arr);
        assertArrayEquals(new int[]{1, 2, 2, 5, 5, 8}, arr);

    }

    @Test
    void insertionSort_negativeElements() {
        int[] arr = {-5, -2, -8, -2, -5, -1};
        BinarySearch.insertionSort(arr);
        assertArrayEquals(new int[]{-8, -5, -5, -2, -2, -1}, arr);
    }



}
