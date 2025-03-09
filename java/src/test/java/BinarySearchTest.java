import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

class BinarySearchTest {

    @Test
    void testTargetFound() {
        int[] arr = {11, 12, 22, 25, 34, 64, 90};
        int target = 25;
        int expectedIndex = 3;
        int actualIndex = BinarySearch.binarySearch(arr, target);
        assertEquals(expectedIndex, actualIndex);
    }

    @Test
    void testTargetNotFound() {
        int[] arr = {11, 12, 22, 25, 34, 64, 90};
        int target = 23;
        int expectedIndex = -1;
        int actualIndex = BinarySearch.binarySearch(arr, target);
        assertEquals(expectedIndex, actualIndex);
    }

    @Test
    void testEmptyArray() {
        int[] arr = {};
        int target = 25;
        int expectedIndex = -1;
        int actualIndex = BinarySearch.binarySearch(arr, target);
        assertEquals(expectedIndex, actualIndex);
    }

    @Test
    void testTargetAtBeginning() {
        int[] arr = {11, 12, 22, 25, 34, 64, 90};
        int target = 11;
        int expectedIndex = 0;
        int actualIndex = BinarySearch.binarySearch(arr, target);
        assertEquals(expectedIndex, actualIndex);
    }

    @Test
    void testTargetAtEnd() {
        int[] arr = {11, 12, 22, 25, 34, 64, 90};
        int target = 90;
        int expectedIndex = 6;
        int actualIndex = BinarySearch.binarySearch(arr, target);
        assertEquals(expectedIndex, actualIndex);
    }
} 
