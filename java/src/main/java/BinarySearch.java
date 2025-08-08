public class BinarySearch {

    /**
     * Sorts an array of integers using the insertion sort algorithm.
     *
     * @param arr the array to be sorted
     */
    public static void insertionSort(int[] arr) {
        int n = arr.length;
        for (int i = 1; i < n; ++i) {
            int key = arr[i];
            int j = i - 1;

            // Move elements of arr[0..i-1] that are greater than key to one position ahead of their current position
            while (j >= 0 && arr[j] > key) {
                arr[j + 1] = arr[j];
                j = j - 1;
            }
            arr[j + 1] = key;
        }
    }

    /**
     * Main method to test the insertionSort function.
     *
     * @param args the command line arguments
     */
    public static void main(String[] args) {
        int[] arr = {11, 12, 22, 25, 34, 64, 90, 23, 5, 78, 2};
        insertionSort(arr);

        // Print the sorted array
        for (int num : arr) {
            System.out.print(num + " ");
        }
        System.out.println();
    }
}