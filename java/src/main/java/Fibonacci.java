import java.util.Scanner;
public class Fibonacci{
public static void main(String[] args){
Scanner scanner=new Scanner(System.in);
System.out.print("Enter the number of elements: ");
int n=scanner.nextInt();
int[] arr=new int[n];
System.out.println("Enter the elements of the array: ");
for(int i=0;i<n;i++){
arr[i]=scanner.nextInt();
}
scanner.close();
bubbleSort(arr);
System.out.println("Sorted array:");
for(int i=0;i<n;i++){
System.out.print(arr[i]+" ");
}
}
public static void bubbleSort(int[] arr){
int n=arr.length;
for(int i=0;i<n-1;i++){
for(int j=0;j<n-i-1;j++){
if(arr[j]>arr[j+1]){
int temp=arr[j];
arr[j]=arr[j+1];
arr[j+1]=temp;
}
}
}
}
}