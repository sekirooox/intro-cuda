#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>

#define N 10
void printout(int *a, int n)
{
    for (int i = 0; i < n; i++)
    {
        printf("%d ", a[i]);
    }
    printf("\n");
}
int main()
{
    int size = N * sizeof(int);
    int *h_a = (int *)malloc(size);
    int *d_a; // 要通过cuda runtime API在GPU上分配内存
    int *h_b = (int *)malloc(size);

    for (int i = 0; i < N; i++)
    {
        h_a[i] = i * i;
    }

    printf("h_a: ");
    printout(h_a, N);
    // 在CPU和GPU之间移动内存
    cudaMalloc((void **)&d_a, size);

    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);

    //   CPU不能直接访问GPU的内存
    //   printf("d_a: ");
    //   printout(d_a, N);

    cudaMemcpy(h_b, d_a, size, cudaMemcpyDeviceToHost);

    printf("h_b: ");
    printout(h_b, N);

    return 0;
}