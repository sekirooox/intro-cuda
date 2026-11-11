#include <cuda_runtime.h>

#include <iostream>
__global__ void test_shuf_broadcast(int *dOutput, const int *dInput,
                                    const int srcLane);
__global__ void test_shuf_down_sync(int *dOutput, const int *dInput, const int delta);
__global__ void test_shuf_up_sync(int *dOutput, const int *dInput, const int delta);

__global__ void test_shuf_broadcast(int *dOutput, const int *dInput,
                                    const int srcLane)
{
    int tid = threadIdx.x;
    int val = dInput[tid];

    // // 将srcLane的局部变量val的值，广播到整个warp中，作为val
    // int res = __shfl_sync(0xffffffff, val, srcLane);
    // dOutput[tid] = res;

    // 技巧：覆盖原先的寄存器值val
    val = __shfl_sync(0xffffffff, val, srcLane);
    dOutput[tid] = val;
}

__global__ void test_shuf_down_sync(int *dOutput, const int *dInput, const int delta)
{
    int tid = threadIdx.x;
    int val = dInput[tid];

    // 将当前线程的val值，向下移动delta个线程，作为val
    int res = __shfl_down_sync(0xffffffff, val, delta);
    dOutput[tid] = res;
}

__global__ void test_shuf_up_sync(int *dOutput, const int *dInput, const int delta)
{
    int tid = threadIdx.x;
    int val = dInput[tid];

    // 将当前线程的val值，向上移动delta个线程，作为val
    int res = __shfl_up_sync(0xffffffff, val, delta);
    dOutput[tid] = res;
}
int main()
{
    const int numThreads = 32;
    const int srcLane = 2;
    const int delta = 2;

    // Host arrays
    int hInput[numThreads];
    int hOutput[numThreads];

    // 初始化数组，每一个位置i的值等于下标i
    for (int i = 0; i < numThreads; ++i)
    {
        hInput[i] = i;
    }

    int *dInput, *dOutput;

    cudaMalloc(&dInput, numThreads * sizeof(int));
    cudaMalloc(&dOutput, numThreads * sizeof(int));

    cudaMemcpy(dInput, hInput, numThreads * sizeof(int), cudaMemcpyHostToDevice);

    // // 广播操作：将当前warp中指定LaneID的值，广播到整个warp中
    // test_shuf_broadcast<<<1, numThreads>>>(dOutput, dInput, srcLane);
    // std::cout << "Broadcasting value from thread " << srcLane << ":\n";

    // // 向下移动操作：将当前warp中每个线程的值，向下移动delta个线程
    // test_shuf_down_sync<<<1, numThreads>>>(dOutput, dInput, delta);
    // std::cout << "Shuffling down by " << delta << " threads:\n";

    test_shuf_up_sync<<<1, numThreads>>>(dOutput, dInput, delta);
    std::cout << "Shuffling up by " << delta << " threads:\n";

    cudaMemcpy(hOutput, dOutput, numThreads * sizeof(int),
               cudaMemcpyDeviceToHost);

    for (int i = 0; i < numThreads; ++i)
    {
        std::cout << "hOutput[" << i << "] = " << hOutput[i] << "\n";
    }

    cudaFree(dInput);
    cudaFree(dOutput);

    return 0;
}
