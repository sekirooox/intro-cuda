#include <cuda_runtime.h>

#include <chrono> // 用于 CPU 计时
#include <iostream>
#include <numeric>
#include <vector>

const int BLOCK_SIZE = 1024;
const int N = 1024 * 1024; // 1M elements

// Reduction：用于求和
__global__ void reduce_v0(float *g_idata, float *g_odata) // 1D数组
{
    // 共享内存：大小为C
    __shared__ float sm[BLOCK_SIZE];

    int bid = blockIdx.x;
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x; // 2D->1D

    // 将输入数据放在显存上
    if (idx + tid < N)
    {
        sm[tid] = g_idata[idx + tid];
    }
    __syncthreads();

    // 归约求和
    for (int stride = 1; stride < BLOCK_SIZE; stride *= 2) // 最高stride为blockDim.x/2
    {
        if (tid % (2 * stride) == 0)
        {
            sm[tid] += sm[tid + stride];
        }
        __syncthreads();
    }
    if (tid == 0)
    {
        g_odata[bid] = sm[0];
    }
}

// 问题0：第一次，tid % (2 * stride) == 0 使得始终有一半的线程闲置了
// 解决：在第一次归约时，让闲置的一半线程有事情干
// 先让所有进程进行一次归约求和，处理的维度从Blocksize，变为BlockSize*2
__global__ void reduce_v1(float *g_idata, float *g_odata)
{
    int bid = blockIdx.x;
    int tid = threadIdx.x;
    // 现在处理的单位是BlockSize*2，也就是行元素个数为BlockSize*2
    int idx = blockIdx.x * blockDim.x * 2;

    // 共享内存的大小保持不变
    __shared__ float sm[BLOCK_SIZE];

    if (idx + blockDim.x + tid < N)
    {
        sm[tid] = g_idata[idx + tid] + g_idata[idx + blockDim.x + tid];
    }
    __syncthreads();

    // 归约求和
    for (int stride = 1; stride < blockDim.x; stride *= 2) // 最高stride为blockDim.x/2
    {
        if (tid % (2 * stride) == 0)
        {
            sm[tid] += sm[tid + stride];
        }
        __syncthreads();
    }
    if (tid == 0)
    {
        g_odata[bid] = sm[0];
    }
}

// 问题1：tid % (2 * stride)==0，线程分化问题
// stride = 1为例，tid = 0, 2, 4, 6, 8, ..., 32, ...
// 一个warp之内，存在活跃和非活跃线程 (行为分化)，GPU需要分批处理这批分化的线程

// 问题2：sm[tid] += sm[tid + stride]，访存方式有问题
// bank冲突指的是：同一个warp内的线程访问共享内存时，
// 如果多个线程访问了同一个bank，就会发生bank冲突，导致访问延迟增加。
__global__ void reduce_v2(float *g_idata, float *g_odata)
{
    int bid = blockIdx.x;
    int tid = threadIdx.x;
    // 现在处理的单位是BlockSize*2，也就是行元素个数为BlockSize*2
    int idx = blockIdx.x * blockDim.x * 2;

    // 共享内存的大小保持不变
    __shared__ float sm[BLOCK_SIZE];

    if (idx + blockDim.x + tid < N)
    {
        sm[tid] = g_idata[idx + tid] + g_idata[idx + blockDim.x + tid];
    }
    __syncthreads();

    // warp分散的问题
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2)
    {
        // tid = 32，64，128等时，活跃的线程数都是32的倍数，因此一个warp内不存在分化问题
        // tid = 1，2，4，8，16，32时，活跃的线程数都不是32的倍数，存在分化问题，一个warp中部分线程不活跃。
        if (tid < stride)
        {
            sm[tid] += sm[tid + stride];
        }
        __syncthreads();
    }
    if (tid == 0)
    {
        g_odata[bid] = sm[0];
    }
}

// 问题3：tid < 32时，仍然会存在单个warp内的分化问题
// 解决：使用warp shuffle指令，直接在寄存器中进行归约求和，避免共享内存的访存延迟
__device__ float warpReduceSum(float *sm, int tid)
{
    // tid: [0,31]
    float v = sm[tid] + sm[tid + 32];
    v += __shfl_down_sync(0xffffffff, v, 16);
    v += __shfl_down_sync(0xffffffff, v, 8);
    v += __shfl_down_sync(0xffffffff, v, 4);
    v += __shfl_down_sync(0xffffffff, v, 2);
    v += __shfl_down_sync(0xffffffff, v, 1);
    return v;
}
__global__ void reduce_v3(float *g_idata, float *g_odata)
{
    int bid = blockIdx.x;
    int tid = threadIdx.x;
    // 现在处理的单位是BlockSize*2，也就是行元素个数为BlockSize*2
    int idx = blockIdx.x * blockDim.x * 2;

    // 共享内存的大小保持不变
    __shared__ float sm[BLOCK_SIZE];

    if (idx + blockDim.x + tid < N)
    {
        sm[tid] = g_idata[idx + tid] + g_idata[idx + blockDim.x + tid];
    }
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 32; stride /= 2) // 至少为32
    {
        if (tid < stride)
        {
            sm[tid] += sm[tid + stride];
        }
        __syncthreads();
    }
    if (tid < 32)
    {
        float val = warpReduceSum(sm, tid);
        if (tid == 0)
        {
            g_odata[bid] = val;
        }
    }
}

__global__ void reduce_v4(float *g_idata, float *g_odata)
{
    int bid = blockIdx.x;
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x; // 处理BLOCK_SIZE个元素
    int laneid = tid % 32;             // warp内的线程id
    int warpid = tid / 32;             // warp的id
    int warpsize = 32;

    // 共享内存的大小保持不变
    __shared__ float sm[BLOCK_SIZE / 32];

    // 初始值
    float val = g_idata[idx + tid];

    // 逐个warp内归约求和
    for (int offset = warpsize / 2; offset > 0; offset /= 2)
    {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    if (laneid == 0)
    {
        sm[warpid] = val;
    }
    __syncthreads();

    // 使用单个warp进行归约求和
    if (warpid == 0)
    {
        val = (tid < (blockDim.x / warpsize)) ? sm[laneid] : 0.0f;
        for (int offset = warpsize / 2; offset > 0; offset /= 2)
        {
            val += __shfl_down_sync(0xffffffff, val, offset);
        }
        // warpid==0, laneid ==0
        if (laneid == 0)
        {
            g_odata[bid] = val;
        }
    }
}

// CPU验证函数
float reduce_cpu(const std::vector<float> &data)
{
    float sum = 0.0f;
    for (float val : data)
    {
        sum += val;
    }
    return sum;
}

int main()
{
    // v0,v4版本
    int num_blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // 注意：v1,v2,v3版本，由于每个block处理2*BLOCK_SIZE个元素，所以num_blocks需要除以2
    // int num_blocks = (N + 2 * BLOCK_SIZE - 1) / (2 * BLOCK_SIZE);

    std::vector<float> h_data(N);

    for (int i = 0; i < N; i++)
    {
        h_data[i] = 1.0f; // 简单起见，全部初始化为1.0
    }

    // -------------------------------
    // CPU 计时开始
    auto cpu_start = std::chrono::high_resolution_clock::now();

    float cpu_result = reduce_cpu(h_data);

    auto cpu_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> cpu_duration = cpu_end - cpu_start;
    // CPU 计时结束
    // -------------------------------

    std::cout << "CPU result: " << cpu_result << std::endl;
    std::cout << "CPU time: " << cpu_duration.count() << " ms" << std::endl;

    float *d_data, *d_result;
    float *d_final_result;
    float gpu_result;

    cudaMalloc(&d_data, N * sizeof(float));            // N
    cudaMalloc(&d_result, num_blocks * sizeof(float)); // num_blocks
    cudaMalloc(&d_final_result, 1 * sizeof(float));    // 1

    cudaMemcpy(d_data, h_data.data(), N * sizeof(float), cudaMemcpyHostToDevice);

    // -------------------------------
    // GPU 计时开始 (CUDA Events)
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    // reduce_v0<<<num_blocks, BLOCK_SIZE>>>(d_data, d_result); // 本质为1D数组，但是启动1024个block分组求和
    // reduce_v0<<<1, num_blocks>>>(d_result, d_final_result);

    // 注意：v1后续版本，需要同步修改num_blocks的定义
    // reduce_v1<<<num_blocks, BLOCK_SIZE>>>(d_data, d_result);
    // reduce_v1<<<1, num_blocks>>>(d_result, d_final_result);

    // reduce_v2<<<num_blocks, BLOCK_SIZE>>>(d_data, d_result);
    // reduce_v2<<<1, num_blocks>>>(d_result, d_final_result);

    // reduce_v3<<<num_blocks, BLOCK_SIZE>>>(d_data, d_result);
    // reduce_v3<<<1, num_blocks>>>(d_result, d_final_result);

    reduce_v4<<<num_blocks, BLOCK_SIZE>>>(d_data, d_result);
    reduce_v4<<<1, num_blocks>>>(d_result, d_final_result);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    // GPU 计时结束
    // -------------------------------

    std::cout << "GPU kernel time: " << milliseconds << " ms" << std::endl;

    cudaMemcpy(&gpu_result, d_final_result, sizeof(float),
               cudaMemcpyDeviceToHost);
    std::cout << "GPU result: " << gpu_result << std::endl;

    if (abs(cpu_result - gpu_result) < 1e-5)
    {
        std::cout << "Result verified successfully!" << std::endl;
    }
    else
    {
        std::cout << "Result verification failed!" << std::endl;
    }

    // 清理资源
    cudaFree(d_data);
    cudaFree(d_result);
    cudaFree(d_final_result);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
