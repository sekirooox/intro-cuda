#include <chrono>  // for timing
#include <cmath>   // for INFINITY
#include <cstdlib> // for malloc/free
#include <iostream>
using namespace std;
// Function to compare results
bool compare_results(const float *cpu, const float *gpu, int N, int C,
                     float epsilon = 1e-3f)
{
  for (int i = 0; i < N * C; ++i)
  {
    if (fabs(cpu[i] - gpu[i]) > epsilon)
    {
      std::cout << "Difference at index " << i << ": CPU=" << cpu[i]
                << ", GPU=" << gpu[i] << ", diff=" << fabs(cpu[i] - gpu[i])
                << std::endl;
      return false;
    }
  }
  return true;
}
// declare CPU/GPU versions of softmax
void softmax_forward_cpu(float *output, const float *input, int N, int C);
__global__ void softmax_forward_kernel1(float *output, const float *input, int N, int C);
__global__ void softmax_forward_kernel2(float *output, const float *input, int N, int C);
__global__ void softmax_forward_kernel3(float *output, const float *input, int N, int C);
__global__ void softmax_forward_kernel4(float *output, const float *input, int N, int C);
__device__ float WarpReduceMax(float val);
__device__ float WarpReduceSum(float val);

void softmax_forward_cpu(float *output, const float *input, int N, int C)
{
  /*
  input : shape (N, C)
  output: shape (N, C)
  注意：本质都是1D数组形式
  */
  for (int i = 0; i < N; i++)
  {
    // 求每一行的最大值
    float maxval = -INFINITY;
    for (int j = 0; j < C; j++)
    {
      int idx = i * C + j; // 2D->1D
      maxval = fmax(maxval, input[idx]);
    }

    // e^z-max(e^j)
    for (int j = 0; j < C; j++)
    {
      int idx = i * C + j; // 2D->1D
      output[idx] = expf(input[idx] - maxval);
    }

    // sum(e^j)
    float sumval = 0.0f;
    for (int j = 0; j < C; j++)
    {
      int idx = i * C + j; // 2D->1D
      sumval += output[idx];
    }
    // normalize
    for (int j = 0; j < C; j++)
    {
      int idx = i * C + j;
      output[idx] /= sumval;
    }
  }
}

// 优化1：每个block处理一行数据，N个blocks, 1个thread
__global__ void softmax_forward_kernel1(float *output, const float *input, int N, int C)
{
  // __global__: 表示GPU kernel函数, 但是不能在CPU上调用, 只能在GPU上调用
  int i = blockIdx.x * C; // 注意：只有一个thread, 因此不需要threadIdx.x

  // max
  float maxval = -INFINITY;
  for (int j = 0; j < C; j++)
  {
    int idx = i + j; // 2D->1D
    maxval = fmaxf(maxval, input[idx]);
  }

  // exp(x-max)
  for (int j = 0; j < C; j++)
  {
    int idx = i + j; // 2D->1D
    output[idx] = expf(input[idx] - maxval);
  }
  // sum(exp)
  float sumval = 0.0f;
  for (int j = 0; j < C; j++)
  {
    int idx = i + j; // 2D->1D
    sumval += output[idx];
  }

  // normalize
  // 小技巧：除法不如乘法快, 因此可以先计算倒数, 然后乘以倒数
  float norm_factor = 1.0 / sumval;
  for (int j = 0; j < C; j++)
  {
    int idx = i + j; // 2D->1D
    output[idx] *= norm_factor;
  }
}

// 优化2：每个block处理一行数据，但是存在多个thread进行归约

__global__ void softmax_forward_kernel2(float *output, const float *input, int N, int C)
{
  int bid = blockIdx.x;
  int tid = threadIdx.x;
  int block_size = blockDim.x;
  int idx = bid * C; // 2D->1D

  // 共享内存，总共需要 block_size 个线程的最大值
  extern __shared__ float shared[];

  // 对于最大值的归约过程
  float maxval = -INFINITY;
  for (int i = tid; i < C; i += block_size)
  {
    maxval = fmaxf(maxval, input[idx + i]);
  }
  shared[tid] = maxval;
  __syncthreads(); // 确保所有线程都有最大值

  for (int stride = block_size / 2; stride > 0; stride /= 2)
  {
    if (tid < stride)
    {
      shared[tid] = fmaxf(shared[tid], shared[tid + stride]);
    }
    // 等待所有线程完成同一步归约
    __syncthreads();
  }
  __syncthreads();

  // exp(x-max)：：遍历赋值
  float offset = shared[0]; // 技巧：设定为寄存器offset
  for (int i = tid; i < C; i += block_size)
  {
    output[idx + i] = expf(input[idx + i] - offset);
  }
  // 等待所有线程完成exp计算
  __syncthreads();

  // sum(exp)
  float sumval = 0.0f;
  for (int i = tid; i < C; i += block_size)
  {
    sumval += output[idx + i];
  }
  shared[tid] = sumval;
  __syncthreads(); // 确保所有线程都有最大值

  for (int stride = block_size / 2; stride > 0; stride /= 2)
  {
    if (tid < stride)
    {
      shared[tid] += shared[tid + stride];
    }
    // 等待所有线程完成同一步归约
    __syncthreads();
  }
  __syncthreads();

  // normalize：遍历赋值
  float normal_factor = 1.0 / shared[0];
  for (int i = tid; i < C; i += block_size)
  {
    output[idx + i] *= normal_factor;
  }
}

// 优化3：每个block处理一行数据，但是存在多个thread进行归约，使用warp shuffle指令进行归约
__global__ void softmax_forward_kernel3(float *output, const float *input, int N, int C)
{
  int bid = blockIdx.x;
  int tid = threadIdx.x;
  int block_size = blockDim.x; // 32
  int idx = bid * C;           // 2D->1D

  // max：先让每个线程负责一部分数据
  float maxval = -INFINITY;
  for (int i = tid; i < C; i += block_size)
  {
    maxval = fmaxf(maxval, input[idx + i]);
  }
  __syncthreads();

  // 使用warp shuffle指令进行归约
  maxval = WarpReduceMax(maxval);
  // 注意：每一个线程都有maxval，需要广播到所有线程
  maxval = __shfl_sync(0xffffffff, maxval, 0);

  // exp(x-max)：遍历赋值
  for (int i = tid; i < C; i += block_size)
  {
    output[idx + i] = expf(input[idx + i] - maxval);
  }
  __syncthreads();

  // sum(exp)
  float sumval = 0.0f;
  for (int i = tid; i < C; i += block_size)
  {
    sumval += output[idx + i];
  }
  __syncthreads();

  // 使用warp shuffle指令进行归约
  sumval = WarpReduceSum(sumval);
  // 注意：每一个线程都有sumval，需要广播到所有线程
  sumval = __shfl_sync(0xffffffff, sumval, 0);

  // normalize
  float normal_factor = 1.0 / sumval;
  for (int i = tid; i < C; i += block_size)
  {
    output[idx + i] *= normal_factor;
  }
  // 不需要__syncthreads()，因为每个线程都已经完成了自己的工作
}

__device__ float WarpReduceMax(float val)
{
  // 32个线程的warp，使用shuffle指令进行归约
  for (int offset = 16; offset > 0; offset /= 2)
  {
    // 这里不需要手动同步，所有有效线程都走到__shfl_down_sync函数处，才会开始执行shuffle操作
    float other_val = __shfl_down_sync(0xffffffff, val, offset); // tid, tid+offset
    val = fmaxf(val, other_val);
  }
  return val;
}

__device__ float WarpReduceSum(float val)
{
  // 32个线程的warp，使用shuffle指令进行归约
  for (int offset = 16; offset > 0; offset /= 2)
  {
    float other_val = __shfl_down_sync(0xffffffff, val, offset); // tid, tid+offset
    val += other_val;
  }
  return val;
}

// 优化4：考虑多个warp，结合共享内存进行同步
__global__ void softmax_forward_kernel4(float *output, const float *input, int N, int C)
{
  int bid = blockIdx.x;
  int tid = threadIdx.x;
  int block_size = blockDim.x; // 32
  int idx = bid * C;           // 2D->1D
  int wid = tid / 32;
  int laneid = tid % 32;
  int n_warp = block_size / 32;
  extern __shared__ float shared[];

  // max：先让每个线程负责一部分数据
  float maxval = -INFINITY;
  for (int i = tid; i < C; i += block_size)
  {
    maxval = fmaxf(maxval, input[idx + i]);
  }
  __syncthreads();

  // warp级别内的归约
  maxval = WarpReduceMax(maxval);

  // 每一个warp的局部结果存储到共享内存
  if (laneid == 0)
  {
    shared[wid] = maxval;
  }
  __syncthreads();

  // 暴力遍历得到全局结果
  if (tid == 0)
  {
    for (int i = 0; i < n_warp; i++)
    {
      maxval = fmaxf(maxval, shared[i]);
    }
    shared[0] = maxval;
  }
  __syncthreads();

  // 更新到共享内存
  float offset = shared[0];

  // exp(x-max)：遍历赋值
  for (int i = tid; i < C; i += block_size)
  {
    output[idx + i] = expf(input[idx + i] - offset);
  }
  __syncthreads();

  // sum(exp)
  float sumval = 0.0f;
  for (int i = tid; i < C; i += block_size)
  {
    sumval += output[idx + i];
  }
  __syncthreads();

  // warp级别内的归约
  sumval = WarpReduceSum(sumval);

  // 每一个warp的局部结果存储到共享内存
  if (laneid == 0)
  {
    shared[wid] = sumval;
  }
  __syncthreads();

  // 暴力遍历得到全局结果
  if (tid == 0)
  {
    float val = 0.0f; // 注意：这里不能使用sumval，因为sumval是每个线程的局部变量
    for (int i = 0; i < n_warp; i++)
    {
      val += shared[i];
    }
    shared[0] = val;
  }
  __syncthreads();

  sumval = shared[0]; // 全局最大值
  // normalize
  float normal_factor = 1.0f / sumval;
  for (int i = tid; i < C; i += block_size)
  {
    output[idx + i] *= normal_factor;
  }
}

int main()
{
  // Example: batch size N=32, classes C=4096
  int N = 32;
  int C = 4096;

  size_t num_elements = N * C;
  float *inp = (float *)malloc(num_elements * sizeof(float));
  float *out_cpu = (float *)malloc(num_elements * sizeof(float));
  float *out_gpu = (float *)malloc(num_elements * sizeof(float));

  // Initialize input with sample data
  for (int n = 0; n < N; ++n)
  {
    for (int c = 0; c < C; ++c)
    {
      inp[n * C + c] = float(c);
    }
  }

  // Run CPU version and measure time
  auto start_cpu = std::chrono::high_resolution_clock::now();
  softmax_forward_cpu(out_cpu, inp, N, C);
  auto end_cpu = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double, std::milli> cpu_time = end_cpu - start_cpu;

  // Run GPU version and measure time using CUDA events
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  float *d_out, *d_inp;
  cudaMalloc((void **)&d_out, N * C * sizeof(float));
  cudaMalloc((void **)&d_inp, N * C * sizeof(float));
  cudaMemcpy(d_inp, inp, N * C * sizeof(float), cudaMemcpyHostToDevice);

  cudaEventRecord(start);
  // Launch kernel

  // kernel 1 : N blocks, 1 block_size int blockSize = N;
  // int numBlocks = 32;
  // int blockSize = 1;
  // softmax_forward_kernel1<<<numBlocks, blockSize>>>(d_out, d_inp, N, C);
  // Results match: YES
  // CPU time: 9.41958 ms
  // GPU time: 0.568256 ms
  // Speedup: 16.5763x

  // kernel 2：N blocks, 128 block_size, 32 data per thread
  // int blockSize = 128; // C / blocksize = 4096 / 128 = 32
  // int numBlocks = N;
  // softmax_forward_kernel2<<<numBlocks, blockSize, sizeof(float) * blockSize>>>(d_out, d_inp, N, C);
  // Results match: YES
  // CPU time: 9.41705 ms
  // GPU time: 0.208768 ms
  // Speedup: 45.1077x

  // kernel 3-Reduction: N blocks, 32 block_size, 128 data per thread
  // int blockSize = 32;
  // int numBlocks = N;
  // softmax_forward_kernel3<<<numBlocks, blockSize>>>(d_out, d_inp, N, C);
  // Results match: YES
  // CPU time: 9.20329 ms
  // GPU time: 0.2872 ms
  // Speedup: 32.0449x

  // kernel 4-Reduction+Shared: N blocks, 128 block_size, 32 data per thread, shared memory
  int blockSize = 128;
  int numBlocks = N;
  softmax_forward_kernel4<<<numBlocks, blockSize, N * sizeof(float) * (blockSize / 32)>>>(d_out, d_inp, N, C);

  cudaEventRecord(stop);

  // Wait for the event to complete
  cudaEventSynchronize(stop);

  // Calculate milliseconds
  float gpu_time_ms = 0;
  cudaEventElapsedTime(&gpu_time_ms, start, stop);

  // Copy result back to host
  cudaMemcpy(out_gpu, d_out, N * C * sizeof(float), cudaMemcpyDeviceToHost);

  // Cleanup
  cudaFree(d_out);
  cudaFree(d_inp);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  // Compare results
  bool success = compare_results(out_cpu, out_gpu, N, C);
  std::cout << "Results match: " << (success ? "YES" : "NO") << std::endl;

  // Print performance comparison
  std::cout << "CPU time: " << cpu_time.count() << " ms" << std::endl;
  std::cout << "GPU time: " << gpu_time_ms << " ms" << std::endl;
  std::cout << "Speedup: " << (cpu_time.count() / (gpu_time_ms)) << "x"
            << std::endl;

  // Cleanup
  free(inp);
  free(out_cpu);
  free(out_gpu);

  return 0;
}
