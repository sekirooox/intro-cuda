#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <cmath>   // for fabsf
#include <fstream> // for CSV output
#include <iostream>
#include <vector>

#define TOL 1e-5f

void checkCudaError(cudaError_t err, const char *msg)
{
  if (err != cudaSuccess)
  {
    std::cerr << msg << " CUDA ERROR: " << cudaGetErrorString(err) << std::endl;
    exit(EXIT_FAILURE);
  }
}

void checkCublasError(cublasStatus_t status, const char *msg)
{
  if (status != CUBLAS_STATUS_SUCCESS)
  {
    std::cerr << msg << " CUBLAS ERROR: " << status << std::endl;
    exit(EXIT_FAILURE);
  }
}

// 假设：一个线程负责一个元素，且一一对应
template <const int BLOCK_SIZE>
__global__ void mysgemm_v2(int M, int N, int K, float alpha, float *A, float *B,
                           float beta, float *C)
{
  // block为单位的2D坐标
  int bx = blockIdx.x;
  int by = blockIdx.y;
  // thread单位的block内2D坐标
  int tx = threadIdx.x;
  int ty = threadIdx.y;

  // 同一个block共享一个内存
  constexpr int BM = BLOCK_SIZE;
  constexpr int BN = BLOCK_SIZE;
  constexpr int BK = BLOCK_SIZE;
  __shared__ float SA[BM][BK];
  __shared__ float SB[BK][BN];

  float subval = 0.0f;

  // 全局2D坐标
  int gy = ty + by * BM;
  int gx = tx + bx * BN; // 偏移

  // 对于每一个分块，都进行局部的矩阵乘法
  // 坐标偏移量为offset，分别表示多个子矩阵
  for (int offset = 0; offset < K; offset += BK)
  {
    // 搬运到共享内存

    // SA[ty][tx] = A[gy][tx+offset]; // 2D坐标
    if (gy < M && tx + offset < K)
    {
      SA[ty][tx] = A[gy * K + tx + offset];
    }
    else
    {
      SA[ty][tx] = 0.0f;
    }
    // SB[ty][tx] = B[ty+offset][gx]; // 2D坐标
    if (ty + offset < K && gx < N)
    {
      SB[ty][tx] = B[(ty + offset) * N + gx];
    }
    else
    {
      SB[ty][tx] = 0.0f;
    }
    __syncthreads();

    // 计算局部矩阵乘法
    for (int k = 0; k < BK; k++)
    {
      subval += SA[ty][k] * SB[k][tx];
    }
    __syncthreads();

    // 累加多个局部值
  }
  // 写入全局
  if (gy < M && gx < N)
  {
    C[gy * N + gx] = alpha * subval + beta * C[gy * N + gx];
  }
}

template <const int BLOCK_SIZE>
__global__ void mysgemm_v2_r(int M, int N, int K, float alpha, float *A, float *B,
                           float beta, float *C)
{
  // block为单位的2D坐标
  int bx = blockIdx.x;
  int by = blockIdx.y;
  // thread单位的block内2D坐标
  int tx = threadIdx.x;
  int ty = threadIdx.y;

  // 同一个block共享一个内存
  constexpr int BM = BLOCK_SIZE;
  constexpr int BN = BLOCK_SIZE;
  constexpr int BK = BLOCK_SIZE;
  __shared__ float SA[BM][BK];
  __shared__ float SB[BK][BN];

  float subval = 0.0f;

  // 全局2D坐标
  int gy = ty + by * BM;
  int gx = tx + bx * BN; // 偏移

  // 对于每一个分块，都进行局部的矩阵乘法
  // 坐标偏移量为offset，分别表示多个子矩阵
  for (int offset = 0; offset < K; offset += BK)
  {
    // 搬运到共享内存

    // SA[ty][tx] = A[gy][tx+offset]; // 2D坐标
    if (gy < M && tx + offset < K)
    {
      SA[ty][tx] = A[gy * K + tx + offset];
    }
    else
    {
      SA[ty][tx] = 0.0f;
    }
    // SB[ty][tx] = B[ty+offset][gx]; // 2D坐标
    if (ty + offset < K && gx < N)
    {
      SB[ty][tx] = B[(ty + offset) * N + gx];
    }
    else
    {
      SB[ty][tx] = 0.0f;
    }
    __syncthreads();

    // 计算局部矩阵乘法
    #pragma unroll
    for (int k = 0; k < BK; k++)
    {
      subval += SA[ty][k] * SB[k][tx];
    }
    __syncthreads();

    // 累加多个局部值
  }
  // 写入全局
  if (gy < M && gx < N)
  {
    C[gy * N + gx] = alpha * subval + beta * C[gy * N + gx];
  }
}
#define CEIL_DIV(M, N) ((M) + (N) - 1) / (N)
int main()
{
  std::vector<int> sizes = {1024};

  // 打开CSV文件
  std::ofstream csv_file("sgemm_benchmark_v2-Tile_r.csv");
  csv_file << "Size,CUBLAS_GFLOPS,MySGEMM_FLOPS,Matched" << std::endl;

  for (int N : sizes)
  {
    std::cout << "Testing size: " << N << std::endl;

    size_t size = N * N * sizeof(float);
    float *A = (float *)malloc(size);
    float *B = (float *)malloc(size);
    float *C_cublas = (float *)malloc(size);
    float *C_v1 = (float *)malloc(size);

    float *d_A, *d_B, *d_C_v1;
    checkCudaError(cudaMalloc(&d_A, size), "cudaMalloc d_A failed");
    checkCudaError(cudaMalloc(&d_B, size), "cudaMalloc d_B failed");
    checkCudaError(cudaMalloc(&d_C_v1, size), "cudaMalloc d_C_v1 failed");

    bool out_of_memory = false;

    try
    {
      // 初始化矩阵 A 和 B
      for (int i = 0; i < N * N; ++i)
      {
        A[i] = 1.0f;
        B[i] = 2.0f;
      }

      // 拷贝到设备
      checkCudaError(cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice),
                     "cudaMemcpy A to device failed");
      checkCudaError(cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice),
                     "cudaMemcpy B to device failed");

      cublasHandle_t handle;
      checkCublasError(cublasCreate(&handle), "cublasCreate failed");

      float alpha = 1.0f;
      float beta = 0.0f;

      cudaEvent_t start, stop;
      checkCudaError(cudaEventCreate(&start), "cudaEventCreate(start) failed");
      checkCudaError(cudaEventCreate(&stop), "cudaEventCreate(stop) failed");

      // warmup
      int warpup_time = 10; // 热身次数
      for (int i = 0; i < warpup_time; ++i)
      {
        checkCublasError(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N,
                                     &alpha, d_B, N, d_A, N, &beta, d_C_v1, N),
                         "cublasSgemm failed");
      }
      cudaDeviceSynchronize();

      // cuBLAS SGEMM
      int repeat_time = 5;
      checkCudaError(cudaEventRecord(start),
                     "cudaEventRecord(start cublas) failed");
      for (int i = 0; i < repeat_time; ++i)
      {
        checkCublasError(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N,
                                     &alpha, d_B, N, d_A, N, &beta, d_C_v1, N),
                         "cublasSgemm failed");
      }

      checkCudaError(cudaEventRecord(stop),
                     "cudaEventRecord(stop cublas) failed");
      checkCudaError(cudaEventSynchronize(stop),
                     "cudaEventSynchronize cublas failed");

      float cublas_time = 0;
      checkCudaError(cudaEventElapsedTime(&cublas_time, start, stop),
                     "cudaEventElapsedTime cublas failed");

      // 拷贝 cuBLAS 结果
      checkCudaError(cudaMemcpy(C_cublas, d_C_v1, size, cudaMemcpyDeviceToHost),
                     "cudaMemcpy C_cublas failed");

      // mysgemm_v1
      checkCudaError(cudaMemset(d_C_v1, 0, size), "cudaMemset d_C_v1 failed");

      // dim3 blockDim(1024);
      dim3 blockDim(32, 32);
      dim3 gridDim(CEIL_DIV(N, 32), CEIL_DIV(N, 32));

      for (int i = 0; i < warpup_time; ++i)
      {
        mysgemm_v2_r<32>
            <<<gridDim, blockDim>>>(N, N, N, alpha, d_A, d_B, beta, d_C_v1);
      }

      cudaDeviceSynchronize();
      checkCudaError(cudaMemset(d_C_v1, 0, size), "cudaMemset d_C_v1 failed");

      checkCudaError(cudaEventRecord(start),
                     "cudaEventRecord(start v1) failed");

      for (int i = 0; i < repeat_time; ++i)
      {
        mysgemm_v2_r<32>
            <<<gridDim, blockDim>>>(N, N, N, alpha, d_A, d_B, beta, d_C_v1);
      }
      checkCudaError(cudaEventRecord(stop), "cudaEventRecord(stop v1) failed");
      checkCudaError(cudaEventSynchronize(stop),
                     "cudaEventSynchronize v1 failed");
      float v1_time = 0;
      checkCudaError(cudaEventElapsedTime(&v1_time, start, stop),
                     "cudaEventElapsedTime v1 failed");

      // 拷贝手写 kernel 结果
      checkCudaError(cudaMemcpy(C_v1, d_C_v1, size, cudaMemcpyDeviceToHost),
                     "cudaMemcpy C_v1 failed");
      // 结果比较
      int error_count = 0;
      for (int i = 0; i < N * N && error_count < 10; ++i)
      {
        if (fabsf(C_cublas[i] - C_v1[i]) > TOL)
        {
          error_count++;
        }
      }

      float cublas_gflops =
          repeat_time * 2.0f * N * N * N / (cublas_time * 1e6f); // GFlops
      float v1_gflops =
          repeat_time * 2.0f * N * N * N / (v1_time * 1e6f); // GFlops
      // 写入CSV
      csv_file << N << "," << cublas_gflops << "," << v1_gflops << ","
               << (error_count == 0 ? "1" : "0") << std::endl;

      // 释放资源
      cublasDestroy(handle);
      cudaEventDestroy(start);
      cudaEventDestroy(stop);
      cudaFree(d_A);
      cudaFree(d_B);
      cudaFree(d_C_v1);

      free(A);
      free(B);
      free(C_cublas);
      free(C_v1);
    }
    catch (...)
    {
      std::cerr << "Out of memory or error during testing size: " << N
                << std::endl;
      out_of_memory = true;
    }

    if (!out_of_memory)
    {
      std::cout << "Finished size: " << N << std::endl;
    }
    else
    {
      csv_file << N << ",OOM,OOM,0" << std::endl;
    }
  }

  csv_file.close();

  std::cout << "Benchmark completed. Results saved to 'sgemm_benchmark.csv'"
            << std::endl;
  return 0;
}
