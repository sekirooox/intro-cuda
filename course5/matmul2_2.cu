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
// 2D tile：每一个线程处理维度为(TM,TN)的元素
template <const int BM, const int BN, const int BK, const int TM, const int TN>
__global__ void mysgemm_v4(int M, int N, int K, float alpha, float *A, float *B,
                           float beta, float *C)
{
  int bx = blockIdx.x;
  int by = blockIdx.y;
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int NT = BM / TM * BN / TN;     // 每个线程块处理的元素总数
  int tid = ty * blockDim.x + tx; // 线程在块内的线性索引

  __shared__ float SA[BM][BK];
  __shared__ float SB[BK][BN];

  float Csub[TM][TN] = {0.0f}; // 当前线程负责的那块子矩阵

  for (int offset = 0; offset < K; offset += BK)
  {
    // 搬运共享内存
    #pragma unroll
    for (int i = tid; i < BM * BK; i += NT)
    {
      int sx = i % BK;
      int sy = i / BK;

      // A中的全局索引
      int gy = by * BM + sy;
      int gx = offset + sx;
      // SA[sy][sx] = A[gy][gx]
      if (gy < M && gx < K)
        SA[sy][sx] = A[gy * K + gx];
      else
        SA[sy][sx] = 0.0f;
    }

    #pragma unroll
    for (int i = tid; i < BK * BN; i += NT)
    {
      int sx = i % BN;
      int sy = i / BN;

      // B中的全局索引
      int gy = offset + sy;
      int gx = bx * BN + sx;
      if (gy < K && gx < N)
        SB[sy][sx] = B[gy * N + gx];
      else
        SB[sy][sx] = 0.0f;
    }

    __syncthreads();

    // 矩阵外积
    #pragma unroll
    for (int k = 0; k < BK; k++)
    {
      // 两个寄存器
      float rga[TM] = {0.0f};
      float rgb[TN] = {0.0f};
      // 易错点：寻找显存对应地址，ty * TM + i
      #pragma unroll
      for (int i = 0; i < TM; i++)
      {
        rga[i] = SA[ty * TM + i][k];
      }
      #pragma unroll
      for (int j = 0; j < TN; j++)
      {
        rgb[j] = SB[k][tx * TN + j];
      }
      #pragma unroll
      for (int i = 0; i < TM; i++)
      {
        #pragma unroll
        for (int j = 0; j < TN; j++)
        {
          Csub[i][j] += rga[i] * rgb[j];
        }
      }
    }
    __syncthreads();
  }
  // 写入
  #pragma unroll
  for (int i = 0; i < TM; i++)
  {
    #pragma unroll
    for (int j = 0; j < TN; j++)
    {
      // C矩阵的全局索引
      int gy = by * BM + ty * TM + i;
      int gx = bx * BN + tx * TN + j;
      if (gy < M && gx < N)
      {
        C[gy * N + gx] = alpha * Csub[i][j] + beta * C[gy * N + gx];
      }
      // else
      //   C[gy * N + gx] = 0.0f;
    }
  }
}

template <const int BM, const int BN, const int BK, const int TM, const int TN>
__global__ void mysgemm_v4_r(int M, int N, int K, float alpha, float *A, float *B,
                           float beta, float *C)
{
  int bx = blockIdx.x;
  int by = blockIdx.y;
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  constexpr int NT = BM / TM * BN / TN;     // 每个线程块处理的元素总数
  int tid = ty * blockDim.x + tx; // 线程在块内的线性索引

  __shared__ float SA[BM][BK];
  __shared__ float SB[BK][BN];

  B += bx * BN;
  C += by * BM * N + bx * BN;

  float Csub[TM][TN] = {0.0f}; // 当前线程负责的那块子矩阵

  for (int offset = 0; offset < K; offset += BK)
  {
    // 搬运共享内存
    #pragma unroll
    for (int i = tid; i < BM * BK; i += NT)
    {
      int sx = i % BK;
      int sy = i / BK;

      // A中的全局索引
      int gy = by * BM + sy;
      int gx = offset + sx;
      // SA[sy][sx] = A[gy][gx]
      if (gy < M && gx < K)
        SA[sy][sx] = A[gy * K + gx];
      else
        SA[sy][sx] = 0.0f;
    }

    #pragma unroll
    for (int i = tid; i < BK * BN; i += NT)
    {
      int sx = i % BN;
      int sy = i / BN;

      // B中的全局索引
      int gy = offset + sy;
      int gx = bx * BN + sx;
      if (gy < K && gx < N)
        SB[sy][sx] = B[gy * N + sx];
      else
        SB[sy][sx] = 0.0f;
    }

    __syncthreads();

    // 矩阵外积
    #pragma unroll
    for (int k = 0; k < BK; k++)
    {
      // 两个寄存器
      float rga[TM] = {0.0f};
      float rgb[TN] = {0.0f};
      // 易错点：寻找显存对应地址，ty * TM + i
      #pragma unroll
      for (int i = 0; i < TM; i++)
      {
        rga[i] = SA[ty * TM + i][k];
      }
      #pragma unroll
      for (int j = 0; j < TN; j++)
      {
        rgb[j] = SB[k][tx * TN + j];
      }
      #pragma unroll
      for (int i = 0; i < TM; i++)
      {
        #pragma unroll
        for (int j = 0; j < TN; j++)
        {
          Csub[i][j] += rga[i] * rgb[j];
        }
      }
    }
    __syncthreads();
  }
  // 写入
  #pragma unroll
  for (int i = 0; i < TM; i++)
  {
    #pragma unroll
    for (int j = 0; j < TN; j++)
    {
      // C矩阵的全局索引
      int gy = by * BM + ty * TM + i;
      int gx = bx * BN + tx * TN + j;
      if (gy < M && gx < N)
      {
        C[(ty * TM + i) * N + tx * TN + j] = alpha * Csub[i][j] + beta * C[(ty * TM + i) * N + tx * TN + j];
      }
      // else
      //   C[(ty * TM + i) * N + tx * TN + j] = 0.0f;
    }
  }
}

#define CEIL_DIV(M, N) ((M) + (N) - 1) / (N)

std::vector<int> generateSizes() { return {4096}; }
int main()
{
  int device_id = 0;
  checkCudaError(cudaSetDevice(device_id), "cudaSetDevice failed");
  std::vector<int> sizes = generateSizes();

  // 打开CSV文件
  std::ofstream csv_file("sgemm_benchmark_v3_2-2D-Tile_r.csv");
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

      // mysgemm_v4
      checkCudaError(cudaMemset(d_C_v1, 0, size), "cudaMemset d_C_v1 failed");

      // BM=BN=128，BK=4，TM=TN=8
      dim3 blockDim(16, 16);
      dim3 gridDim(CEIL_DIV(N, 128), CEIL_DIV(N, 128));

      for (int i = 0; i < warpup_time; ++i)
      {
        mysgemm_v4_r<128, 128, 8, 8, 8>
            <<<gridDim, blockDim>>>(N, N, N, alpha, d_A, d_B, beta, d_C_v1);
      }

      cudaDeviceSynchronize();
      checkCudaError(cudaMemset(d_C_v1, 0, size), "cudaMemset d_C_v1 failed");

      checkCudaError(cudaEventRecord(start),
                     "cudaEventRecord(start v1) failed");

      for (int i = 0; i < repeat_time; ++i)
      {
        // TM,TN
        mysgemm_v4_r<128, 128, 8, 8, 8>
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
