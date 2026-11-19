#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <cmath>   // for fabsf
#include <fstream> // for CSV output
#include <iostream>
#include <vector>

#define TOL 1e-5f
#define OFFSET(row, col, ld) ((row) * (ld) + (col))
#define FETCH_FLOAT4(pointer) (reinterpret_cast<float4 *>(&(pointer))[0])
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
// 2D-Tile + 向量化加载
template <const int BM, const int BN, const int BK, const int TM, const int TN>
__global__ void mysgemm_v3_1(int M, int N, int K, float alpha, float *A, float *B,
                             float beta, float *C)
{
  int ty = threadIdx.y;
  int tx = threadIdx.x;
  int by = blockIdx.y;
  int bx = blockIdx.x;
  int tid = ty * blockDim.x + tx;
  int NT = BM / TM * BN / TN;

  static_assert(BK % 4 == 0, "BK must be divisible by 4");
  static_assert(BN % 4 == 0, "BN must be divisible by 4");
  static_assert(TN % 4 == 0, "TN must be divisible by 4");

  __shared__ float SA[BM][BK];
  __shared__ float SB[BK][BN];
  float sum[TM][TN] = {{0.0f}};

  for (int offset = 0; offset < K; offset += BK)
  {
// 向量化搬运数据
#pragma unroll
    for (int i = tid; i < BM * BK / 4; i += NT)
    {
      // SA中的坐标
      int sy = i / (BK / 4);
      int sx = i % (BK / 4) * 4; // 注意：起始偏移

      // A中的坐标
      int gy = by * BM + sy;
      int gx = offset + sx;

      // SA[sy][sx] = A[gy * K + gx];
      // 地址对齐的方式加载4个float
      if (gy < M && gx < K)
      {
        *(float4 *)&SA[sy][sx] = *(float4 *)&A[gy * K + gx];
      }
      else
        *(float4 *)&SA[sy][sx] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    }

#pragma unroll
    for (int i = tid; i < BK * BN / 4; i += NT)
    {
      // SB
      int sy = i / (BN / 4);
      int sx = i % (BN / 4) * 4;

      // B
      int gy = sy + offset;
      int gx = bx * BN + sx;

      if (gy < K && gx < N)
      {
        *(float4 *)&SB[sy][sx] = *(float4 *)&B[gy * N + gx];
      }
      else
      {
        *(float4 *)&SB[sy][sx] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
      }
    }

    __syncthreads();

// 矩阵乘法
#pragma unroll
    for (int k = 0; k < BK; k++)
    {
#pragma unroll
      for (int n = 0; n < TN; n += 4)
      {
        float4 b_val = *(float4 *)&SB[k][tx * TN + n];
#pragma unroll
        for (int m = 0; m < TM; m++)
        {
          float a_val = SA[ty * TM + m][k];
          sum[m][n + 0] += a_val * b_val.x;
          sum[m][n + 1] += a_val * b_val.y;
          sum[m][n + 2] += a_val * b_val.z;
          sum[m][n + 3] += a_val * b_val.w;
        }
      }
    }
    __syncthreads();
  }

// 写回结果
#pragma unroll
  for (int m = 0; m < TM; m++)
  {
    int gy = by * BM + ty * TM + m;
#pragma unroll
    for (int n = 0; n < TN; n += 4)
    { // 向量化写入
      int gx = bx * BN + tx * TN + n;
      if (gy < M && gx < N)
      {
        *(float4 *)&C[gy * N + gx] = *(float4 *)&sum[m][n];
      }
    }
  }
}

template <const int BM, const int BN, const int BK, const int TM, const int TN>
__global__ void mysgemm_v3_1_r(int M, int N, int K, float alpha, float *A, float *B,
                             float beta, float *C)
{
  int ty = threadIdx.y;
  int tx = threadIdx.x;
  int by = blockIdx.y;
  int bx = blockIdx.x;
  int tid = ty * blockDim.x + tx;
  constexpr int NT = BM / TM * BN / TN;

  static_assert(BK % 4 == 0, "BK must be divisible by 4");
  static_assert(BN % 4 == 0, "BN must be divisible by 4");
  static_assert(TN % 4 == 0, "TN must be divisible by 4");

  __shared__ float SA[BM][BK];
  __shared__ float SB[BK][BN];
  float sum[TM][TN] = {{0.0f}};

  for (int offset = 0; offset < K; offset += BK)
  {
// 向量化搬运数据
#pragma unroll
    for (int i = 0; i < BM * BK / 4; i += NT)
    {
      if ((BM * BK / 4) % NT != 0 && tid + i >= BM * BK / 4)
        continue;
      // SA中的坐标
      int sy = (tid + i) / (BK / 4);
      int sx = (tid + i) % (BK / 4) * 4; // 注意：起始偏移

      // A中的坐标
      int gy = by * BM + sy;
      int gx = offset + sx;

      // SA[sy][sx] = A[gy * K + gx];
      // 地址对齐的方式加载4个float
      if (gy < M && gx < K)
      {
        *(float4 *)&SA[sy][sx] = *(float4 *)&A[gy * K + gx];
      }
      else
        *(float4 *)&SA[sy][sx] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    }

#pragma unroll
    for (int i = 0; i < BK * BN / 4; i += NT)
    {
      if ((BK * BN / 4) % NT != 0 && tid + i >= BK * BN / 4)
        continue;
      // SB
      int sy = (tid + i) / (BN / 4);
      int sx = (tid + i) % (BN / 4) * 4;

      // B
      int gy = sy + offset;
      int gx = bx * BN + sx;

      if (gy < K && gx < N)
      {
        *(float4 *)&SB[sy][sx] = *(float4 *)&B[gy * N + gx];
      }
      else
      {
        *(float4 *)&SB[sy][sx] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
      }
    }

    __syncthreads();

// 矩阵乘法
#pragma unroll
    for (int k = 0; k < BK; k++)
    {
#pragma unroll
      for (int n = 0; n < TN; n += 4)
      {
        float4 b_val = *(float4 *)&SB[k][tx * TN + n];
#pragma unroll
        for (int m = 0; m < TM; m++)
        {
          float a_val = SA[ty * TM + m][k];
          sum[m][n + 0] += a_val * b_val.x;
          sum[m][n + 1] += a_val * b_val.y;
          sum[m][n + 2] += a_val * b_val.z;
          sum[m][n + 3] += a_val * b_val.w;
        }
      }
    }
    __syncthreads();
  }

// 写回结果
#pragma unroll
  for (int m = 0; m < TM; m++)
  {
    int gy = by * BM + ty * TM + m;
#pragma unroll
    for (int n = 0; n < TN; n += 4)
    { // 向量化写入
      int gx = bx * BN + tx * TN + n;
      if (gy < M && gx < N)
      {
        *(float4 *)&C[gy * N + gx] = *(float4 *)&sum[m][n];
      }
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
  std::ofstream csv_file("sgemm_benchmark_v4-1D-vector_r.csv");
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

      dim3 blockDim(16, 16);
      dim3 gridDim(CEIL_DIV(N, 128), CEIL_DIV(N, 128));

      for (int i = 0; i < warpup_time; ++i)
      {
        mysgemm_v3_1_r<128, 128, 8, 8, 8>
            <<<gridDim, blockDim>>>(N, N, N, alpha, d_A, d_B, beta, d_C_v1);
      }

      cudaDeviceSynchronize();
      checkCudaError(cudaMemset(d_C_v1, 0, size), "cudaMemset d_C_v1 failed");

      checkCudaError(cudaEventRecord(start),
                     "cudaEventRecord(start v1) failed");

      for (int i = 0; i < repeat_time; ++i)
      {
        mysgemm_v3_1_r<128, 128, 8, 8, 8>
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
