#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <cmath>   // for fabsf
#include <fstream> // for CSV output
#include <iostream>
#include <vector>

#define TOL 1e-5f
#define OFFSET(row, col, ld) ((row) * (ld) + (col))

// 相当于转换为float4，然后获得第0个元素
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
__global__ void mysgemm_v3_2(int M, int N, int K, float alpha, float *A, float *B,
                             float beta, float *C)
{

  static_assert(BM % TM == 0, "BM must be divisible by TM");
  static_assert(BN % TN == 0, "BN must be divisible by TN");
  static_assert(BK % 4 == 0, "BK must be divisible by 4");
  static_assert(BM % 4 == 0, "BM must be divisible by 4");
  static_assert(BN % 4 == 0, "BN must be divisible by 4");
  static_assert(TM % 4 == 0, "TM must be divisible by 4");
  static_assert(TN % 4 == 0, "TN must be divisible by 4");

  int ty = threadIdx.y;
  int tx = threadIdx.x;
  int by = blockIdx.y;
  int bx = blockIdx.x;
  int tid = ty * blockDim.x + tx;
  int NT = (BM / TM) * (BN / TN);

  __shared__ __align__(16) float SA[BK][BM];
  __shared__ __align__(16) float SB[BK][BN];

  float sum[TM][TN] = {{0.0f}};

  for (int offset = 0; offset < K; offset += BK)
  {
    // load tiles SA
    for (int i = tid; i < BK * BM / 4; i += NT)
    {
      // 先暂时不考虑转置
      int sy = i / (BK / 4);
      int sx = i % (BK / 4) * 4;
      int gy = by * BM + sy;
      int gx = sx + offset;

      float4 a_val = FETCH_FLOAT4(A[OFFSET(gy, gx, K)]);
      // 转置
      SA[sx + 0][sy] = a_val.x;
      SA[sx + 1][sy] = a_val.y;
      SA[sx + 2][sy] = a_val.z;
      SA[sx + 3][sy] = a_val.w;
    }

    // load tiles SB
    for (int i = tid; i < BK * BN / 4; i += NT)
    {
      int sy = i / (BN / 4);
      int sx = i % (BN / 4) * 4;
      int gy = sy + offset;
      int gx = bx * BN + sx;

      // 向量化赋值
      FETCH_FLOAT4(SB[sy][sx]) = FETCH_FLOAT4(B[OFFSET(gy, gx, N)]);
    }

    __syncthreads();

    for (int k = 0; k < BK; k++)
    {
      float a_val[TM];
      float b_val[TN];
      for (int m = 0; m < TM; m += 4)
      {
        // SA[ty*TM+m][k]，转置的结果
        FETCH_FLOAT4(a_val[m]) = FETCH_FLOAT4(SA[k][ty * TM + m]);
      }
      for (int n = 0; n < TN; n += 4)
      {
        FETCH_FLOAT4(b_val[n]) = FETCH_FLOAT4(SB[k][tx * TN + n]);
      }

      // 外积
      for (int m = 0; m < TM; m++)
      {
        for (int n = 0; n < TN; n++)
        {
          sum[m][n] += a_val[m] * b_val[n];
        }
      }
      __syncthreads();
    }
  }
  // 易错点：行不能+=4，也要逐行向量化写回
  for (int m = 0; m < TM; m++)
  {
    for (int n = 0; n < TN; n += 4)
    {
      int gy = by * BM + ty * TM + m;
      int gx = bx * BN + tx * TN + n;
      if (gy < M && gx < N)
      {
        // 不考虑alpha和beta的情况，直接写回C矩阵
        if (gy < M && gx < N)
          FETCH_FLOAT4(C[OFFSET(gy, gx, N)]) = FETCH_FLOAT4(sum[m][n]);
      }
    }
  }
}

// 寄存器优化版本-参考版本：与v3_1寄存器用量相当
template <const int BM, const int BN, const int BK, const int TM, const int TN>
__global__ void mysgemm_v3_2_r(int M, int N, int K, float alpha, float *A, float *B,
                               float beta, float *C)
{
  int ty = threadIdx.y;
  int tx = threadIdx.x;
  int by = blockIdx.y;
  int bx = blockIdx.x;
  int tid = ty * blockDim.x + tx;
  constexpr int NT = (BM / TM) * (BN / TN);

  static_assert(BM % TM == 0, "BM must be divisible by TM");
  static_assert(BN % TN == 0, "BN must be divisible by TN");
  static_assert(BK % 4 == 0, "BK must be divisible by 4");
  static_assert(BM % 4 == 0, "BM must be divisible by 4");
  static_assert(BN % 4 == 0, "BN must be divisible by 4");
  static_assert(TM % 4 == 0, "TM must be divisible by 4");
  static_assert(TN % 4 == 0, "TN must be divisible by 4");

  __shared__ __align__(16) float SA[BK][BM]; // 转置SA获得更好的局部性
  __shared__ __align__(16) float SB[BK][BN];

  float sum[TM][TN] = {{0.0f}};

  for (int offset = 0; offset < K; offset += BK)
  {
#pragma unroll

    // 注意：SA转置过
    for (int i = 0; i < BM * BK / 4; i += NT)
    {
      if ((BM * BK / 4) % NT != 0 && tid + i >= BM * BK / 4)
        continue;
      // SA (转置前)
      int sy = (tid + i) / (BK / 4);
      int sx = (tid + i) % (BK / 4) * 4;

      // A
      int gy = by * BM + sy;
      int gx = offset + sx;
      if (gy < M && gx < K)
      {
        // 我们需要逐列访问A元素，现在的存储方式是行优先的
        // 局部性比较差
        // FETCH_FLOAT4(SA[sy][sx]) = FETCH_FLOAT4(A[OFFSET(gy, gx, K)]);

        // 现在我们将A矩阵转置存储，变为列优先存储
        // 这样我们就可以逐列访问A元素，局部性更好
        float4 a_val = FETCH_FLOAT4(A[OFFSET(gy, gx, K)]);

        // 转置(i,j) -> (j,i)
        SA[sx + 0][sy] = a_val.x;
        SA[sx + 1][sy] = a_val.y;
        SA[sx + 2][sy] = a_val.z;
        SA[sx + 3][sy] = a_val.w;
      }
      else
      {
        SA[sx + 0][sy] = 0.0f;
        SA[sx + 1][sy] = 0.0f;
        SA[sx + 2][sy] = 0.0f;
        SA[sx + 3][sy] = 0.0f;
      }
    }

#pragma unroll
    for (int i = 0; i < BN * BK / 4; i += NT)
    {
      if ((BN * BK / 4) % NT != 0 && tid + i >= BN * BK / 4)
        continue;
      // SB
      int sy = (tid + i) / (BN / 4);
      int sx = (tid + i) % (BN / 4) * 4;

      // B
      int gy = offset + sy;
      int gx = bx * BN + sx;
      if (gy < K && gx < N)
      {
        FETCH_FLOAT4(SB[sy][sx]) = FETCH_FLOAT4(B[OFFSET(gy, gx, N)]);
      }
      else
      {
        FETCH_FLOAT4(SB[sy][sx]) = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
      }
    }

    __syncthreads();

    // compute
#pragma unroll

    for (int k = 0; k < BK; k++)
    {
      float a_val[TM];
      float b_val[TN];
      // 分别向量化读取SA和SB
#pragma unroll
      for (int m = 0; m < TM; m += 4)
      {
        FETCH_FLOAT4(a_val[m]) = FETCH_FLOAT4(SA[k][ty * TM + m]);
      }
#pragma unroll
      for (int n = 0; n < TN; n += 4)
      {
        FETCH_FLOAT4(b_val[n]) = FETCH_FLOAT4(SB[k][tx * TN + n]);
      }
#pragma unroll
      for (int m = 0; m < TM; m++)
      {
        for (int n = 0; n < TN; n++)
        {
          sum[m][n] += a_val[m] * b_val[n];
        }
      }
    }

    __syncthreads();
  }
  // write back
#pragma unroll
  for (int m = 0; m < TM; m++)
  {
#pragma unroll
    for (int n = 0; n < TN; n += 4) // 也是写量化读取
    {
      int gy = by * BM + ty * TM + m;
      int gx = bx * BN + tx * TN + n;
      if (gy < M && gx < N)
      {
        // 不考虑alpha和beta的情况，直接写回C矩阵
        if (gy < M && gx < N)
          FETCH_FLOAT4(C[OFFSET(gy, gx, N)]) = FETCH_FLOAT4(sum[m][n]);
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
  std::ofstream csv_file("sgemm_benchmark_v4-2D-vector.csv");
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
        mysgemm_v3_2_r<128, 128, 8, 8, 8>
            <<<gridDim, blockDim>>>(N, N, N, alpha, d_A, d_B, beta, d_C_v1);
      }

      cudaDeviceSynchronize();
      checkCudaError(cudaMemset(d_C_v1, 0, size), "cudaMemset d_C_v1 failed");

      checkCudaError(cudaEventRecord(start),
                     "cudaEventRecord(start v1) failed");

      for (int i = 0; i < repeat_time; ++i)
      {
        mysgemm_v3_2_r<128, 128, 8, 8, 8>
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
