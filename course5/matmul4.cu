#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <cmath>   // for fabsf
#include <fstream> // for CSV output
#include <iostream>
#include <vector>

#define BLOCK_SIZE 128
#define TOL 1e-5f
#define CEIL_DIV(M, N) ((M) + (N) - 1) / (N)

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

#define OFFSET(row, col, ld) ((row) * (ld) + (col))
#define FETCH_FLOAT4(pointer) (reinterpret_cast<float4 *>(&(pointer))[0])

// 嵌套双层流水线：
// 1. 使用流水线技术，先预读取共享显存。
// 2. 在当前显存下，使用流水线技术，预先读取线程负责的某一行或者列。
// 3. 确保最后一个buffer的计算。
// 4. 确保最后一个显存的计算。
template <const int BM, const int BN, const int BK, const int TM, const int TN>
__global__ void __launch_bounds__(256)
    mysgemm_v4(int M, int N, int K, float alpha, float *A, float *B, float beta,
               float *C)
{
  int tx = threadIdx.x % (BN / TN);
  int ty = threadIdx.x / (BN / TN);
  int bx = blockIdx.x;
  int by = blockIdx.y;
  int tid = threadIdx.x;
  constexpr int NT = (BM / TM) * (BN / TN);
  constexpr int A_NUM_PER_THREAD = CEIL_DIV(BM * BK / 4, NT);
  constexpr int B_NUM_PER_THREAD = CEIL_DIV(BK * BN / 4, NT);

  float4 a_next[A_NUM_PER_THREAD]; // 寄存器预读取A加载的数据，数量为A_NUM_PER_THREAD*4
  float4 b_next[B_NUM_PER_THREAD]; // 寄存器预读取B加载的数据，数量为B_NUM_PER_THREAD*4

  // 共享内存双缓存
  __shared__ __align__(16) float SA[2][BK][BM];
  __shared__ __align__(16) float SB[2][BK][BN];

  // 寄存器双缓存
  float a_val[2][TM];
  float b_val[2][TN];

  float sum[TM][TN] = {{0.0f}};

  int cur_idx = 0;

  // 1.提前加载SA[0]和SB[0]
#pragma unroll
  for (int i = tid; i < BK * BM / 4; i += NT)
  {
    // 先暂时不考虑转置
    int sy = i / (BK / 4);
    int sx = i % (BK / 4) * 4;
    int gy = by * BM + sy;
    int gx = sx + 0;

    float4 a_val = FETCH_FLOAT4(A[OFFSET(gy, gx, K)]);
    // 转置
    SA[cur_idx][sx + 0][sy] = a_val.x;
    SA[cur_idx][sx + 1][sy] = a_val.y;
    SA[cur_idx][sx + 2][sy] = a_val.z;
    SA[cur_idx][sx + 3][sy] = a_val.w;
  }

#pragma unroll
  for (int i = tid; i < BK * BN / 4; i += NT)
  {
    int sy = i / (BN / 4);
    int sx = i % (BN / 4) * 4;
    int gy = sy + 0;
    int gx = bx * BN + sx;

    // 向量化赋值
    FETCH_FLOAT4(SB[cur_idx][sy][sx]) = FETCH_FLOAT4(B[OFFSET(gy, gx, N)]);
  }

  // 易错点：加载内存需要同步
  __syncthreads();

  // 第0列和第0行元素
#pragma unroll
  for (int m = 0; m < TM; m += 4)
  {
    // SA[ty*TM+m][k]，转置的结果
    FETCH_FLOAT4(a_val[cur_idx][m]) = FETCH_FLOAT4(SA[cur_idx][0][ty * TM + m]); // 第0个共享显存的第0行
  }
#pragma unroll
  for (int n = 0; n < TN; n += 4)
  {
    FETCH_FLOAT4(b_val[cur_idx][n]) = FETCH_FLOAT4(SB[cur_idx][0][tx * TN + n]);
  }

  // cur：当前读取的共享显存索引
  for (int offset = 0; offset < K; offset += BK)
  {
    int nxt_idx = cur_idx ^ 1;
    int nxt_offset = offset + BK; // 第k+1个offset

    // 2.第一级流水线：加载SA[k+1]和SB[k+1]，第k+1列和第k+1行元素
    // 注意：不直接搬运到显存中，先分为两步：1.搬运到寄存器，2.计算后再搬运到显存中
    if (nxt_offset < K)
    {
#pragma unroll
      for (int i = tid; i < BK * BM / 4; i += NT)
      {
        // 先暂时不考虑转置
        int sy = i / (BK / 4);
        int sx = i % (BK / 4) * 4;
        int gy = by * BM + sy;
        int gx = sx + nxt_offset;

        // 先放在寄存器中
        a_next[i / NT] = FETCH_FLOAT4(A[OFFSET(gy, gx, K)]);
      }

#pragma unroll
      for (int i = tid; i < BK * BN / 4; i += NT)
      {
        int sy = i / (BN / 4);
        int sx = i % (BN / 4) * 4;
        int gy = sy + nxt_offset;
        int gx = bx * BN + sx;

        // 先放在寄存器中
        b_next[i / NT] = FETCH_FLOAT4(B[OFFSET(gy, gx, N)]);
      }
    }

    // 3.第二级流水线：计算第k行/列元素，写入k+1行/列元素
    // 默认k%2==0，放在第0个寄存器槽位，k+1%2==1，放在第1个寄存器槽位
    // 且默认起始计算的寄存器槽位为0！
#pragma unroll
    for (int k = 0; k < BK - 1; k++)
    {
      // 写入第k+1行/列元素
      for (int m = 0; m < TM; m += 4)
      {
        // 始终使用当前idx的共享显存
        FETCH_FLOAT4(a_val[(k + 1) % 2][m]) = FETCH_FLOAT4(SA[cur_idx][k + 1][ty * TM + m]);
      }
#pragma unroll
      for (int n = 0; n < TN; n += 4)
      {
        FETCH_FLOAT4(b_val[(k + 1) % 2][n]) = FETCH_FLOAT4(SB[cur_idx][k + 1][tx * TN + n]);
      }

      // 计算第k行/列元素
#pragma unroll
      for (int m = 0; m < TM; m++)
      {
        for (int n = 0; n < TN; n++)
        {
          sum[m][n] += a_val[k % 2][m] * b_val[k % 2][n];
        }
      }
    }
    if (nxt_offset < K)
    {
      // 4.1：从寄存器写入到内存中
#pragma unroll
      for (int i = tid; i < BK * BM / 4; i += NT)
      {
        // 先暂时不考虑转置
        int sy = i / (BK / 4);
        int sx = i % (BK / 4) * 4;
        // int gy = by * BM + sy;
        // int gx = sx + nxt_offset;

        float4 tmp = a_next[i / NT];
        SA[nxt_idx][sx + 0][sy] = tmp.x;
        SA[nxt_idx][sx + 1][sy] = tmp.y;
        SA[nxt_idx][sx + 2][sy] = tmp.z;
        SA[nxt_idx][sx + 3][sy] = tmp.w;
      }

#pragma unroll
      for (int i = tid; i < BK * BN / 4; i += NT)
      {
        int sy = i / (BN / 4);
        int sx = i % (BN / 4) * 4;
        // int gy = sy + nxt_offset;
        // int gx = bx * BN + sx;

        FETCH_FLOAT4(SB[nxt_idx][sy][sx]) = b_next[i / NT];
      }

      __syncthreads();
      // 4.2：读取下一个共享块的第0行/列的元素，始终寄存器槽位0
#pragma unroll
      for (int m = 0; m < TM; m += 4)
      {
        FETCH_FLOAT4(a_val[0][m]) = FETCH_FLOAT4(SA[nxt_idx][0][ty * TM + m]); // 第0个共享显存的第0行
      }
#pragma unroll
      for (int n = 0; n < TN; n += 4)
      {
        FETCH_FLOAT4(b_val[0][n]) = FETCH_FLOAT4(SB[nxt_idx][0][tx * TN + n]);
      }
    }

    // 4.2：单独计算最后一个k=BK-1的行/列元素
#pragma unroll
    for (int m = 0; m < TM; m++)
    {
#pragma unroll
      for (int n = 0; n < TN; n++)
      {
        sum[m][n] += a_val[(BK - 1) % 2][m] * b_val[(BK - 1) % 2][n];
      }
    }

    // 反转cur_idx和nxt_idx
    cur_idx = nxt_idx;
  }

  // 5. 将结果写回全局内存
#pragma unroll
  for (int m = 0; m < TM; m++)
  {
#pragma unroll
    for (int n = 0; n < TN; n += 4)
    {
      int gy = by * BM + ty * TM + m;
      int gx = bx * BN + tx * TN + n;

      // 暂时不考虑alpha和beta的情况，直接写回
      FETCH_FLOAT4(C[OFFSET(gy, gx, N)]) = FETCH_FLOAT4(sum[m][n]);
    }
  }
}

template <const int BM, const int BN, const int BK, const int TM, const int TN>
__global__ void __launch_bounds__(256)
    mysgemm_v4_r(int M, int N, int K, float alpha, float *A, float *B, float beta,
                 float *C)
{
  int tx = threadIdx.x % (BN / TN);
  int ty = threadIdx.x / (BN / TN);
  int bx = blockIdx.x;
  int by = blockIdx.y;
  int tid = threadIdx.x;
  constexpr int NT = (BM / TM) * (BN / TN);
  constexpr int A_NUM_PER_THREAD = CEIL_DIV(BM * BK / 4, NT);
  constexpr int B_NUM_PER_THREAD = CEIL_DIV(BK * BN / 4, NT);

  float4 a_next[A_NUM_PER_THREAD]; // 寄存器预读取A加载的数据，数量为A_NUM_PER_THREAD*4
  float4 b_next[B_NUM_PER_THREAD]; // 寄存器预读取B加载的数据，数量为B_NUM_PER_THREAD*4

  // 共享内存双缓存
  __shared__ __align__(16) float SA[2][BK][BM];
  __shared__ __align__(16) float SB[2][BK][BN];

  // 寄存器双缓存
  float a_val[2][TM];
  float b_val[2][TN];

  float sum[TM][TN] = {{0.0f}};

  int cur_idx = 0;

  // 1.提前加载SA[0]和SB[0]
#pragma unroll
  for (int i = 0; i < BK * BM / 4; i += NT)
  {
    if ((BK * BM / 4) % NT != 0 && tid + i >= BK * BM / 4)
      continue;
    // 先暂时不考虑转置
    int sy = (tid + i) / (BK / 4);
    int sx = (tid + i) % (BK / 4) * 4;
    int gy = by * BM + sy;
    int gx = sx + 0;

    float4 a_val = FETCH_FLOAT4(A[OFFSET(gy, gx, K)]);
    // 转置
    SA[cur_idx][sx + 0][sy] = a_val.x;
    SA[cur_idx][sx + 1][sy] = a_val.y;
    SA[cur_idx][sx + 2][sy] = a_val.z;
    SA[cur_idx][sx + 3][sy] = a_val.w;
  }

#pragma unroll
  for (int i = 0; i < BK * BN / 4; i += NT)
  {
    if ((BK * BN / 4) % NT != 0 && tid + i >= BK * BN / 4)
      continue;
    int sy = (tid + i) / (BN / 4);
    int sx = (tid + i) % (BN / 4) * 4;
    int gy = sy + 0;
    int gx = bx * BN + sx;

    // 向量化赋值
    FETCH_FLOAT4(SB[cur_idx][sy][sx]) = FETCH_FLOAT4(B[OFFSET(gy, gx, N)]);
  }

  // 易错点：加载内存需要同步
  __syncthreads();

  // 第0列和第0行元素
#pragma unroll
  for (int m = 0; m < TM; m += 4)
  {
    // SA[ty*TM+m][k]，转置的结果
    FETCH_FLOAT4(a_val[cur_idx][m]) = FETCH_FLOAT4(SA[cur_idx][0][ty * TM + m]); // 第0个共享显存的第0行
  }
#pragma unroll
  for (int n = 0; n < TN; n += 4)
  {
    FETCH_FLOAT4(b_val[cur_idx][n]) = FETCH_FLOAT4(SB[cur_idx][0][tx * TN + n]);
  }

  // cur：当前读取的共享显存索引
  for (int offset = 0; offset < K; offset += BK)
  {
    int nxt_idx = cur_idx ^ 1;
    int nxt_offset = offset + BK; // 第k+1个offset

    // 2.第一级流水线：加载SA[k+1]和SB[k+1]，第k+1列和第k+1行元素
    // 注意：不直接搬运到显存中，先分为两步：1.搬运到寄存器，2.计算后再搬运到显存中
    if (nxt_offset < K)
    {
#pragma unroll
      for (int i = 0; i < BK * BM / 4; i += NT)
      {
        if ((BK * BM / 4) % NT != 0 && tid + i >= BK * BM / 4)
          continue;
        // 先暂时不考虑转置
        int sy = (tid + i) / (BK / 4);
        int sx = (tid + i) % (BK / 4) * 4;
        int gy = by * BM + sy;
        int gx = sx + nxt_offset;

        // 先放在寄存器中
        a_next[i / NT] = FETCH_FLOAT4(A[OFFSET(gy, gx, K)]);
      }

#pragma unroll
      for (int i = 0; i < BK * BN / 4; i += NT)
      {
        if ((BK * BN / 4) % NT != 0 && tid + i >= BK * BN / 4)
          continue;
        int sy = (tid + i) / (BN / 4);
        int sx = (tid + i) % (BN / 4) * 4;
        int gy = sy + nxt_offset;
        int gx = bx * BN + sx;

        // 先放在寄存器中
        b_next[i / NT] = FETCH_FLOAT4(B[OFFSET(gy, gx, N)]);
      }
    }

    // 3.第二级流水线：计算第k行/列元素，写入k+1行/列元素
    // 默认k%2==0，放在第0个寄存器槽位，k+1%2==1，放在第1个寄存器槽位
    // 且默认起始计算的寄存器槽位为0！
#pragma unroll
    for (int k = 0; k < BK - 1; k++)
    {
      // 写入第k+1行/列元素
      for (int m = 0; m < TM; m += 4)
      {
        // 始终使用当前idx的共享显存
        FETCH_FLOAT4(a_val[(k + 1) % 2][m]) = FETCH_FLOAT4(SA[cur_idx][k + 1][ty * TM + m]);
      }
#pragma unroll
      for (int n = 0; n < TN; n += 4)
      {
        FETCH_FLOAT4(b_val[(k + 1) % 2][n]) = FETCH_FLOAT4(SB[cur_idx][k + 1][tx * TN + n]);
      }

      // 计算第k行/列元素
#pragma unroll
      for (int m = 0; m < TM; m++)
      {
        for (int n = 0; n < TN; n++)
        {
          sum[m][n] += a_val[k % 2][m] * b_val[k % 2][n];
        }
      }
    }
    if (nxt_offset < K)
    {
      // 4.1：从寄存器写入到内存中
#pragma unroll
      for (int i = 0; i < BK * BM / 4; i += NT)
      {
        if ((BK * BM / 4) % NT != 0 && tid + i >= BK * BM / 4)
          continue;
        // 先暂时不考虑转置
        int sy = (tid + i) / (BK / 4);
        int sx = (tid + i) % (BK / 4) * 4;
        // int gy = by * BM + sy;
        // int gx = sx + nxt_offset;

        float4 tmp = a_next[i / NT];
        SA[nxt_idx][sx + 0][sy] = tmp.x;
        SA[nxt_idx][sx + 1][sy] = tmp.y;
        SA[nxt_idx][sx + 2][sy] = tmp.z;
        SA[nxt_idx][sx + 3][sy] = tmp.w;
      }

#pragma unroll
      for (int i = 0; i < BK * BN / 4; i += NT)
      {
        if ((BK * BN / 4) % NT != 0 && tid + i >= BK * BN / 4)
          continue;
        int sy = (tid + i) / (BN / 4);
        int sx = (tid + i) % (BN / 4) * 4;
        // int gy = sy + nxt_offset;
        // int gx = bx * BN + sx;

        FETCH_FLOAT4(SB[nxt_idx][sy][sx]) = b_next[i / NT];
      }

      __syncthreads();
      // 4.2：读取下一个共享块的第0行/列的元素，始终寄存器槽位0
#pragma unroll
      for (int m = 0; m < TM; m += 4)
      {
        FETCH_FLOAT4(a_val[0][m]) = FETCH_FLOAT4(SA[nxt_idx][0][ty * TM + m]); // 第0个共享显存的第0行
      }
#pragma unroll
      for (int n = 0; n < TN; n += 4)
      {
        FETCH_FLOAT4(b_val[0][n]) = FETCH_FLOAT4(SB[nxt_idx][0][tx * TN + n]);
      }
    }

    // 4.2：单独计算最后一个k=BK-1的行/列元素
#pragma unroll
    for (int m = 0; m < TM; m++)
    {
#pragma unroll
      for (int n = 0; n < TN; n++)
      {
        sum[m][n] += a_val[(BK - 1) % 2][m] * b_val[(BK - 1) % 2][n];
      }
    }

    // 反转cur_idx和nxt_idx
    cur_idx = nxt_idx;
  }

  // 5. 将结果写回全局内存
#pragma unroll
  for (int m = 0; m < TM; m++)
  {
#pragma unroll
    for (int n = 0; n < TN; n += 4)
    {
      int gy = by * BM + ty * TM + m;
      int gx = bx * BN + tx * TN + n;

      // 暂时不考虑alpha和beta的情况，直接写回
      FETCH_FLOAT4(C[OFFSET(gy, gx, N)]) = FETCH_FLOAT4(sum[m][n]);
    }
  }
}

std::vector<int> generateSizes()
{
  std::vector<int> sizes;
  for (int i = 256; i <= 8192; i += 256)
  {
    sizes.push_back(i);
  }
  // sizes.push_back(4096);
  return sizes;
}
int main()
{
  std::vector<int> sizes = generateSizes();

  // 打开CSV文件
  std::ofstream csv_file("sgemm_benchmark_v5-double-pipeline_r.csv");
  csv_file << "Size,CUBLAS_GFLOPS,MySGEMM_FLOPS,Matched,Ratio" << std::endl;

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

      dim3 blockDim(256);
      dim3 gridDim(CEIL_DIV(N, 128), CEIL_DIV(N, 128));

      for (int i = 0; i < warpup_time; ++i)
      {
        mysgemm_v4_r<128, 128, 8, 8, 8>
            <<<gridDim, blockDim>>>(N, N, N, alpha, d_A, d_B, beta, d_C_v1);
      }
      cudaDeviceSynchronize();

      checkCudaError(cudaEventRecord(start),
                     "cudaEventRecord(start v1) failed");
      for (int i = 0; i < repeat_time; ++i)
      {
        mysgemm_v4_r<128, 128, 8, 8, 8>
            <<<gridDim, blockDim>>>(N, N, N, alpha, d_A, d_B, beta, d_C_v1);
      }
      checkCudaError(cudaEventRecord(stop), "cudaEventRecord(stop v1) failed");
      checkCudaError(cudaEventSynchronize(stop),
                     "cudaEventSynchronize v1 failed");
      checkCudaError(cudaGetLastError(), "cuda get last error failed");
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

      float ratio = v1_gflops / cublas_gflops;
      // 写入CSV
      csv_file << N << "," << cublas_gflops << "," << v1_gflops << ","
               << (error_count == 0 ? "1" : "0") << "," << ratio << std::endl;

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
      cudaDeviceSynchronize();
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
