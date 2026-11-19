# SGEMM 统一基准

在 `ml-course` 下执行：

```bash
python course5/benchmark_all.py
```

脚本在 `build/course5/all_gemm_*` 创建独立目录，用 `make` 编译，然后依次测试 cuBLAS SGEMM、matmul0、matmul1、matmul2_1、matmul2_2、matmul3_1、matmul3_2、matmul4。每个手写版本同时测试原实现与 `_r`。各文件独立运行时默认调用 `_r`。

可以通过 `--device` 指定 GPU，通过 `--nvcc` 指定编译器，通过 `--output` 指定结果目录。`benchmark_all.cu.in` 是生成基准的模板，不直接编译；kernel 从当前源文件提取，只修改函数名，源码快照、SHA256 和启动配置写入结果目录。原有 `course*` 参考代码不参与此次基准。

## 对比条件

- 当前仓库的编译参数：`-g -std=c++14 --generate-code=arch=compute_52,code=[compute_52,sm_52]`。未启用设备调试 `-G`、fast math 或强制寄存器上限；展开指令仅使用普通 `#pragma unroll`。保留各实现原有的 launch bounds。
- 所有版本使用 FP32、行优先、`M=N=K`、输出 `C=N×N`、`alpha=1`、`beta=0`。矩阵尺寸为 128、256、512、1024、2048、4096、8192，均满足现有向量化版本的整 tile 条件。
- 输入使用固定种子 20261118、均匀分布 `[-1,1]`。cuBLAS 使用 `cublasSgemm` 和 `CUBLAS_PEDANTIC_MATH`，控制为 FP32 而非 TF32。交换 A/B 参数适配行优先布局。
- 所有版本各预热 3 次，单独校准一次耗时，以 `clamp(ceil(20ms / 单次耗时), 1, 30)` 确定每轮重复次数；5 轮轮换执行，报告单次耗时中位数。校准和预热不计入结果。
- 使用 CUDA Event，只统计计算调用，不包含初始化、传输、校验；FMA 计 2 次浮点运算，`TFLOPS = 2MNK / (ms × 1e9)`。
- 每个手写版本的全部输出与 cuBLAS 比较，检查非有限值，容差 `atol=1e-3, rtol=1e-4`；cuBLAS 结果另外与 CPU double 累加抽样比较。此测试不验证任意 alpha/beta 或尾块。

各算法的线程组织保持其原有设计，不强行统一 block：

| 版本 | Block | C tile/block | 其他参数 |
|---|---|---|---|
| matmul0 | 32×32 | 32×32 | 每线程 1 个输出 |
| matmul1 | 32×32 | 32×32 | BK=32 |
| matmul2_1 | 128×8 | 128×128 | BK=8, TM=16 |
| matmul2_2 / matmul3_1 / matmul3_2 | 16×16 | 128×128 | BK=8, TM=TN=8 |
| matmul4 | 256×1 | 128×128 | BK=8, TM=TN=8，逻辑线程分工为 16×16 |

## 输出

- `comparison.csv`：默认 `_r` 版本和 cuBLAS 在各尺寸下的 TFLOPS。
- `optimization_comparison.csv`：各原版与 `_r` 的性能和提速比。
- `performance.csv`、各版本独立 CSV：耗时中位数、范围、TFLOPS、重复次数。
- `all_runs.csv`：全部轮次的原始计时。
- `correctness.csv`：全输出正确性结果。
- `resources.csv`：实际 GPU 的寄存器、本地内存、共享内存和可驻留 block 数。
- `environment.txt`、`manifest.json`、源码快照：环境、启动参数和来源。

寄存器数量与性能分别报告；较少的寄存器不保证更快。不同时间生成的 CSV 不宜直接作为细微性能差距的证据。
