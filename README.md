# CUDA 学习与复现（ml-course）

个人 CUDA 课程复现代码，按课程逐步添加实现与学习注释。本目录是独立的 CMake 项目，只构建 `ml-course` 内的代码，不引入上级目录的原始 `course*` 示例。

## 当前内容

| 目录 | 内容 | 可执行目标 |
| --- | --- | --- |
| `course2/` | 显存分配与 CPU/GPU 数据复制 | `malloc_cpy` |
| `course3/` | Warp Shuffle、Softmax 的 CPU/GPU 实现 | `shuffle`、`softmax` |
| `course4/` | Reduce 求和及逐步优化 | `reduce` |

## 环境要求

- CMake 3.18 或以上、支持 C++14 的 C++ 编译器。
- CUDA Toolkit（包含 `nvcc` 和 CUDA Runtime）；当前使用 CUDA 12.8 验证构建。
- 运行示例需要可用的 NVIDIA GPU 和兼容的驱动。

## 构建与运行

从本目录执行，确保 `cmake` 和 `nvcc` 已加入 `PATH`：

```bash
cmake -S . -B build
cmake --build build -j 2

./build/course2/malloc_cpy
./build/course3/shuffle
./build/course3/softmax
./build/course4/reduce
```

默认构建类型为 `Debug`。性能测试使用单独的 `Release` 构建目录：

```bash
cmake -S . -B build/release -DCMAKE_BUILD_TYPE=Release
cmake --build build/release -j 2
```

如果 CMake 找不到 CUDA，可在首次配置时指定安装位置：

```bash
cmake -S . -B build \
  -DCMAKE_CUDA_COMPILER=/home/shared/ml/cuda-12.8/bin/nvcc \
  -DCUDAToolkit_ROOT=/home/shared/ml/cuda-12.8
```

VS Code CMake Tools 的源码目录应指向本目录，而非上级 `cuda_code/`。构建产物放在 `build/`，不提交到 Git。

## 新增示例

在对应课程目录添加源文件，并在该目录的 `CMakeLists.txt` 中显式注册，例如：

```cmake
add_executable(vec_add vec_add.cu)
target_link_libraries(vec_add PRIVATE CUDA::cudart)
```

新增课程目录时，还需在本目录的 `CMakeLists.txt` 中添加 `add_subdirectory(courseN)`。只注册已经复现的文件。

## 使用 Nsight Compute

在有 GPU 的计算节点上，用命令行采集 Kernel 指标：

```bash
ncu --set basic --launch-count 1 -o build/shuffle_profile \
  ./build/release/course3/shuffle
```

生成的 `.ncu-rep` 报告可以下载到本地，用 Nsight Compute GUI 打开。普通用户需要管理员开放 GPU 性能计数器权限；遇到 `ERR_NVGPUCTRPERM` 时，应联系管理员，安装工具本身不会解除驱动权限限制。`malloc_cpy` 不启动 Kernel，适合练习内存复制，不用于此类 Kernel 分析。

这些代码用于学习；构建成功不代表所有运行结果均已验证。修改 Kernel 后，应在可用 GPU 上检查输出及 CPU/GPU 结果一致性。
