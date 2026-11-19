#!/usr/bin/env python3
"""Snapshot the course5 kernels, build with make, and run a common SGEMM benchmark."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import tempfile


def extract(source, name):
    match = re.search(r"\bvoid\s+(?:__launch_bounds__\([^)]*\)\s*)?" + name + r"\s*\(", source)
    if match is None:
        raise ValueError(f"Kernel not found: {name}")
    start = source.rfind("template <", 0, match.start())
    global_start = source.rfind("__global__", 0, match.start())
    if start < 0 or "\n}" in source[start:global_start]:
        start = global_start
    end = source.index("{", match.end()) + 1
    depth = 1
    while depth:
        depth += (source[end] == "{") - (source[end] == "}")
        end += 1
    return source[start:end]


def summarize(out):
    rows = list(csv.DictReader((out / "performance.csv").open()))
    checks = list(csv.DictReader((out / "correctness.csv").open()))
    if len(rows) != 105 or len(checks) != 98 or any(r["Matched"] != "1" for r in checks):
        raise RuntimeError("Incomplete benchmark or failed correctness checks")
    sizes = sorted({int(r["N"]) for r in rows})
    versions = ["cuBLAS_FP32", "matmul0_r", "matmul1_r", "matmul2_1_r", "matmul2_2_r", "matmul3_1_r", "matmul3_2_r", "matmul4_r"]
    values = {(r["Version"], int(r["N"])): float(r["TFLOPS"]) for r in rows}
    with (out / "comparison.csv").open("w") as f:
        writer = csv.writer(f)
        writer.writerow(["N"] + versions)
        for n in sizes:
            writer.writerow([n] + [values[v, n] for v in versions])
    with (out / "optimization_comparison.csv").open("w") as f:
        writer = csv.writer(f)
        writer.writerow(["Version", "N", "Original_TFLOPS", "Optimized_TFLOPS", "Speedup"])
        for v in versions[1:]:
            for n in sizes:
                a, b = values[v[:-2], n], values[v, n]
                writer.writerow([v, n, a, b, b / a])
    for v in [r["Version"] for r in rows if int(r["N"]) == sizes[0]]:
        with (out / (v + ".csv")).open("w") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(r for r in rows if r["Version"] == v)
    report = "# SGEMM 实测结果\n\n单位：TFLOPS。各文件默认使用 `_r`，原版也保留并测试。\n\n"
    report += "| N | " + " | ".join(versions) + " |\n"
    report += "|---:|" + "---:|" * len(versions) + "\n"
    for n in sizes:
        report += f"| {n} | " + " | ".join(f"{values[v, n]:.3f}" for v in versions) + " |\n"
    report += f"\n{len(checks)} 项全输出正确性检查均通过，最大绝对误差为 {max(float(r['MaxAbsError']) for r in checks):.6g}。\n\n"
    report += "## 原版与 _r 对照（8192）\n\n| 版本 | 原版 TFLOPS | _r TFLOPS | 提速比 |\n|---|---:|---:|---:|\n"
    for v in versions[1:]:
        a, b = values[v[:-2], 8192], values[v, 8192]
        report += f"| {v[:-2]} | {a:.3f} | {b:.3f} | {b / a:.3f}x |\n"
    report += "\n完整数据见 comparison.csv、optimization_comparison.csv、performance.csv、all_runs.csv、correctness.csv 和 resources.csv。\n"
    report += "\n结果仅适用于本次 GPU、编译参数、整 tile 和 alpha=1、beta=0。小差距可能受频率和系统负载影响；没有采集性能计数器。方法见 METHOD.md；在本目录执行 make 和 ./benchmark 可以重跑快照。\n"
    (out / "README.md").write_text(report)
    shutil.copyfile(Path(__file__).with_name("BENCHMARK.md"), out / "METHOD.md")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--nvcc", default=shutil.which("nvcc") or "/home/shared/ml/cuda-12.8/bin/nvcc")
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()
    here = Path(__file__).resolve().parent
    parent = here.parent / "build" / "course5"
    parent.mkdir(parents=True, exist_ok=True)
    out = args.output.resolve() if args.output else Path(tempfile.mkdtemp(prefix="all_gemm_", dir=parent))
    out.mkdir(parents=True, exist_ok=True)
    specs = [
        ("matmul0", "mysgemm_v1", "", 32, 32, 32),
        ("matmul1", "mysgemm_v2", "<32>", 32, 32, 32),
        ("matmul2_1", "mysgemm_v3_1", "<128,128,8,16>", 128, 8, 128),
        ("matmul2_2", "mysgemm_v4", "<128,128,8,8,8>", 16, 16, 128),
        ("matmul3_1", "mysgemm_v3_1", "<128,128,8,8,8>", 16, 16, 128),
        ("matmul3_2", "mysgemm_v3_2", "<128,128,8,8,8>", 16, 16, 128),
        ("matmul4", "mysgemm_v4", "<128,128,8,8,8>", 256, 1, 128),
    ]
    kernels = """#include <cuda_runtime.h>
#define OFFSET(row,col,ld) ((row)*(ld)+(col))
#define FETCH_FLOAT4(pointer) (reinterpret_cast<float4 *>(&(pointer))[0])
#define CEIL_DIV(M,N) (((M)+(N)-1)/(N))
"""
    names, launches, resources, hashes, configurations = [], [], [], {}, []
    for filename, function, template, x, y, tile in specs:
        path = here / (filename + ".cu")
        source = path.read_text()
        hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        (out / path.name).write_text(source)
        for suffix in ("", "_r"):
            name = filename + suffix
            alias = "bench_" + name
            original = function + suffix
            kernels += extract(source, original).replace(original + "(", alias + "(") + "\n"
            index = len(names)
            names.append(name)
            launches.append(f"case {index}: {alias}{template}<<<dim3(N/{tile},M/{tile}),dim3({x},{y})>>>(M,N,K,1,A,B,0,C); break;")
            resources.append(f'resource(res,"{name}",{alias}{template},{x * y});')
            configurations.append(dict(name=name, block=[x, y, 1], tile=tile, template=template, default=bool(suffix)))
    (out / "kernels.cuh").write_text(kernels)
    code = (here / "benchmark_all.cu.in").read_text()
    code = code.replace("@NAMES@", ",".join(json.dumps(n) for n in names + ["cuBLAS_FP32"]))
    code = code.replace("@LAUNCHES@", "\n".join(launches)).replace("@RESOURCES@", "\n".join(resources))
    code = code.replace("@COUNT@", str(len(names))).replace("@DEVICE@", str(args.device))
    (out / "benchmark.cu").write_text(code)
    (out / "manifest.json").write_text(json.dumps(dict(sources=hashes, configurations=configurations), indent=2))
    (out / "Makefile").write_text(f"NVCC := {args.nvcc}\nbenchmark: benchmark.cu kernels.cuh\n\t$(NVCC) -g -std=c++14 --generate-code=arch=compute_52,code=[compute_52,sm_52] benchmark.cu -lcublas -o benchmark\n")
    print(f"Results: {out}", flush=True)
    subprocess.run(["make", "-C", str(out)], check=True)
    with (out / "run.log").open("w") as log:
        result = subprocess.run([str(out / "benchmark")], cwd=out, stdout=log, stderr=subprocess.STDOUT)
    print((out / "run.log").read_text(), end="")
    if result.returncode:
        raise SystemExit(result.returncode)
    for path, digest in hashes.items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != digest:
            raise RuntimeError(f"Source changed during benchmark: {path}")
    summarize(out)


if __name__ == "__main__":
    main()
