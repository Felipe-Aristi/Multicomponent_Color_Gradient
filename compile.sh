#!/usr/bin/env bash
set -e

nvcc -O3 -std=c++20 -rdc=true -Xptxas -v \
  -gencode arch=compute_86,code=sm_86 \
  -gencode arch=compute_86,code=lto_86 \
  main.cu \
  stencil.cu \
  kernels.cu \
  -o main 

# ./main