#!/usr/bin/env bash
set -e

nvcc -O3 -std=c++20 --extended-lambda -rdc=true \
  -gencode arch=compute_89,code=sm_89 \
  -gencode arch=compute_89,code=lto_89 \
  main.cu \
  kernels.cu \
  -o main

# ./main
