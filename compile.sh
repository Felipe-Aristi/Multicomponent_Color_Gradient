#!/usr/bin/env bash
set -e

nvcc -O3 -std=c++17 -rdc=true \
  main.cu \
  stencil.cu \
  kernels.cu \
  -o main

./main