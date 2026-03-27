rm main
nvcc -O3 -std=c++20 --extended-lambda -rdc=true -gencode arch=compute_89,code=sm_89 -gencode arch=compute_89,code=lto_89 main.cu kernels.cu -DWEBER=10 -o main
./main

rm main
nvcc -O3 -std=c++20 --extended-lambda -rdc=true -gencode arch=compute_89,code=sm_89 -gencode arch=compute_89,code=lto_89 main.cu kernels.cu -DWEBER=100 -o main
./main

rm main
nvcc -O3 -std=c++20 --extended-lambda -rdc=true -gencode arch=compute_89,code=sm_89 -gencode arch=compute_89,code=lto_89 main.cu kernels.cu -DWEBER=500 -o main
./main

rm main
nvcc -O3 -std=c++20 --extended-lambda -rdc=true -gencode arch=compute_89,code=sm_89 -gencode arch=compute_89,code=lto_89 main.cu kernels.cu -DWEBER=2000 -o main
./main

rm main
nvcc -O3 -std=c++20 --extended-lambda -rdc=true -gencode arch=compute_89,code=sm_89 -gencode arch=compute_89,code=lto_89 main.cu kernels.cu -DWEBER=20000 -o main
./main