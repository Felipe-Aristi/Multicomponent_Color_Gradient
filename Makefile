default:
	nvcc -O3 -std=c++20 --extended-lambda -Xptxas -v -rdc=true -gencode arch=compute_89,code=sm_89 -gencode arch=compute_89,code=lto_89 main.cu kernels.cu -DWEBER=2500 -o main

clean:
	rm main

# ./main
