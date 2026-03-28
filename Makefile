default:
	nvcc -O3 -std=c++20 --restrict --extended-lambda -Xptxas -v -rdc=true -gencode arch=compute_86,code=sm_86 -gencode arch=compute_86,code=lto_86 main.cu kernels.cu -DWEBER=2500 -o main

clean:
	rm main

# ./main
