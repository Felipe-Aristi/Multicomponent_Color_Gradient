default:
	nvcc -O3 -std=c++20 -rdc=true -gencode arch=compute_86,code=sm_86 -gencode arch=compute_86,code=lto_86 main.cu stencil.cu kernels.cu -o main

clean:
	rm main

# ./main
