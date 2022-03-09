
GCC = gcc
NVCC = nvcc

NVCCFLAGS += -Xcompiler -fopenmp -arch=sm_60

all: data_process hpsptm1

pre: data_process.c
	${GCC} -o data_process data_process.c

hpsptm1: hpsptm1.cu
	${NVCC} ${NVCCFLAGS} -o hpsptm1 hpsptm1.cu

clean:
	-rm -rf data_process hpsptm1
