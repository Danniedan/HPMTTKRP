
GCC = gcc
NVCC = nvcc

NVCCFLAGS += -Xcompiler -fopenmp -arch=sm_60

all: data_process hpsptm1 hpsptm2 multi_hpsptm1

pre: data_process.c
	${GCC} -o data_process data_process.c

hpsptm1: hpsptm1.cu
	${NVCC} ${NVCCFLAGS} -o hpsptm1 hpsptm1.cu
	
hpsptm2: hpsptm2.cu
	${NVCC} ${NVCCFLAGS} -o hpsptm2 hpsptm2.cu

multi_hpsptm1: multi_hpsptm1.cu
	${NVCC} ${NVCCFLAGS} -o multi_hpsptm1 multi_hpsptm1.cu

clean:
	-rm -rf data_process hpsptm1 hpsptm2 multi_hpsptm1
