# HPMTTKRP

Exploiting Hierarchical Parallelism and Reusability in Tensor Kernel Processing on Heterogeneous HPC Systems

## Tensor Format

The input format is expected to start with the length of each dimension and the number of nonzero elements. The following lines will have the coordinates and values of each nonzero elements. 

An example of a 5x3x3 tensor - test.tns:

5	3	3	8

2	3	1	1.00

2	3	2	2.00

2	3	3	10.00

3	1	3	7.00

3	3	1	6.00

3	3	2	5.00

5	3	2	3.00

5	1	3	1.00


## Build requirements:

- GCC Compiler

- CUDA SDK

- OpenMP


## Build

make all


## Run

1. run ./data_process
2. enter the input filename (for example: ./test.tns)
3. enter the output filename (for example: ./test_output.tns), the useless data in test_output.tns was deleted
4. run ./multi_hpsptm1
5. enter the input filename (for example: ./test_output.tns)
6. enter the length of each of the other two dimensions in test_output.tns (Y and Z) and the rank dimension (R) (for example: 2 3 16)
7. enter the tile size (ix, iy, and iz), the number of threads in each GPU, and the number of GPUs (for example: 2 1 2 2 2)

