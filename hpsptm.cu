
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "device_functions.h"
#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include <stdlib.h>
#include <ctype.h>
#include <math.h>
#include <omp.h>
#include <math.h>

const int y = 1729;
const int z = 12;
const int r = 16;

__global__ void SPTM(int NUM_THREAD, int NUM_BLOCK, int num_column, int num_row,
int* mark, int* Ap, int* col, int* rows, double* values, double* B, double* C, double* D) {
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	if (idx < NUM_THREAD) {
		long long i, j, k, m, n;
		
		for(n=mark[idx]; n<mark[idx+1]; n++)
		{
			for(m=Ap[n]; m<Ap[n+1]; m++)
			{
				for(i=0; i<r; i++)
				{
					D[(num_row*i+rows[m])] += values[m] * C[z*i+(col[m]/y)] * B[y*i+(col[m]%y)];
				}
			}
		}
		
	}
}

int com(double* A, double* B, int num) {
	int flag = 1;
	for (int i = 0; i < num; i++) {
		if (fabs(A[i] - B[i]) > 0.00001) {
			printf("%lf %lf %d\n", A[i], B[i], i);
			flag = 0;
		}
	}
	return flag;
}

int compare(int* A, int* B, int num) {
	for (int i = 0; i < num; i++) {
		if (A[i] != B[i])
			return 0;
	}
	return 1;
}

int main(int argc, char** argv)
{

	struct timeval startt;
    struct timeval endt;
    unsigned long timer;
	
    FILE *f, *fw;
    int M;
	int N;
	long nz;	
	long long i, j, k, b, q;
	
	char * filename="/your/path/to/dataset.tns";

    if ((f = fopen(filename, "r")) == NULL) 
            exit(1);
	
	fscanf(f, "%d	%d	%ld", &N, &M,&nz);
	
    int num_row;
	num_row=N;
	int num_column;
	num_column=M;
	long nnz;
	nnz=nz;
	printf("num_row=%d, num_column=%d, nnz=%ld\n", num_row, num_column, nnz);
	
    /* reseve memory for matrices */
	
	
	int *rows;
	cudaHostAlloc((int**)&rows, nnz * sizeof(int), cudaHostAllocDefault);
	long long *columns=(long long *) malloc(nnz * sizeof(long long));
	double *values;
	cudaHostAlloc((double**)&values, nnz * sizeof(double), cudaHostAllocDefault);

    /* NOTE: when reading in doubles, ANSI C requires the use of the "l"  */
    /*   specifier as in "%lg", "%lf", "%le", otherwise errors will occur */
    /*  (ANSI C X3.159-1989, Sec. 4.9.6.2, p. 136 lines 13-15)            */
	//assume iz = Z
	int iz = z;
	int NUM_THREAD=1024;
	int NUM_BLOCK;
	if((iz%2)==0)
		NUM_BLOCK = NUM_THREAD * z;
	else
		NUM_BLOCK = NUM_THREAD * (z-1);
	int row;
	int column;
	double value;
	long g=0;
	
	int *numnonperblock=(int *) malloc((NUM_BLOCK+1) * sizeof(int));
	numnonperblock[0]=0;
	
	
	q=0;
	
	for(k=0; k<z; k++)
	{	
		printf("k=%lld\n", k);
		for(j=0; j<NUM_THREAD; j++)
		{
			for (i=0; i<nnz; i++)
			{
				fscanf(f, "%d	%d", &(row), &(column));
				if((k==0)&&((iz%2)!=0))
				{
					if( (column-1)>=(y*k) && (column-1)<(y*(k+2)) )
					{
						if( ((row-1)>=(num_row*j/NUM_THREAD)) && ((row-1)<(num_row*(j+1)/NUM_THREAD)) )
						{
							rows[g] = row-1;  /* adjust from 1-based to 0-based */
							columns[g] = column-1;
							values[g] = 1.0;
							g++;
						}
					}
				}
				else
				{
					if( (column-1)>=(y*k) && (column-1)<(y*(k+1)) )
					{
						if( ((row-1)>=(num_row*j/NUM_THREAD)) && ((row-1)<(num_row*(j+1)/NUM_THREAD)) )
						{
							rows[g] = row-1;  /* adjust from 1-based to 0-based */
							columns[g] = column-1;
							values[g] = 1.0;
							g++;
						}
					}
				}
			}
			rewind(f);
			fscanf(f, "%d	%d	%ld", &N, &M,&nz);
			if((k!=0)&&((iz%2)!=0))
				numnonperblock[(k-1)*NUM_THREAD+j+1] = g;
			else
				numnonperblock[k*NUM_THREAD+j+1] = g;
		}
		if((k==0)&&((iz%2)!=0))
			k++;
	}

	printf("g = %ld\n", g);
	if (f !=stdin) fclose(f);

	
	
	long long num_c;
	num_c = (long long)num_column * (long long)NUM_BLOCK;
	
	int *Ap;
	cudaHostAlloc((int**)&Ap, (num_c+1) * sizeof(int), cudaHostAllocDefault);
	int *col;
	cudaHostAlloc((int**)&col, num_c * sizeof(int), cudaHostAllocDefault);
	int *mark;
	cudaHostAlloc((int**)&mark, (NUM_BLOCK+1) * sizeof(int), cudaHostAllocDefault);
	
	int maxl=0;
	Ap[0]=0;
	mark[0]=0;
	j=0;
	
	int *num_nonzeros=(int *) malloc(num_column * sizeof(int));
	int sum=0;
	
	for(b=0; b<NUM_BLOCK; b++)
	{
		for(i=0; i<num_column; i++)
			num_nonzeros[i]=0;
		for(i=numnonperblock[b];i<numnonperblock[b+1];i++)
		{
			num_nonzeros[columns[i]]++;
		}
		for(k=0; k<num_column; k++)
		{
			if(num_nonzeros[k]!=0)
			{
				sum+=num_nonzeros[k];
				Ap[j+1]=sum;
				if(maxl < (Ap[j+1]-Ap[j]))
				{
					maxl=Ap[j+1]-Ap[j];	
				}
				col[j] = k;
				j++;
			}
		}
		mark[b+1] = j;
	}
	long num_noempcol;
	num_noempcol=j;
	printf("num_noempcol: %lld\n", num_noempcol);
	printf("maxl: %d\n", maxl);
	free(num_nonzeros);
	
	
	
	double *B, *C, *D;
	cudaHostAlloc((double**)&B, y*r * sizeof(double), cudaHostAllocDefault);
	cudaHostAlloc((double**)&C, z*r * sizeof(double), cudaHostAllocDefault);
	cudaHostAlloc((double**)&D, num_row*r * sizeof(double), cudaHostAllocDefault);
	
	for(i=0; i<(y*r); i++)
		B[i]=1.0;
	for(i=0; i<(z*r); i++)
		C[i]=1.0;
	for(i=0; i<(num_row*r); i++)
		D[i]=0.0;
	
	
	
	dim3 block(64*1);
	dim3 grid(64);
	int* mark_device1, * mark_device2;
	int* Ap_device1, * Ap_device2;
	int* col_device1, * col_device2;
	int* rows_device1, * rows_device2;
	double* values_device1, * values_device2;
	double* B_device;
	double* C_device;
	double* D_device1, * D_device2;
	
	double* D_host = (double*)malloc(num_row*r*sizeof(double));


	cudaMalloc((int**)&mark_device1, (NUM_THREAD + 1) * sizeof(int));
	cudaMalloc((int**)&mark_device2, (NUM_THREAD + 1) * sizeof(int));
	cudaMalloc((int**)&Ap_device1, (num_noempcol + 1) * sizeof(int));
	cudaMalloc((int**)&Ap_device2, (num_noempcol + 1) * sizeof(int));
	cudaMalloc((int**)&col_device1, num_noempcol * sizeof(int));
	cudaMalloc((int**)&col_device2, num_noempcol * sizeof(int));
	cudaMalloc((int**)&rows_device1, nnz * sizeof(int));
	cudaMalloc((int**)&rows_device2, nnz * sizeof(int));
	cudaMalloc((double**)&values_device1, nnz * sizeof(double));
	cudaMalloc((double**)&values_device2, nnz * sizeof(double));
	cudaMalloc((double**)&B_device, (y*r) * sizeof(double));
	cudaMalloc((double**)&C_device, z*r * sizeof(double));
	cudaMalloc((double**)&D_device1, num_row*r * sizeof(double));
	cudaMalloc((double**)&D_device2, num_row*r * sizeof(double));
	
	
	//Timer setup
	float milliseconds = 0;
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

	cudaStream_t stream0, stream1;
	cudaStreamCreate(&stream0);
	cudaStreamCreate(&stream1);
	
	
	
	int iz_r;
	if((iz%2)!=0)
		iz_r = iz+1;
	else
		iz_r = iz;
	
	q=0;
	
	int num_tilec;
	num_tilec = z/iz;
	
	int count = 0;
	
	
	cudaEventRecord(start);
	cudaMemcpyAsync(B_device, B, (y*r) * sizeof(double), cudaMemcpyHostToDevice, stream0);
	
	
	while(q<z)
	{
		cudaMemcpyAsync(C_device, C+(q*r), (iz_r*r) * sizeof(double), cudaMemcpyHostToDevice, stream0);
		
		for(i=0; i<(iz-1); i+=2)
		{
			cudaMemcpyAsync(mark_device1, mark + ((count*(iz-1)+i)*NUM_THREAD), 
				( ((count*(iz-1)+i+1)*NUM_THREAD)-((count*(iz-1)+i)*NUM_THREAD) + 1) * sizeof(int), 
				cudaMemcpyHostToDevice, stream0);
			cudaMemcpyAsync(Ap_device1, Ap + mark[(count*(iz-1)+i)*NUM_THREAD], 
				(mark[(count*(iz-1)+i+1)*NUM_THREAD]-mark[(count*(iz-1)+i)*NUM_THREAD] + 1) * sizeof(int), 
				cudaMemcpyHostToDevice, stream0);
			cudaMemcpyAsync(col_device1, col + mark[(count*(iz-1)+i)*NUM_THREAD], 
				(mark[(count*(iz-1)+i+1)*NUM_THREAD]-mark[(count*(iz-1)+i)*NUM_THREAD]) * sizeof(int), 
				cudaMemcpyHostToDevice, stream0);
			cudaMemcpyAsync(rows_device1, rows + Ap[mark[(count*(iz-1)+i)*NUM_THREAD]], 
				(Ap[mark[(count*(iz-1)+i+1)*NUM_THREAD]] - Ap[mark[(count*(iz-1)+i)*NUM_THREAD]]) * sizeof(int), 
				cudaMemcpyHostToDevice, stream0);
			cudaMemcpyAsync(values_device1, values + Ap[mark[(count*(iz-1)+i)*NUM_THREAD]], 
				(Ap[mark[(count*(iz-1)+i+1)*NUM_THREAD]] - Ap[mark[(count*(iz-1)+i)*NUM_THREAD]]) * sizeof(double), 
				cudaMemcpyHostToDevice, stream0);
				
				
			cudaMemcpyAsync(mark_device2, mark + ((count*(iz-1)+i+1)*NUM_THREAD), 
				( ((count*(iz-1)+i+2)*NUM_THREAD)-((count*(iz-1)+i+1)*NUM_THREAD) + 1) * sizeof(int), 
				cudaMemcpyHostToDevice, stream1);
			cudaMemcpyAsync(Ap_device2, Ap + mark[(count*(iz-1)+i+1)*NUM_THREAD], 
				(mark[(count*(iz-1)+i+2)*NUM_THREAD]-mark[(count*(iz-1)+i+1)*NUM_THREAD] + 1) * sizeof(int), 
				cudaMemcpyHostToDevice, stream1);
			cudaMemcpyAsync(col_device2, col + mark[(count*(iz-1)+i+1)*NUM_THREAD], 
				(mark[(count*(iz-1)+i+2)*NUM_THREAD]-mark[(count*(iz-1)+i+1)*NUM_THREAD]) * sizeof(int), 
				cudaMemcpyHostToDevice, stream1);
			cudaMemcpyAsync(rows_device2, rows + Ap[mark[(count*(iz-1)+i+1)*NUM_THREAD]], 
				(Ap[mark[(count*(iz-1)+i+2)*NUM_THREAD]] - Ap[mark[(count*(iz-1)+i+1)*NUM_THREAD]]) * sizeof(int), 
				cudaMemcpyHostToDevice, stream1);
			cudaMemcpyAsync(values_device2, values + Ap[mark[(count*(iz-1)+i+1)*NUM_THREAD]], 
				(Ap[mark[(count*(iz-1)+i+2)*NUM_THREAD]] - Ap[mark[(count*(iz-1)+i+1)*NUM_THREAD]]) * sizeof(double), 
				cudaMemcpyHostToDeåvice, stream1);
				
			
			SPTM << < grid, block, 0, stream0 >> > (NUM_THREAD, NUM_BLOCK, num_column, num_row, 
				mark_device1, Ap_device1, col_device1, rows_device1, values_device1, B_device, C_device, D_device1);
			SPTM << < grid, block, 0, stream1 >> > (NUM_THREAD, NUM_BLOCK, num_column, num_row, 
				mark_device2, Ap_device2, col_device2, rows_device2, values_device2, B_device, C_device, D_device1);
		}
		
		count++;
		q+=iz_r;
		iz_r = iz;
	}
	
	cudaMemcpyAsync(D_host, D_device1, num_row*r * sizeof(double), cudaMemcpyDeviceToHost, stream0);
	
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("return milliseconds = %lf\n", milliseconds);
	
	
	cudaDeviceReset();

	return 0;
}
