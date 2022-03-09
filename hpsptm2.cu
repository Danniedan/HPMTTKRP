
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
//#include "device_functions.h"
//#include "cuda_runtime.h"
//#include "cuda_runtime_api.h"
#ifndef __CUDACC__
    #define __CUDACC__
    #include <device_functions.h>
#endif
#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include <stdlib.h>
#include <ctype.h>
#include <math.h>
#include <omp.h>
#include <math.h>


__global__ void SPMV_multi(int NUM_THREAD, int NUM_TILE, int num_row, int j, int k, int ix, int ir, int y,
int* PtoNUM_NECOlperTILE, int* Cp, int* Ci, int* rows, double* values, double* B, double* C, double* D) {
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	//printf("idx=%d\n",idx);
	if (idx < NUM_THREAD) {
		int i, m, n;
		int startD;
		int startn, endn, startm, endm;
		
		startn = PtoNUM_NECOlperTILE[idx]-PtoNUM_NECOlperTILE[0];
		endn = PtoNUM_NECOlperTILE[idx+1]-PtoNUM_NECOlperTILE[0];
		
		startD = idx*ix;
		
		for(n=startn; n<endn; n++)
		{
			startm = Cp[n]-Cp[0];
			endm = Cp[n+1]-Cp[0];
			for(m=startm; m<endm; m++)
			{
				for(i=0; i<ir; i++)
				{
					atomicAdd(&D[(startD+rows[m])*ir+i], values[m] * C[(Ci[n]/y-k)*ir+i] * B[(Ci[n]%y-j)*ir+i]);
				}
			}
		}
		
		
	}
}



int main(int argc, char** argv)
{

	//float milliseconds = 0;
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	//cudaEventRecord(start, 0);
	
    FILE *f;
    int M;
	int N;
	int nz;	
	int i, j, k, b, q;
	int n;
	
	 char filename[256];
    printf("Please Input filename (256 characters or less):\n");
    scanf("%s", filename);
    int y, z, r;
    printf("Please Input Y Z R:\n");
    scanf("%d %d %d", &y, &z, &r);
    int ix, iy, iz, NUM_THREAD, num_gpus;
    printf("Please Input ix iy iz Number_of_Threads Number_of_GPUs:\n");
    scanf("%d %d %d %d %d", &ix, &iy, &iz, &NUM_THREAD, &num_gpus);
	

    if ((f = fopen(filename, "r")) == NULL) 
            exit(1);
	
	fscanf(f, "%d	%d	%d", &N, &M,&nz);
	
    int num_row;
	num_row=N;
	int num_column;
	num_column=M;
	int nnz;
	nnz=nz;
	printf("num_row=%d, num_column=%d, nnz=%d\n", num_row, num_column, nnz);
	
    /* reseve memory for matrices */
	
	int *rows_original=(int *) malloc(nnz * sizeof(int));
	int *columns_original=(int *) malloc(nnz * sizeof(int));
	double *values_original=(double *) malloc(nnz * sizeof(double));
	
	int row;
	int column;
	double value;
	
	for (i=0; i<nnz; i++)
	{
		fscanf(f, "%d	%d	%lf", &(row), &(column), &(value));
		//fscanf(f, "%d	%d", &(row), &(column));
		rows_original[i] = row-1;
		columns_original[i] = column-1;
		values_original[i] = value;
		//rows_original[i] = row-1;
		//columns_original[i] = column-1;
	}
	if (f !=stdin) fclose(f);
	/*****************************************************************/
	


	
	int NUM_SUBBLOCK;
	NUM_SUBBLOCK = (num_row + ix - 1) / ix;
	printf("NUM_SUBBLOCK = %d, ", NUM_SUBBLOCK);
	
	int NUM_TILEperSUBBLOCK;
	NUM_TILEperSUBBLOCK = ( (y+iy-1) / iy) * z;
	printf("NUM_TILEperSUBBLOCK = %d, ", NUM_TILEperSUBBLOCK);
	
	
	int NUM_SUBBLOCKperTHREAD;
	NUM_SUBBLOCKperTHREAD = (NUM_SUBBLOCK + NUM_THREAD -1) / NUM_THREAD;
	printf("NUM_SUBBLOCKperTHREAD = %d, ", NUM_SUBBLOCKperTHREAD);
	
	NUM_SUBBLOCKperTHREAD=NUM_SUBBLOCK/NUM_THREAD;
	for(i=0; i<NUM_THREAD; i++)
	{
		j = NUM_SUBBLOCK*(i+1)/NUM_THREAD - NUM_SUBBLOCK*i/NUM_THREAD;
		if(NUM_SUBBLOCKperTHREAD<j)	NUM_SUBBLOCKperTHREAD=j;
	}
	printf("NUM_SUBBLOCKperTHREAD = %d\n", NUM_SUBBLOCKperTHREAD);
	
	
	int NUM_TILE;
	NUM_TILE = NUM_SUBBLOCKperTHREAD * NUM_THREAD * NUM_TILEperSUBBLOCK;
	printf("NUM_TILE = %d\n", NUM_TILE);
	
	int *numnonpertile=(int *) malloc((NUM_TILE+1) * sizeof(int));
	numnonpertile[0]=0;
	
	
	int *NUM_ITERATE=(int *) malloc((NUM_SUBBLOCKperTHREAD * NUM_TILEperSUBBLOCK) * sizeof(int));

	
	
	
	//tiling
	/************************************************************************************************/
	int *rows_read=(int *) malloc(nnz * sizeof(int));
	int *columns_read=(int *) malloc(nnz * sizeof(int));
	double *values_read=(double *) malloc(nnz * sizeof(double));
	
	
	int g=0;
	
	int izz;
	if((z%iz)==0)
		izz = iz;
	else
		izz = z%iz;
		
	int iyy;
	if((y%iy)==0)
		iyy = iy;
	else
		iyy = y%iy;
		
	int ik, ib, it, inum_tile;
	inum_tile = 0;
	
	q=0;
	
	
	for(ib=0; ib<NUM_SUBBLOCKperTHREAD; ib++)
	{
		k=0;
		if((z%iz)==0)
			izz = iz;
		else
			izz = z%iz;
		
		while(k<z)
		{
			j=0;
			if((y%iy)==0)
				iyy = iy;
			else
				iyy = y%iy;
			
			while(j<y)
			{	
				for(ik=0; ik<izz; ik++)
				{
					for(it=0; it<NUM_THREAD; it++)
					{
						for (i=0; i<nnz; i++)
						{
							if( (columns_original[i]>=(y*(k+ik)+j)) && 
								(columns_original[i] < (y*(k+ik)+j+iyy)) )
							{
								if( ( rows_original[i] >= (num_row*(it*NUM_SUBBLOCKperTHREAD+ib)/(NUM_THREAD*NUM_SUBBLOCKperTHREAD)) ) && 
									( rows_original[i] < (num_row*(it*NUM_SUBBLOCKperTHREAD+ib+1)/(NUM_THREAD*NUM_SUBBLOCKperTHREAD)) ) )
								{
									rows_read[g] = rows_original[i] - (num_row*(it*NUM_SUBBLOCKperTHREAD+ib)/(NUM_THREAD*NUM_SUBBLOCKperTHREAD));
									columns_read[g] = columns_original[i];
									values_read[g] = values_original[i];
									//values_read[g] = 1.0;
									g++;
								}
							}
							else{
								if( columns_original[i]>=(y*(k+ik)+j+iyy) )
									break;
							}
						}
						numnonpertile[inum_tile+1] = g;
						inum_tile++;
					}
					NUM_ITERATE[q]=numnonpertile[(q+1)*NUM_THREAD]-numnonpertile[q*NUM_THREAD];
					q++;
				}
				j+=iyy;
				iyy=iy;
			}
			k+=izz;
			izz=iz;
		}	
	}
	/************************************************************************************************/

	
	printf("******************\n");
	
	free(rows_original);
	free(columns_original);
	free(values_original);
	
	
	
	
	
	//permuting
	/************************************************************************************************/
	int *rows;
	cudaHostAlloc((int**)&rows, nnz * sizeof(int), cudaHostAllocDefault);
	int *columns=(int *) malloc(nnz * sizeof(int));
	double *values;
	cudaHostAlloc((double**)&values, nnz * sizeof(double), cudaHostAllocDefault);
	
	int *numnonpertile_perm=(int *) malloc((NUM_TILE+1) * sizeof(int));
	numnonpertile_perm[0]=0;
	
	int *order=(int *) malloc((NUM_SUBBLOCKperTHREAD * NUM_TILEperSUBBLOCK) * sizeof(int));
	for(i=0; i<(NUM_SUBBLOCKperTHREAD * NUM_TILEperSUBBLOCK); i++)
		order[i] = i;
	
	int min, temp;
	q=0;
	for(ib=0; ib<NUM_SUBBLOCKperTHREAD; ib++)
	{
		k=0;
		if((z%iz)==0)
			izz = iz;
		else
			izz = z%iz;
		while(k<z)
		{
			j=0;
			if((y%iy)==0)
				iyy = iy;
			else
				iyy = y%iy;
			while(j<y)
			{
				for(ik=0; ik<izz; ik++)
				{
					min=q;
					for(it=q+1; it<(q+(izz-ik)); it++)
					{
						if(NUM_ITERATE[it]>NUM_ITERATE[min])
							min=it;
					}
					
					if(min!=q)
					{	
						temp = NUM_ITERATE[min];
						NUM_ITERATE[min] = NUM_ITERATE[q];
						NUM_ITERATE[q] = temp;
						
						temp = order[min];
						order[min] = order[q];
						order[q] = temp;
					}
					q++;
				}
				
				j+=iyy;
				iyy=iy;
			}
			
			k+=izz;
			izz=iz;
		}
	}
	
	
	
	g=0;
	for(i=0; i<(NUM_SUBBLOCKperTHREAD * NUM_TILEperSUBBLOCK); i++)
	{
		for(n=numnonpertile[order[i]*NUM_THREAD]; n<numnonpertile[(order[i]+1)*NUM_THREAD]; n++)
		{
			rows[g] = rows_read[n];
			columns[g] = columns_read[n];
			values[g] = values_read[n];
			//printf("i=%d, order[%d]=%d, rows[%d]=%d, columns[%d]=%d, values[%d]=%lf\n", i, i, order[i], g, rows[g], g, columns[g], g, values[g]);
			g++;
		}
		for(n=0; n<NUM_THREAD; n++)
		{
			numnonpertile_perm[i*NUM_THREAD+n+1] = numnonpertile_perm[i*NUM_THREAD+n] + numnonpertile[order[i]*NUM_THREAD+n+1] - numnonpertile[order[i]*NUM_THREAD+n];
			//printf("numnonpertile_perm[%d]=%d\n", i*NUM_THREAD+n+1, numnonpertile_perm[i*NUM_THREAD+n+1]);
		}
	}
	/************************************************************************************************/
	
	free(rows_read);
	free(columns_read);
	free(values_read);
	
	
	
	
	
	
	
	//format reversing
	/************************************************************************************************/
	
	long long num_c;
	num_c = (long long)num_column * (long long)NUM_SUBBLOCK;
	printf("num_c = %lld\n", num_c);
	
	num_c = 116256000;
	
	int *Cp;
	cudaHostAlloc((int**)&Cp, (num_c+1) * sizeof(int), cudaHostAllocDefault);
	int *Ci;
	cudaHostAlloc((int**)&Ci, num_c * sizeof(int), cudaHostAllocDefault);
	int *PtoNUM_NECOlperTILE;
	cudaHostAlloc((int**)&PtoNUM_NECOlperTILE, (NUM_TILE+1) * sizeof(int), cudaHostAllocDefault);
	
	printf("******************\n");
	int maxl=0;
	Cp[0]=0;
	PtoNUM_NECOlperTILE[0]=0;
	j=0;
	
	int *num_nonzeros=(int *) malloc(num_column * sizeof(int));
	int sum=0;
	
	for(b=0; b<NUM_TILE; b++)
	{
		for(i=0; i<num_column; i++)
			num_nonzeros[i]=0;
		for(i=numnonpertile_perm[b];i<numnonpertile_perm[b+1];i++)
		{
			num_nonzeros[columns[i]]++;
		}
		for(k=0; k<num_column; k++)
		{
			if(num_nonzeros[k]!=0)
			{
				sum+=num_nonzeros[k];
				Cp[j+1]=sum;
				if(maxl < (Cp[j+1]-Cp[j]))
				{
					maxl=Cp[j+1]-Cp[j];
				}
				Ci[j] = k;
				j++;
			}
		}
		PtoNUM_NECOlperTILE[b+1] = j;
	}
	int NUM_NECOL;
	NUM_NECOL=j;
	printf("NUM_NECOL = %d\n", NUM_NECOL);
	printf("maxl: %d\n", maxl);
	//free(num_nonzeros);
	/************************************************************************************************/
	
	
	
	
	
	
	
	

	
	
	
	
	
	double *B, *C, *D, *D_host;
	cudaHostAlloc((double**)&B, y*r * sizeof(double), cudaHostAllocDefault);
	cudaHostAlloc((double**)&C, z*r * sizeof(double), cudaHostAllocDefault);
	cudaHostAlloc((double**)&D, ix*NUM_THREAD*r * sizeof(double), cudaHostAllocDefault);
	cudaHostAlloc((double**)&D_host, ix*NUM_THREAD*NUM_SUBBLOCKperTHREAD*r * sizeof(double), cudaHostAllocDefault);

	
	for(i=0; i<(y*r); i++)
		B[i]=1.0;
	for(i=0; i<(z*r); i++)
		C[i]=1.0;
	for(i=0; i<(ix*NUM_THREAD*r); i++)
		D[i]=0.0;
	for(i=0; i<(ix*NUM_THREAD*NUM_SUBBLOCKperTHREAD*r); i++)
		D_host[i]=0.0;
		
	printf("******************\n");
	
	
		
	
	

	
	int ir=0;
	ir = r/num_gpus;
	
	
	
	
		
	omp_set_num_threads(num_gpus);
	#pragma omp parallel
    {
	
		unsigned int cpu_thread_id = omp_get_thread_num();
		unsigned int num_cpu_threads = omp_get_num_threads();
		
		// set and check the CUDA device for this CPU thread
		int gpu_id = -1;
		cudaSetDevice(cpu_thread_id % num_gpus);        // "% num_gpus" allows more CPU threads than GPU devices
		cudaGetDevice(&gpu_id);

		printf("CPU thread %d (of %d) uses CUDA device %d\n", cpu_thread_id, num_cpu_threads, gpu_id);
		
		
		cudaStream_t stream0, stream1;
		cudaStreamCreate(&stream0);
		cudaStreamCreate(&stream1);
		
		
		int num_grid, num_block;
		if(NUM_THREAD<=512)
		{
			num_block=1;
			num_grid=NUM_THREAD;
		}
		else
		{
			num_block=NUM_THREAD/512;
			num_grid=512;
		}
		dim3 block(num_block);
		dim3 grid(num_grid);
		int* PtoNUM_NECOlperTILE_device1, * PtoNUM_NECOlperTILE_device2;
		int* Cp_device1, * Cp_device2;
		int* Ci_device1, * Ci_device2;
		int* rows_device1, * rows_device2;
		double* values_device1, * values_device2;
		double* B_device;
		double* C_device;
		double* D_device;
		

		cudaMalloc((int**)&PtoNUM_NECOlperTILE_device1, (NUM_THREAD + 1) * sizeof(int));
		cudaMalloc((int**)&PtoNUM_NECOlperTILE_device2, (NUM_THREAD + 1) * sizeof(int));
		cudaMalloc((int**)&Cp_device1, (NUM_NECOL + 1) * sizeof(int));
		cudaMalloc((int**)&Cp_device2, (NUM_NECOL + 1) * sizeof(int));
		cudaMalloc((int**)&Ci_device1, NUM_NECOL * sizeof(int));
		cudaMalloc((int**)&Ci_device2, NUM_NECOL * sizeof(int));
		cudaMalloc((int**)&rows_device1, nnz * sizeof(int));
		cudaMalloc((int**)&rows_device2, nnz * sizeof(int));
		cudaMalloc((double**)&values_device1, nnz * sizeof(double));
		cudaMalloc((double**)&values_device2, nnz * sizeof(double));
		cudaMalloc((double**)&B_device, (y*ir) * sizeof(double));
		cudaMalloc((double**)&C_device, z*ir * sizeof(double));
		cudaMalloc((double**)&D_device, ix*NUM_THREAD*ir * sizeof(double));
		
		
		double *subD_host = D_host + (ix*NUM_THREAD*NUM_SUBBLOCKperTHREAD * ir*gpu_id);
		
		
		q=0;
		double start_time, end_time;
		start_time = omp_get_wtime();
		//float milliseconds = 0;
		//cudaEvent_t start, stop;
		//cudaEventCreate(&start);
		//cudaEventCreate(&stop);
		//cudaEventRecord(start);
		for(ib=0; ib<NUM_SUBBLOCKperTHREAD; ib++)
		{
			cudaStreamCreate(&stream0);
			cudaStreamCreate(&stream1);
		
			cudaMemcpyAsync(D_device, D, ix*NUM_THREAD*ir * sizeof(double), cudaMemcpyHostToDevice, stream0);
			
			k=0;
			if((z%iz)==0)
				izz = iz;
			else
				izz = z%iz;
			while(k<z)				
			{
				cudaMemcpyAsync(C_device, C+( (k*ir)+(z*ir*gpu_id) ), (izz*ir) * sizeof(double), cudaMemcpyHostToDevice, stream0);
				
				j=0;
				if((y%iy)==0)
					iyy = iy;
				else
					iyy = y%iy;
				while(j<y)
				{
					cudaMemcpyAsync(B_device, B+( (j*ir)+(y*ir*gpu_id) ), (iyy*ir) * sizeof(double), cudaMemcpyHostToDevice, stream0);
				
					if(izz>=2)
					{
						for(i=0; (i+2)<=izz; i+=2)
						{
							printf("i=%d\n", i);
							cudaMemcpyAsync(PtoNUM_NECOlperTILE_device1, PtoNUM_NECOlperTILE + (q*NUM_THREAD), 
								((q+1)*NUM_THREAD - q*NUM_THREAD + 1) * sizeof(int), 
								cudaMemcpyHostToDevice, stream0);
							cudaMemcpyAsync(Cp_device1, Cp + PtoNUM_NECOlperTILE[q*NUM_THREAD], 
								(PtoNUM_NECOlperTILE[(q+1)*NUM_THREAD] - PtoNUM_NECOlperTILE[q*NUM_THREAD] + 1) * sizeof(int), 
								cudaMemcpyHostToDevice, stream0);
							cudaMemcpyAsync(Ci_device1, Ci + PtoNUM_NECOlperTILE[q*NUM_THREAD], 
								(PtoNUM_NECOlperTILE[(q+1)*NUM_THREAD] - PtoNUM_NECOlperTILE[q*NUM_THREAD]) * sizeof(int), 
								cudaMemcpyHostToDevice, stream0);
							cudaMemcpyAsync(rows_device1, rows + Cp[PtoNUM_NECOlperTILE[q*NUM_THREAD]], 
								(Cp[PtoNUM_NECOlperTILE[(q+1)*NUM_THREAD]] - Cp[PtoNUM_NECOlperTILE[q*NUM_THREAD]]) * sizeof(int), 
								cudaMemcpyHostToDevice, stream0);
							cudaMemcpyAsync(values_device1, values + Cp[PtoNUM_NECOlperTILE[q*NUM_THREAD]], 
								(Cp[PtoNUM_NECOlperTILE[(q+1)*NUM_THREAD]] - Cp[PtoNUM_NECOlperTILE[q*NUM_THREAD]]) * sizeof(double), 
								cudaMemcpyHostToDevice, stream0);
								
							SPMV_multi << < grid, block, 0, stream0 >> > (NUM_THREAD, NUM_TILE, num_row, j, k, ix, ir, y,
								PtoNUM_NECOlperTILE_device1, Cp_device1, Ci_device1, rows_device1, values_device1, B_device, C_device, D_device);
								
							
							cudaMemcpyAsync(PtoNUM_NECOlperTILE_device2, PtoNUM_NECOlperTILE + ((q+1)*NUM_THREAD), 
								((q+2)*NUM_THREAD - (q+1)*NUM_THREAD + 1) * sizeof(int), 
								cudaMemcpyHostToDevice, stream1);
							cudaMemcpyAsync(Cp_device2, Cp + PtoNUM_NECOlperTILE[(q+1)*NUM_THREAD], 
								(PtoNUM_NECOlperTILE[(q+2)*NUM_THREAD] - PtoNUM_NECOlperTILE[(q+1)*NUM_THREAD] + 1) * sizeof(int), 
								cudaMemcpyHostToDevice, stream1);
							cudaMemcpyAsync(Ci_device2, Ci + PtoNUM_NECOlperTILE[(q+1)*NUM_THREAD], 
								(PtoNUM_NECOlperTILE[(q+2)*NUM_THREAD] - PtoNUM_NECOlperTILE[(q+1)*NUM_THREAD]) * sizeof(int), 
								cudaMemcpyHostToDevice, stream1);
							cudaMemcpyAsync(rows_device2, rows + Cp[PtoNUM_NECOlperTILE[(q+1)*NUM_THREAD]], 
								(Cp[PtoNUM_NECOlperTILE[(q+2)*NUM_THREAD]] - Cp[PtoNUM_NECOlperTILE[(q+1)*NUM_THREAD]]) * sizeof(int), 
								cudaMemcpyHostToDevice, stream1);
							cudaMemcpyAsync(values_device2, values + Cp[PtoNUM_NECOlperTILE[(q+1)*NUM_THREAD]], 
								(Cp[PtoNUM_NECOlperTILE[(q+2)*NUM_THREAD]] - Cp[PtoNUM_NECOlperTILE[(q+1)*NUM_THREAD]]) * sizeof(double), 
								cudaMemcpyHostToDevice, stream1);
								
							q+=2;
							
							
							SPMV_multi << < grid, block, 0, stream1 >> > (NUM_THREAD, NUM_TILE, num_row, j, k, ix, ir, y,
								PtoNUM_NECOlperTILE_device2, Cp_device2, Ci_device2, rows_device2, values_device2, B_device, C_device, D_device);
							
						}
					}
					if( (izz%2)!=0 )
					{
						printf("izz=%d\n", izz);
						cudaMemcpyAsync(PtoNUM_NECOlperTILE_device1, PtoNUM_NECOlperTILE + (q*NUM_THREAD), 
							((q+1)*NUM_THREAD - q*NUM_THREAD + 1) * sizeof(int), 
							cudaMemcpyHostToDevice, stream0);
						cudaMemcpyAsync(Cp_device1, Cp + PtoNUM_NECOlperTILE[q*NUM_THREAD], 
							(PtoNUM_NECOlperTILE[(q+1)*NUM_THREAD] - PtoNUM_NECOlperTILE[q*NUM_THREAD] + 1) * sizeof(int), 
							cudaMemcpyHostToDevice, stream0);
						cudaMemcpyAsync(Ci_device1, Ci + PtoNUM_NECOlperTILE[q*NUM_THREAD], 
							(PtoNUM_NECOlperTILE[(q+1)*NUM_THREAD] - PtoNUM_NECOlperTILE[q*NUM_THREAD]) * sizeof(int), 
							cudaMemcpyHostToDevice, stream0);
						cudaMemcpyAsync(rows_device1, rows + Cp[PtoNUM_NECOlperTILE[q*NUM_THREAD]], 
							(Cp[PtoNUM_NECOlperTILE[(q+1)*NUM_THREAD]] - Cp[PtoNUM_NECOlperTILE[q*NUM_THREAD]]) * sizeof(int), 
							cudaMemcpyHostToDevice, stream0);
						cudaMemcpyAsync(values_device1, values + Cp[PtoNUM_NECOlperTILE[q*NUM_THREAD]], 
							(Cp[PtoNUM_NECOlperTILE[(q+1)*NUM_THREAD]] - Cp[PtoNUM_NECOlperTILE[q*NUM_THREAD]]) * sizeof(double), 
							cudaMemcpyHostToDevice, stream0);
							
						q+=1;
							
						SPMV_multi << < grid, block, 0, stream0 >> > (NUM_THREAD, NUM_TILE, num_row, j, k, ix, ir, y,
							PtoNUM_NECOlperTILE_device1, Cp_device1, Ci_device1, rows_device1, values_device1, B_device, C_device, D_device);
						
					}
					
					j+=iyy;
					iyy=iy;
					
				}
				
				k+=izz;
				izz=iz;
			}
			
			cudaStreamSynchronize(stream1);
			cudaMemcpyAsync(subD_host+(ib*ix*NUM_THREAD*ir), D_device, ix*NUM_THREAD*ir * sizeof(double), cudaMemcpyDeviceToHost, stream0);
			
			
			cudaStreamSynchronize(stream0);
			
			cudaStreamDestroy(stream0);
			cudaStreamDestroy(stream1);
			
			
		}
		end_time = omp_get_wtime();
		printf("Total time: %lf\n", end_time-start_time);
		//cudaEventRecord(stop);
		//cudaEventSynchronize(stop);
		//cudaEventElapsedTime(&milliseconds, start, stop);
		//printf("milliseconds = %lf\n", milliseconds);
		
		
	
	}


	return 0;
}
