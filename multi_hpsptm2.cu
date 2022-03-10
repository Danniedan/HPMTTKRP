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
                           int *PtoNUM_NECOlperTILE, int *Cp, int *Ci, int *rows, double *values, double *B, double *C, double *D)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    // printf("idx=%d\n",idx);
    if (idx < NUM_THREAD)
    {
        int i, m, n;
        int startD;
        int startn, endn, startm, endm;

        startn = PtoNUM_NECOlperTILE[idx] - PtoNUM_NECOlperTILE[0];
        endn = PtoNUM_NECOlperTILE[idx + 1] - PtoNUM_NECOlperTILE[0];
        startD = idx * ix;

        for (n = startn; n < endn; n++)
        {
            startm = Cp[n] - Cp[0];
            endm = Cp[n + 1] - Cp[0];

            for (m = startm; m < endm; m++)
            {
                for (i = 0; i < ir; i++)
                {
                    atomicAdd(&D[(startD + rows[m]) * ir + i], values[m] * C[(Ci[n] / y - k) * ir + i] * B[(Ci[n] % y - j) * ir + i]);
                }
            }
        }
    }
}

__host__ void read_reduced_matricization(char *filename, int **rows_original, int **columns_original, double **values_original, int **numnonpertile, int **NUM_ITERATE, int *NUM_SUBBLOCK_, int *NUM_TILEperSUBBLOCK_, int *NUM_SUBBLOCKperTHREAD_, int *NUM_TILE_, int y, int z, int r, int ix, int iy, int iz, int NUM_THREAD, int *num_row_, int *num_column_, int *nnz_)
{
    FILE *f;
    int num_row, num_column, nnz;
    if ((f = fopen(filename, "r")) == NULL)
        exit(1);
    fscanf(f, "%d	%d	%d", &num_row, &num_column, &nnz);
    printf("num_row=%d, num_column=%d, nnz=%d\n", num_row, num_column, nnz);
    *num_row_ = num_row;
    *num_column_ = num_column;
    *nnz_ = nnz;

    /* reseve memory for matrices */
    int row;
    int column;
    double value;
    int i, j;
    *rows_original = (int *)malloc(nnz * sizeof(int));
    *columns_original = (int *)malloc(nnz * sizeof(int));
    *values_original = (double *)malloc(nnz * sizeof(double));
    for (i = 0; i < nnz; i++)
    {
        fscanf(f, "%d	%d	%lf", &(row), &(column), &(value));
        // fscanf(f, "%d	%d", &(row), &(column));
        (*rows_original)[i] = row - 1;
        (*columns_original)[i] = column - 1;
        (*values_original)[i] = value;
    }

    if (f != stdin)
        fclose(f);

    int NUM_SUBBLOCK;
    NUM_SUBBLOCK = (num_row + ix - 1) / ix;
    printf("NUM_SUBBLOCK = %d, ", NUM_SUBBLOCK);
    *NUM_SUBBLOCK_ = NUM_SUBBLOCK;

    int NUM_TILEperSUBBLOCK;
    NUM_TILEperSUBBLOCK = ((y + iy - 1) / iy) * z;
    printf("NUM_TILEperSUBBLOCK = %d, ", NUM_TILEperSUBBLOCK);
    *NUM_TILEperSUBBLOCK_ = NUM_TILEperSUBBLOCK;

    int NUM_SUBBLOCKperTHREAD;
    NUM_SUBBLOCKperTHREAD = (NUM_SUBBLOCK + NUM_THREAD - 1) / NUM_THREAD;
    printf("NUM_SUBBLOCKperTHREAD = %d, ", NUM_SUBBLOCKperTHREAD);

    NUM_SUBBLOCKperTHREAD = NUM_SUBBLOCK / NUM_THREAD;
    for (i = 0; i < NUM_THREAD; i++)
    {
        j = NUM_SUBBLOCK * (i + 1) / NUM_THREAD - NUM_SUBBLOCK * i / NUM_THREAD;
        if (NUM_SUBBLOCKperTHREAD < j)
            NUM_SUBBLOCKperTHREAD = j;
    }
    printf("NUM_SUBBLOCKperTHREAD = %d\n", NUM_SUBBLOCKperTHREAD);
    *NUM_SUBBLOCKperTHREAD_ = NUM_SUBBLOCKperTHREAD;

    int NUM_TILE;
    NUM_TILE = NUM_SUBBLOCKperTHREAD * NUM_THREAD * NUM_TILEperSUBBLOCK;
    printf("NUM_TILE = %d\n", NUM_TILE);
    *NUM_TILE_ = NUM_TILE;

    *numnonpertile = (int *)malloc((NUM_TILE + 1) * sizeof(int));
    (*numnonpertile)[0] = 0;

    *NUM_ITERATE = (int *)malloc((NUM_SUBBLOCKperTHREAD * NUM_TILEperSUBBLOCK) * sizeof(int));
}

__host__ void tiling(int *rows_original, int *columns_original, double *values_original, int **rows_read, int **columns_read, double **values_read, int **numnonpertile, int **NUM_ITERATE, int NUM_SUBBLOCKperTHREAD, int num_row, int nnz, int y, int z, int r, int iy, int iz, int NUM_THREAD)
{
    *rows_read = (int *)malloc(nnz * sizeof(int));
    *columns_read = (int *)malloc(nnz * sizeof(int));
    *values_read = (double *)malloc(nnz * sizeof(double));

    int g = 0;

    int izz;
    if ((z % iz) == 0)
        izz = iz;
    else
        izz = z % iz;

    int iyy;
    if ((y % iy) == 0)
        iyy = iy;
    else
        iyy = y % iy;

    int ik, ib, it, inum_tile;
    inum_tile = 0;

    int q = 0;
    int i = 0, j = 0, k = 0;

    for (ib = 0; ib < NUM_SUBBLOCKperTHREAD; ib++)
    {
        k = 0;
		if ((z % iz) == 0)
			izz = iz;
		else
			izz = z % iz;

		while (k < z)
		{
			j = 0;
			if ((y % iy) == 0)
				iyy = iy;
			else
				iyy = y % iy;
				
			while (j < y)
			{
                for (ik = 0; ik < izz; ik++)
                {
                    for (it = 0; it < NUM_THREAD; it++)
                    {
                        for (i = 0; i < nnz; i++)
                        {
                            if ((columns_original[i] >= (y * (k + ik) + j)) &&
                                (columns_original[i] < (y * (k + ik) + j + iyy)))
                            {
                                if ((rows_original[i] >= (num_row * (it * NUM_SUBBLOCKperTHREAD + ib) / (NUM_THREAD * NUM_SUBBLOCKperTHREAD))) &&
                                    (rows_original[i] < (num_row * (it * NUM_SUBBLOCKperTHREAD + ib + 1) / (NUM_THREAD * NUM_SUBBLOCKperTHREAD))))
                                {
                                    (*rows_read)[g] = rows_original[i] - (num_row * (it * NUM_SUBBLOCKperTHREAD + ib) / (NUM_THREAD * NUM_SUBBLOCKperTHREAD));
                                    (*columns_read)[g] = columns_original[i];
                                    (*values_read)[g] = values_original[i];
                                    g++;
                                }
                            }
                            else
                            {
                                if (columns_original[i] >= (y * (k + ik) + j + iyy))
                                    break;
                            }
                        }
                        (*numnonpertile)[inum_tile + 1] = g;
                        inum_tile++;
                    }
                    (*NUM_ITERATE)[q] = (*numnonpertile)[(q + 1) * NUM_THREAD] - (*numnonpertile)[q * NUM_THREAD];
                    q++;
                }
				j += iyy;
				iyy = iy;
            }
            k += izz;
            izz = iz;
        }
    }

    free(rows_original);
    free(columns_original);
    free(values_original);
}

__host__ void permuting(int nnz, int NUM_THREAD, int NUM_TILE, int NUM_TILEperSUBBLOCK, int NUM_SUBBLOCKperTHREAD, int y, int z, int r, int iy, int iz, int *rows_read, int *columns_read, double *values_read, int *numnonpertile, int **NUM_ITERATE, int **numnonpertile_perm, int *rows, int *columns, double *values)
{
    *numnonpertile_perm = (int *)malloc((NUM_TILE + 1) * sizeof(int));
    (*numnonpertile_perm)[0] = 0;
    int i, j, k, q, g, n, ib, it, ik;
    int iyy, izz;
    int *order = (int *)malloc((NUM_SUBBLOCKperTHREAD * NUM_TILEperSUBBLOCK) * sizeof(int));
    for (i = 0; i < (NUM_SUBBLOCKperTHREAD * NUM_TILEperSUBBLOCK); i++)
        order[i] = i;

    int min, temp;
    q = 0;
    for (ib = 0; ib < NUM_SUBBLOCKperTHREAD; ib++)
    {
		
		k = 0;
		if ((z % iz) == 0)
			izz = iz;
		else
			izz = z % iz;
        while (k < z)
        {
			j = 0;
			if ((y % iy) == 0)
				iyy = iy;
			else
				iyy = y % iy;
			while (j < y)
            {
                for (ik = 0; ik < izz; ik++)
                {
                    min = q;
                    for (it = q + 1; it < (q + (izz - ik)); it++)
                    {
                        if ((*NUM_ITERATE)[it] > (*NUM_ITERATE)[min])
                            min = it;
                    }

                    if (min != q)
                    {
                        temp = (*NUM_ITERATE)[min];
                        (*NUM_ITERATE)[min] = (*NUM_ITERATE)[q];
                        (*NUM_ITERATE)[q] = temp;

                        temp = order[min];
                        order[min] = order[q];
                        order[q] = temp;
                    }
                    q++;
                }
				j += iyy;
				iyy = iy;
            }
			k += izz;
			izz = iz;
        }
    }

    g = 0;
    for (i = 0; i < (NUM_SUBBLOCKperTHREAD * NUM_TILEperSUBBLOCK); i++)
    {
        for (n = numnonpertile[order[i] * NUM_THREAD]; n < numnonpertile[(order[i] + 1) * NUM_THREAD]; n++)
        {
            rows[g] = rows_read[n];
            columns[g] = columns_read[n];
            values[g] = values_read[n];
            g++;
        }
        for (n = 0; n < NUM_THREAD; n++)
        {
            (*numnonpertile_perm)[i * NUM_THREAD + n + 1] = (*numnonpertile_perm)[i * NUM_THREAD + n] + numnonpertile[order[i] * NUM_THREAD + n + 1] - numnonpertile[order[i] * NUM_THREAD + n];
        }
    }

    free(rows_read);
    free(columns_read);
    free(values_read);
}

__host__ void format_reversing(int num_row, int num_column, int nnz, int NUM_SUBBLOCK, int NUM_TILE, int ix, int iy, int iz, int *NUM_NECOL_, int *numnonpertile_perm, int *columns, int *Cp, int *Ci, int *PtoNUM_NECOlperTILE)
{
    int i, j, k, b;

    int maxl = 0;
    Cp[0] = 0;
    PtoNUM_NECOlperTILE[0] = 0;
    j = 0;

    int *num_nonzeros = (int *)malloc(num_column * sizeof(int));
    int sum = 0;

    for (b = 0; b < NUM_TILE; b++)
    {
        for (i = 0; i < num_column; i++)
            num_nonzeros[i] = 0;
        for (i = numnonpertile_perm[b]; i < numnonpertile_perm[b + 1]; i++)
        {
            num_nonzeros[columns[i]]++;
        }
        for (k = 0; k < num_column; k++)
        {
            if (num_nonzeros[k] != 0)
            {
                sum += num_nonzeros[k];
                Cp[j + 1] = sum;
                if (maxl < (Cp[j + 1] - Cp[j]))
                {
                    maxl = Cp[j + 1] - Cp[j];
                    printf("Cp[%d]=%d, Cp[%d]=%d\n", j, Cp[j], j + 1, Cp[j + 1]);
                }
                Ci[j] = k;
                j++;
            }
        }
        PtoNUM_NECOlperTILE[b + 1] = j;
    }
    int NUM_NECOL;
    NUM_NECOL = j;
    *NUM_NECOL_ = NUM_NECOL;
    printf("NUM_NECOL = %d\n", NUM_NECOL);
    printf("maxl: %d\n", maxl);
    // free(num_nonzeros);
}

__host__ void hpsptm_multi(int num_row, int num_column, int nnz, int NUM_SUBBLOCK, int NUM_THREAD, int NUM_SUBBLOCKperTHREAD, int NUM_TILE, int NUM_NECOL, int y, int z, int r, int ix, int iy, int iz, int num_gpus, int *numnonpertile_perm, int *rows, int *columns, int *Cp, int *Ci, int *PtoNUM_NECOlperTILE, double *values, double *B, double *C, double *D, double *D_host)
{
    int i = 0, j = 0, k = 0, q = 0, ib = 0;
    int iyy = 0, izz = 0;
    for (i = 0; i < (y * r); i++)
        B[i] = 1.0;
    for (i = 0; i < (z * r); i++)
        C[i] = 1.0;
    for (i = 0; i < (ix * NUM_THREAD * r); i++)
        D[i] = 0.0;
    for (i = 0; i < (ix * NUM_THREAD * NUM_SUBBLOCKperTHREAD * r); i++)
        D_host[i] = 0.0;

    printf("******************\n");

    int ir = 0;
    ir = r / num_gpus;

    omp_set_num_threads(num_gpus);
	#pragma omp parallel
    {

        unsigned int cpu_thread_id = omp_get_thread_num();
        unsigned int num_cpu_threads = omp_get_num_threads();

        // set and check the CUDA device for this CPU thread
        int gpu_id = -1;
        cudaSetDevice(cpu_thread_id % num_gpus); // "% num_gpus" allows more CPU threads than GPU devices
        cudaGetDevice(&gpu_id);

        printf("CPU thread %d (of %d) uses CUDA device %d\n", cpu_thread_id, num_cpu_threads, gpu_id);

        cudaStream_t stream0, stream1;
        cudaStreamCreate(&stream0);
        cudaStreamCreate(&stream1);

        int num_grid, num_block;
        if (NUM_THREAD <= 512)
        {
            num_block = 1;
            num_grid = NUM_THREAD;
        }
        else
        {
            num_block = NUM_THREAD / 512;
            num_grid = 512;
        }
        dim3 block(num_block);
        dim3 grid(num_grid);
        int *PtoNUM_NECOlperTILE_device1, *PtoNUM_NECOlperTILE_device2;
        int *Cp_device1, *Cp_device2;
        int *Ci_device1, *Ci_device2;
        int *rows_device1, *rows_device2;
        double *values_device1, *values_device2;
        double *B_device;
        double *C_device;
        double *D_device;

        cudaMalloc((int **)&PtoNUM_NECOlperTILE_device1, (NUM_THREAD + 1) * sizeof(int));
        cudaMalloc((int **)&PtoNUM_NECOlperTILE_device2, (NUM_THREAD + 1) * sizeof(int));
        cudaMalloc((int **)&Cp_device1, (NUM_NECOL + 1) * sizeof(int));
        cudaMalloc((int **)&Cp_device2, (NUM_NECOL + 1) * sizeof(int));
        cudaMalloc((int **)&Ci_device1, NUM_NECOL * sizeof(int));
        cudaMalloc((int **)&Ci_device2, NUM_NECOL * sizeof(int));
        cudaMalloc((int **)&rows_device1, nnz * sizeof(int));
        cudaMalloc((int **)&rows_device2, nnz * sizeof(int));
        cudaMalloc((double **)&values_device1, nnz * sizeof(double));
        cudaMalloc((double **)&values_device2, nnz * sizeof(double));
        cudaMalloc((double **)&B_device, (y * ir) * sizeof(double));
        cudaMalloc((double **)&C_device, z * ir * sizeof(double));
        cudaMalloc((double **)&D_device, ix * NUM_THREAD * ir * sizeof(double));

        double *subD_host = D_host + (ix * NUM_THREAD * NUM_SUBBLOCKperTHREAD * ir * gpu_id);

        q = 0;
        double start_time, end_time;
        start_time = omp_get_wtime();
        // float milliseconds = 0;
        // cudaEvent_t start, stop;
        // cudaEventCreate(&start);
        // cudaEventCreate(&stop);
        // cudaEventRecord(start);
        for (ib = 0; ib < NUM_SUBBLOCKperTHREAD; ib++)
        {
            cudaStreamCreate(&stream0);
            cudaStreamCreate(&stream1);

            cudaMemcpyAsync(D_device, D, ix * NUM_THREAD * ir * sizeof(double), cudaMemcpyHostToDevice, stream0);
			
			k = 0;
			if ((z % iz) == 0)
				izz = iz;
			else
				izz = z % iz;
			while (k < z)
			{
			
				cudaMemcpyAsync(C_device, C + ((k * ir) + (z * ir * gpu_id)), (izz * ir) * sizeof(double), cudaMemcpyHostToDevice, stream0);
				
				j = 0;
				if ((y % iy) == 0)
					iyy = iy;
				else
					iyy = y % iy;
				while (j < y)
				{

					cudaMemcpyAsync(B_device, B + ((j * ir) + (y * ir * gpu_id)), (iyy * ir) * sizeof(double), cudaMemcpyHostToDevice, stream0);

                    if (izz >= 2)
                    {
                        for (i = 0; (i + 2) <= izz; i += 2)
                        {
                            
                            cudaMemcpyAsync(PtoNUM_NECOlperTILE_device1, PtoNUM_NECOlperTILE + (q * NUM_THREAD),
                                            ((q + 1) * NUM_THREAD - q * NUM_THREAD + 1) * sizeof(int),
                                            cudaMemcpyHostToDevice, stream0);
							cudaMemcpyAsync(Cp_device1, Cp + PtoNUM_NECOlperTILE[q * NUM_THREAD],
                                            (PtoNUM_NECOlperTILE[(q + 1) * NUM_THREAD] - PtoNUM_NECOlperTILE[q * NUM_THREAD] + 1) * sizeof(int),
                                            cudaMemcpyHostToDevice, stream0);
                            cudaMemcpyAsync(Ci_device1, Ci + PtoNUM_NECOlperTILE[q * NUM_THREAD],
                                            (PtoNUM_NECOlperTILE[(q + 1) * NUM_THREAD] - PtoNUM_NECOlperTILE[q * NUM_THREAD]) * sizeof(int),
                                            cudaMemcpyHostToDevice, stream0);
                            cudaMemcpyAsync(rows_device1, rows + Cp[PtoNUM_NECOlperTILE[q * NUM_THREAD]],
                                            (Cp[PtoNUM_NECOlperTILE[(q + 1) * NUM_THREAD]] - Cp[PtoNUM_NECOlperTILE[q * NUM_THREAD]]) * sizeof(int),
                                            cudaMemcpyHostToDevice, stream0);
                            cudaMemcpyAsync(values_device1, values + Cp[PtoNUM_NECOlperTILE[q * NUM_THREAD]],
                                            (Cp[PtoNUM_NECOlperTILE[(q + 1) * NUM_THREAD]] - Cp[PtoNUM_NECOlperTILE[q * NUM_THREAD]]) * sizeof(double),
                                            cudaMemcpyHostToDevice, stream0);
							
							cudaStreamSynchronize(stream0);

                            SPMV_multi<<<grid, block, 0, stream0>>>(NUM_THREAD, NUM_TILE, num_row, j, k, ix, ir, y,
                                                                    PtoNUM_NECOlperTILE_device1, Cp_device1, Ci_device1, rows_device1, values_device1, B_device, C_device, D_device);

                            cudaMemcpyAsync(PtoNUM_NECOlperTILE_device2, PtoNUM_NECOlperTILE + ((q + 1) * NUM_THREAD),
                                            ((q + 2) * NUM_THREAD - (q + 1) * NUM_THREAD + 1) * sizeof(int),
                                            cudaMemcpyHostToDevice, stream1);
                            cudaMemcpyAsync(Cp_device2, Cp + PtoNUM_NECOlperTILE[(q + 1) * NUM_THREAD],
                                            (PtoNUM_NECOlperTILE[(q + 2) * NUM_THREAD] - PtoNUM_NECOlperTILE[(q + 1) * NUM_THREAD] + 1) * sizeof(int),
                                            cudaMemcpyHostToDevice, stream1);
                            cudaMemcpyAsync(Ci_device2, Ci + PtoNUM_NECOlperTILE[(q + 1) * NUM_THREAD],
                                            (PtoNUM_NECOlperTILE[(q + 2) * NUM_THREAD] - PtoNUM_NECOlperTILE[(q + 1) * NUM_THREAD]) * sizeof(int),
                                            cudaMemcpyHostToDevice, stream1);
                            cudaMemcpyAsync(rows_device2, rows + Cp[PtoNUM_NECOlperTILE[(q + 1) * NUM_THREAD]],
                                            (Cp[PtoNUM_NECOlperTILE[(q + 2) * NUM_THREAD]] - Cp[PtoNUM_NECOlperTILE[(q + 1) * NUM_THREAD]]) * sizeof(int),
                                            cudaMemcpyHostToDevice, stream1);
                            cudaMemcpyAsync(values_device2, values + Cp[PtoNUM_NECOlperTILE[(q + 1) * NUM_THREAD]],
                                            (Cp[PtoNUM_NECOlperTILE[(q + 2) * NUM_THREAD]] - Cp[PtoNUM_NECOlperTILE[(q + 1) * NUM_THREAD]]) * sizeof(double),
                                            cudaMemcpyHostToDevice, stream1);
											
							cudaStreamSynchronize(stream1);

                            SPMV_multi<<<grid, block, 0, stream1>>>(NUM_THREAD, NUM_TILE, num_row, j, k, ix, ir, y,
                                                                    PtoNUM_NECOlperTILE_device2, Cp_device2, Ci_device2, rows_device2, values_device2, B_device, C_device, D_device);

                            q += 2;

                        }
                    }
                    if ((izz % 2) != 0)
                    {
                        cudaMemcpyAsync(PtoNUM_NECOlperTILE_device1, PtoNUM_NECOlperTILE + (q * NUM_THREAD),
                                        ((q + 1) * NUM_THREAD - q * NUM_THREAD + 1) * sizeof(int),
                                        cudaMemcpyHostToDevice, stream0);
                        cudaMemcpyAsync(Cp_device1, Cp + PtoNUM_NECOlperTILE[q * NUM_THREAD],
                                        (PtoNUM_NECOlperTILE[(q + 1) * NUM_THREAD] - PtoNUM_NECOlperTILE[q * NUM_THREAD] + 1) * sizeof(int),
                                        cudaMemcpyHostToDevice, stream0);
                        cudaMemcpyAsync(Ci_device1, Ci + PtoNUM_NECOlperTILE[q * NUM_THREAD],
                                        (PtoNUM_NECOlperTILE[(q + 1) * NUM_THREAD] - PtoNUM_NECOlperTILE[q * NUM_THREAD]) * sizeof(int),
                                        cudaMemcpyHostToDevice, stream0);
                        cudaMemcpyAsync(rows_device1, rows + Cp[PtoNUM_NECOlperTILE[q * NUM_THREAD]],
                                        (Cp[PtoNUM_NECOlperTILE[(q + 1) * NUM_THREAD]] - Cp[PtoNUM_NECOlperTILE[q * NUM_THREAD]]) * sizeof(int),
                                        cudaMemcpyHostToDevice, stream0);
                        cudaMemcpyAsync(values_device1, values + Cp[PtoNUM_NECOlperTILE[q * NUM_THREAD]],
                                        (Cp[PtoNUM_NECOlperTILE[(q + 1) * NUM_THREAD]] - Cp[PtoNUM_NECOlperTILE[q * NUM_THREAD]]) * sizeof(double),
                                        cudaMemcpyHostToDevice, stream0);

						cudaStreamSynchronize(stream0);

                        SPMV_multi<<<grid, block, 0, stream0>>>(NUM_THREAD, NUM_TILE, num_row, j, k, ix, ir, y,
                                                                PtoNUM_NECOlperTILE_device1, Cp_device1, Ci_device1, rows_device1, values_device1, B_device, C_device, D_device);
                    
                        q += 1;
					}
					
					j += iyy;
					iyy = iy;

                }
				
				k += izz;
				izz = iz;
            }

            cudaStreamSynchronize(stream1);
            cudaMemcpyAsync(subD_host + (ib * ix * NUM_THREAD * ir), D_device, ix * NUM_THREAD * ir * sizeof(double), cudaMemcpyDeviceToHost, stream0);

            cudaStreamSynchronize(stream0);

            cudaStreamDestroy(stream0);
            cudaStreamDestroy(stream1);
        }
        end_time = omp_get_wtime();
        printf("Total time: %lf\n", end_time - start_time);
        // cudaEventRecord(stop);
        // cudaEventSynchronize(stop);
        // cudaEventElapsedTime(&milliseconds, start, stop);
        // printf("milliseconds = %lf\n", milliseconds);

    }
}

int main(int argc, char **argv)
{

    // float milliseconds = 0;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // read_reduced_matricization
    char filename[256];
    printf("Please Input filename (256 characters or less):\n");
    scanf("%s", filename);
    int y, z, r;
    printf("Please Input Y, Z, and R:\n");
    scanf("%d %d %d", &y, &z, &r);
    int ix, iy, iz, NUM_THREAD, num_gpus;
    printf("Please Input ix, iy, iz, Number_of_threads, Number_of_gpus:\n");
    scanf("%d %d %d %d %d", &ix, &iy, &iz, &NUM_THREAD, &num_gpus);
    int NUM_SUBBLOCK, NUM_TILEperSUBBLOCK, NUM_SUBBLOCKperTHREAD, NUM_TILE;

    int num_row, num_column, nnz;
    int *rows_original, *columns_original, *numnonpertile, *NUM_ITERATE, *numnonpertile_perm;
    double *values_original;
    read_reduced_matricization(filename, &rows_original, &columns_original, &values_original, &numnonpertile, &NUM_ITERATE, &NUM_SUBBLOCK, &NUM_TILEperSUBBLOCK, &NUM_SUBBLOCKperTHREAD, &NUM_TILE, y, z, r, ix, iy, iz, NUM_THREAD, &num_row, &num_column, &nnz);

    // tiling
	printf("Tiling...\n");
    int *rows_read, *columns_read;
    double *values_read;
    tiling(rows_original, columns_original, values_original, &rows_read, &columns_read, &values_read, &numnonpertile, &NUM_ITERATE, NUM_SUBBLOCKperTHREAD, num_row, nnz, y, z, r, iy, iz, NUM_THREAD);

    // permuting
	printf("Permuting...\n");
    int *rows;
    int *columns = (int *)malloc(nnz * sizeof(int));
    double *values;
    cudaHostAlloc((int **)&rows, nnz * sizeof(int), cudaHostAllocDefault);
    cudaHostAlloc((double **)&values, nnz * sizeof(double), cudaHostAllocDefault);
    permuting(nnz, NUM_THREAD, NUM_TILE, NUM_TILEperSUBBLOCK, NUM_SUBBLOCKperTHREAD, y, z, r, iy, iz, rows_read, columns_read, values_read, numnonpertile, &NUM_ITERATE, &numnonpertile_perm, rows, columns, values);

    // format_reversing
	printf("Format_reversing...\n");
    int NUM_NECOL;
    long long num_c;
    num_c = (long long)num_column * (long long)NUM_SUBBLOCK;
    printf("num_c = %lld\n", num_c);
    num_c = 116256000;
    int *Cp;
    cudaHostAlloc((int **)&Cp, (num_c + 1) * sizeof(int), cudaHostAllocDefault);
    int *Ci;
    cudaHostAlloc((int **)&Ci, num_c * sizeof(int), cudaHostAllocDefault);
    int *PtoNUM_NECOlperTILE;
    cudaHostAlloc((int **)&PtoNUM_NECOlperTILE, (NUM_TILE + 1) * sizeof(int), cudaHostAllocDefault);
    format_reversing(num_row, num_column, nnz, NUM_SUBBLOCK, NUM_TILE, ix, iy, iz, &NUM_NECOL, numnonpertile_perm, columns, Cp, Ci, PtoNUM_NECOlperTILE);

    // hpsptm
	printf("Computing...\n");
    double *B, *C, *D, *D_host;
    cudaHostAlloc((double **)&B, y * r * sizeof(double), cudaHostAllocDefault);
    cudaHostAlloc((double **)&C, z * r * sizeof(double), cudaHostAllocDefault);
    cudaHostAlloc((double **)&D, ix * NUM_THREAD * r * sizeof(double), cudaHostAllocDefault);
    cudaHostAlloc((double **)&D_host, ix * NUM_THREAD * NUM_SUBBLOCKperTHREAD * r * sizeof(double), cudaHostAllocDefault);
    hpsptm_multi(num_row, num_column, nnz, NUM_SUBBLOCK, NUM_THREAD, NUM_SUBBLOCKperTHREAD, NUM_TILE, NUM_NECOL, y, z, r, ix, iy, iz, num_gpus, numnonpertile_perm, rows, columns, Cp, Ci, PtoNUM_NECOlperTILE, values, B, C, D, D_host);
    
	
    return 0;
}