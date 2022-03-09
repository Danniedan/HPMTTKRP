#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include <stdlib.h>
#include <ctype.h>
#include <math.h>


struct tensor{
    int rows;
	int columns;
	double values;
}arr[2000000];

int cmp_row(const void *a, const void *b){
    int mark1=((struct tensor *)a)->rows;
    int mark2=((struct tensor *)b)->rows;
    return mark1>mark2 ? 1:-1;
}

int cmp_column(const void *a, const void *b){
    int mark1=((struct tensor *)a)->columns;
    int mark2=((struct tensor *)b)->columns;
    return mark1>mark2 ? 1:-1;
}

void remove_row(struct tensor *arr, long nnz, int *x_reduce)
{
	int i;
	int temp;
	int count=1;
	for (i=0; i<(nnz-1); i++)
    {
		temp = arr[i].rows;
		arr[i].rows = count;
		if(arr[i+1].rows != temp)
			count++;
    }
	temp = arr[i].rows;
	arr[i].rows = count;
	(*x_reduce) = count;
}


void coo2csc(struct tensor *arr, int *Ap, long nnz, int y, int z)
{
	int i, k;
	int maxl=0;
	int sum=0;
	
	int *num_nonzeros=(int *) malloc(y*z * sizeof(int));
	memset(num_nonzeros, 0, sizeof(int)*y*z);
	
	for(i=0;i<nnz;i++)
	{
		num_nonzeros[arr[i].columns-1]++;
	}
	
	for(k=0; k<y*z; k++)
	{
		sum+=num_nonzeros[k];
		Ap[k+1]=sum;
		if(maxl < (Ap[k+1]-Ap[k]))
		{
			maxl=Ap[k+1]-Ap[k];	
		}
	}

	printf("maxl: %d\n", maxl);
	free(num_nonzeros);
}

void remove_col(struct tensor *arr, int *Ap, int *mark, long nnz, int y, int z, int *y_reduce, int *z_reduce)
{
	int i, j;
	int flag;
	int count1=0;
	int count2=0;
	
	for(i=0; i<z; i++)
	{
		flag=0;
		for(j=0; j<y; j++)
		{
			if((Ap[i*y+j+1]-Ap[i*y+j])!=0)
				flag=1;
		}
		if(flag==0)
		{
			for(j=0; j<y; j++)
			{
				mark[i*y+j] = 1;
			}
			count1++;
			//printf("mark[%d] = %d\n", i, mark[i]);
		}
	}
	
	for(i=0; i<y; i++)
	{
		flag=0;
		for(j=0; j<z; j++)
		{
			if((Ap[j*y+i+1]-Ap[j*y+i])!=0)
				flag=1;
		}
		if(flag==0)
		{
			for(j=0; j<z; j++)
			{
				mark[j*y+i] = 1;
			}
			count2++;
			//printf("mark2[%d] = %d\n", i, mark2[i]);
		}
	}
	
	(*y_reduce) = y-count2;
	(*z_reduce) = z-count1;
}


int main(int *argc,char ***argv)
{
	int i, j;
    FILE *f;
    int x, y, z;
	long nnz;
	int row, column;
	double value;	
	
	
	//reading raw dataset
	//char * filename="/root/cyd/data/test2.txt";
	char filename[256];
	printf("Please Input filename (256 characters or less):\n");
	scanf("%s", filename);
    if ((f = fopen(filename, "r")) == NULL) 
            exit(1);
	fscanf(f, "%d	%d	%d	%ld", &x, &y, &z, &nnz);
	printf("x=%d, y=%d, z=%d, nnz=%ld\n", x, y, z, nnz);
	
    for (i=0; i<nnz; i++)
    {
		fscanf(f, "%d	%d	%d	%lf", &(arr[i].rows), &(row), &(column), &(arr[i].values));
		arr[i].columns = y*(column-1)+row;
    }
	if (f !=stdin) fclose(f);
	printf("Reading\n");
	
	
	
	//removing empty rows
	qsort(arr, nnz, sizeof(struct tensor), cmp_row);
	int x_reduce;
	remove_row(arr, nnz, &x_reduce);
	printf("x_reduce = %d\n", x_reduce);
	
	
	
	//COO2CSC
	qsort(arr, nnz, sizeof(struct tensor), cmp_column);
	int *Ap=(int *) malloc((y*z+1) * sizeof(int));
	Ap[0]=0;
	coo2csc(arr, Ap, nnz, y, z);
	
	
	
	//removing uesless columns
	int y_reduce, z_reduce;
	int *mark=(int *) malloc(y*z * sizeof(int));
	for(i=0; i<(y*z); i++)
		mark[i]=0;
	remove_col(arr, Ap, mark, nnz, y, z, &y_reduce, &z_reduce);
	printf("y_reduce = %d, z_reduce = %d, numcol_reduce = %d\n", y_reduce, z_reduce, y_reduce * z_reduce);
	
	
	
	//writing to a new file
	FILE *pw;
	char filename2[256];
	printf("Please Output filename (256 characters or less):\n");
	scanf("%s", filename2);
    if((pw = fopen(filename2,"a+"))==NULL)
    {
		printf("Fail to open file2!\n");
    	exit(1);
	}
	fprintf(pw, "%d	%d	%ld\n", x_reduce, y_reduce * z_reduce, nnz);
	
	int sum=0;
	int k;
	for(i=0; i<z; i++)
	{
		for(j=0; j<y; j++)
		{
			if(mark[i*y+j] == 1)
			{
				sum++;
			}
			if(mark[i*y+j] == 0)
			{	
				for(k=Ap[i*y+j]; k<Ap[i*y+j+1]; k++)
				{
					fprintf(pw, "%d	%d	%lf\n", arr[k].rows, arr[k].columns-sum, arr[k].values);
				}
			}
		}
	}
	
	
	return 0;
}
