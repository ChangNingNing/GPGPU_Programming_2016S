#include "counting.h"
#include <cstdio>
#include <cassert>
#include <thrust/scan.h>
#include <thrust/transform.h>
#include <thrust/functional.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>

__device__ __host__ int CeilDiv(int a, int b) { return (a-1)/b + 1; }
__device__ __host__ int CeilAlign(int a, int b) { return CeilDiv(a, b) * b; }

struct is_char
{
	__host__ __device__
	int operator()(char x)
	{
		if (x != ' ' && x != '\n')
			return 1;
		return 0;
	}
};

void CountPosition1(const char *text, int *pos, int text_size)
{
	struct is_char op;
	thrust::transform(thrust::device, text, text+text_size, pos, op);
	thrust::inclusive_scan_by_key(thrust::device, pos, pos+text_size, pos, pos);
}

#define ThreadSize 1024

__global__ void myTransform(const char *text, int *pos, int n){
	int x = blockIdx.x, y = threadIdx.x;
	int index = x * blockDim.x + y;

	if (index < n){
		if (text[index] != ' ' && text[index] != '\n')
			pos[index] = 1;
		else
			pos[index] = 0;
	}
}

__global__ void myBuildTree(const char *text, int *pos, int *BIT, int n){
	int x = blockIdx.x;
	int y = (blockIdx.y == 1)? threadIdx.x + ThreadSize/2: threadIdx.x;
	int index = x * blockDim.x + y;
	int left = (blockIdx.y == 1)? x*blockDim.x + ThreadSize/2: x*blockDim.x;

	if (index < n){
		int *row_pre = (int *)(BIT);
		int *row_cur = (int *)(BIT + n);
		int base = 1;
		for (int i=1; i<10; i++){
			int offset = (1 << (i-1));
			int tmp = index - offset;
			if (tmp >= left){
				if (row_pre[tmp] != 0 && row_pre[index] != 0){
					row_cur[index] = row_pre[index] + row_pre[tmp];
					base = i + 1;
				}
				else
					row_cur[index] = 0;
			}
			else{
				row_cur[index] = row_pre[index];
				base = i + 1;
			}
			__syncthreads();

			row_pre = row_cur;
			row_cur += n;
		}
		__syncthreads();

		int offset = index;
		row_cur = (int *)(BIT + (base-1)*n);
		for (int i=base-1; i>=0 && offset>=left; i--){
			offset -= row_cur[offset];
			row_cur -= n;
		}

		if (index >= left + ThreadSize/2 || index < ThreadSize / 2)
			pos[index] = index - offset;
	}
}

__global__ void mySet(int *pos, int *BIT, int n){
	int x = blockIdx.x, y = threadIdx.x;
	int index = x * blockDim.x + y;

	if (index < n){
		int base = 0;
		int *row = (int *)(BIT);
		for (; base < 10; base++){
			if (row[index] == 0)
				break;
			row += n;
		}

		int offset = index;
		row -= n;
		for (int i=base-1; i>=0 && offset>=0; i--){
			offset -= row[offset];
			row -= n;
		}

		pos[index] = index - offset;
	}
}

void CountPosition2(const char *text, int *pos, int text_size)
{
	int *BIT;
	size_t pitch;
	dim3 grid(CeilDiv(text_size, ThreadSize), 1), block(ThreadSize, 1);
	dim3 grid2(CeilDiv(text_size, ThreadSize), 2);
	cudaMallocPitch(&BIT, &pitch, sizeof(int)*text_size, 10);

	myTransform<<< grid, block>>>(text, BIT, text_size);
	myBuildTree<<< grid2, block>>>(text, pos, BIT, text_size);
//	mySet<<< grid, block>>>(pos, BIT, text_size);

	cudaFree(BIT);
}