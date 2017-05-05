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
			return x > ' ';
	}
};

void CountPosition1(const char *text, int *pos, int text_size)
{
	struct is_char op;
	thrust::transform(thrust::device, text, text+text_size, pos, op);
	thrust::inclusive_scan_by_key(thrust::device, pos, pos+text_size, pos, pos);
}

#define ThreadSize 1024

__global__ void myCudaCount(const char *text, int *pos, int n){
	int x = blockIdx.x;
	int left = (blockIdx.y == 1)? x * blockDim.x + ThreadSize/2: x * blockDim.x;
	int y = threadIdx.x;
	int index = left + y;
	int mapIndex = index - left;

	__shared__ int BIT[ThreadSize][10];

	if (index < n){
		// Transform
		BIT[mapIndex][0] = text[index] > ' ';
		__syncthreads();

		// Build tree
		int base = 1;
		int before = BIT[mapIndex][0];
		for (int i=1, offset=1; i<10; i++, offset <<= 1){
			int tmp = index - offset;
			if (tmp >= left){
				if (before != 0 && BIT[tmp - left][i-1] != 0){
					before = (BIT[mapIndex][i] = before + BIT[tmp - left][i-1]);
					base = i + 1;
				}
				else
					before = (BIT[mapIndex][i] = 0);
			}
			else{
				BIT[mapIndex][i] = before;
				base = i + 1;
			}
			__syncthreads();
		}

		// Set
		int offset = index;
		for (int i=base-1; i>=0 && offset>=left; i--)
			offset -= BIT[offset - left][i];

		if (index >= left + ThreadSize/2 || index < ThreadSize / 2)
			pos[index] = index - offset;
	}
}

void CountPosition2(const char *text, int *pos, int text_size)
{
	dim3 grid(CeilDiv(text_size, ThreadSize), 2), block(ThreadSize, 1);
	myCudaCount<<< grid, block>>>(text, pos, text_size);
}
