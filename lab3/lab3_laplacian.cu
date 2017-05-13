#include "lab3.h"
#include <cstdio>

__device__ __host__ int CeilDiv(int a, int b) { return (a-1)/b + 1; }
__device__ __host__ int CeilAlign(int a, int b) { return CeilDiv(a, b) * b; }

__global__ void SimpleClone(
	const float *background,
	const float *target,
	const float *mask,
	float *output,
	const int wb, const int hb, const int wt, const int ht,
	const int oy, const int ox
)
{
	const int yt = blockIdx.y * blockDim.y + threadIdx.y;
	const int xt = blockIdx.x * blockDim.x + threadIdx.x;
	const int curt = wt*yt+xt;
	if (yt < ht and xt < wt and mask[curt] > 127.0f) {
		const int yb = oy+yt, xb = ox+xt;
		const int curb = wb*yb+xb;
		if (0 <= yb and yb < hb and 0 <= xb and xb < wb) {
			output[curb*3+0] = target[curt*3+0];
			output[curb*3+1] = target[curt*3+1];
			output[curb*3+2] = target[curt*3+2];
		}
	}
}

__global__ void CalculateFixed(
	const float *background,
	const float *target,
	const float *mask,
	float *fixed,
	const int wb, const int hb, const int wt, const int ht,
	const int oy, const int ox
){
	const int dir[12][2] = {{0, -2},
							{-1, -1}, {0, -1}, {1, -1},
							{-2, 0}, {-1, 0}, {1, 0}, {2, 0},
							{-1, 1}, {0, 1}, {1, 1},
							{0, 2}};
	const int coef[12] = {	1,
							1, 2, 1,
							1, 2, 2, 1,
							1, 2, 1,
							1};

	const int yt = blockIdx.y * blockDim.y + threadIdx.y;
	const int xt = blockIdx.x * blockDim.x + threadIdx.x;
	const int curt = wt * yt + xt;
	if (yt < ht && xt < wt){
		if (mask[curt] > 127.0f){
			float sum[3] = {0}, bsum[3] = {0};
			int num = 0, bnum = 16;
			for (int i=0; i<12; i++){
				int dxt = xt + dir[i][0];
				int dyt = yt + dir[i][1];
				int dcurt = wt * dyt + dxt;
				int dxb = ox + dxt;
				int dyb = oy + dyt;
				int dcurb = wb * dyb + dxb;
				if (dxt >= 0 && dxt < wt && dyt >= 0 && dyt < ht){
					sum[0] += target[dcurt*3 + 0] * coef[i];
					sum[1] += target[dcurt*3 + 1] * coef[i];
					sum[2] += target[dcurt*3 + 2] * coef[i];
					num += coef[i];

					if (mask[dcurt] < 127.0f &&
						dxb >= 0 && dxb < wb && dyb >= 0 && dyb < hb){
						bsum[0] += background[dcurb*3 + 0] * coef[i];
						bsum[1] += background[dcurb*3 + 1] * coef[i];
						bsum[2] += background[dcurb*3 + 2] * coef[i];	
					}
				}
				else if (dxb >= 0 && dxb < wb && dyb >= 0 && dyb < hb){
					bsum[0] += background[dcurb*3 + 0] * coef[i];
					bsum[1] += background[dcurb*3 + 1] * coef[i];
					bsum[2] += background[dcurb*3 + 2] * coef[i];	
				}
			}
			fixed[curt*3+0] = target[curt*3+0] - sum[0] / num + bsum[0] / bnum;
			fixed[curt*3+1] = target[curt*3+1] - sum[1] / num + bsum[1] / bnum;
			fixed[curt*3+2] = target[curt*3+2] - sum[2] / num + bsum[2] / bnum;
		}
		else {
			fixed[curt*3+0] = 0;
			fixed[curt*3+1] = 0;
			fixed[curt*3+2] = 0;
		}
	}
}

__global__ void PossionImageCloningIteration(
	const float *fixed,
	const float *mask,
	float *input,
	float *output,
	const int wt, const int ht,
	const int round
){
	const int dir[12][2] = {{0, -2},
							{-1, -1}, {0, -1}, {1, -1},
							{-2, 0}, {-1, 0}, {1, 0}, {2, 0},
							{-1, 1}, {0, 1}, {1, 1},
							{0, 2}};
	const int coef[12] = {	1,
							1, 2, 1,
							1, 2, 2, 1,
							1, 2, 1,
							1};

	const int yt = blockIdx.y * blockDim.y + threadIdx.y;
	const int xt = blockIdx.x * blockDim.x + threadIdx.x;
	const int curt = wt * yt + xt;
	if (yt < ht && xt < wt && mask[curt] > 127.0f){
		float sum[3] = {0};
		int num = 16;
		for (int i=0; i<12; i++){
			int dxt = xt + dir[i][0];
			int dyt = yt + dir[i][1];
			if (dxt >= 0 && dxt < wt && dyt >= 0 && dyt < ht){
				int dcurt = wt * dyt + dxt;
				if (mask[dcurt] > 127.0f){
					sum[0] += input[dcurt*3+0] * coef[i];
					sum[1] += input[dcurt*3+1] * coef[i];
					sum[2] += input[dcurt*3+2] * coef[i];
				}
			}
		}
		output[curt*3+0] = fixed[curt*3+0] + sum[0] / num;
		output[curt*3+1] = fixed[curt*3+1] + sum[1] / num;
		output[curt*3+2] = fixed[curt*3+2] + sum[2] / num;
	}
}

void PoissonImageCloning(
	const float *background,
	const float *target,
	const float *mask,
	float *output,
	const int wb, const int hb, const int wt, const int ht,
	const int oy, const int ox
)
{
	// set up
	float *fixed, *buf1, *buf2;
	cudaMalloc(&fixed, 3*wt*ht*sizeof(float));
	cudaMalloc(&buf1, 3*wt*ht*sizeof(float));
	cudaMalloc(&buf2, 3*wt*ht*sizeof(float));

	// initialize the iteration
	dim3 gdim(CeilDiv(wt, 32), CeilDiv(ht, 16)), bdim(32, 16);
	CalculateFixed<<< gdim, bdim>>>(
		background, target, mask, fixed,
		wb, hb, wt, ht, oy, ox
	);
	cudaMemcpy(buf1, target, sizeof(float)*3*wt*ht, cudaMemcpyDeviceToDevice);

	// iterate
	for (int i = 0; i < 5000; i++){
		PossionImageCloningIteration<<<gdim, bdim>>>(
			fixed, mask, buf1, buf2, wt, ht, i
		);
		PossionImageCloningIteration<<<gdim, bdim>>>(
			fixed, mask, buf2, buf1, wt, ht, i
		);
	}

	// copy the image back
	cudaMemcpy(output, background, wb*hb*sizeof(float)*3, cudaMemcpyDeviceToDevice);
	SimpleClone<<<gdim, bdim>>>(
		background, buf1, mask, output,
		wb, hb, wt, ht, oy, ox
	);
}
