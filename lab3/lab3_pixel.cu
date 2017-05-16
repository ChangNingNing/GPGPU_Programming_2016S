#include "lab3.h"
#include <cstdio>

__device__ __host__ int CeilDiv(int a, int b) { return (a-1)/b + 1; }
__device__ __host__ int CeilAlign(int a, int b) { return CeilDiv(a, b) * b; }

__global__ void SimpleClone(
	const float *background,
	const float *target,
	const int *mask,
	float *output,
	const int wb, const int hb, const int wt, const int ht,
	const int oy, const int ox
)
{
	const int yt = blockIdx.y * blockDim.y + threadIdx.y;
	const int xt = blockIdx.x * blockDim.x + threadIdx.x;
	const int curt = wt * yt + xt;
	if (yt < ht and xt < wt and mask[curt]) {
		const int yb = oy+yt, xb = ox+xt;
		const int curb = wb*yb+xb;
		if (0 <= yb and yb < hb and 0 <= xb and xb < wb){
			output[curb*3+0] = target[curt*3+0];
			output[curb*3+1] = target[curt*3+1];
			output[curb*3+2] = target[curt*3+2];
		}
	}
}

__global__ void CalculateFixed(
	const float *subBG,
	const float *subT,
	const int *subM,
	float *fixed,
	const int wb, const int hb, const int wt, const int ht,
	const int oy, const int ox
){
	const int dir[16][2] = {{-2, -2},				{0, -2},			{2, -2},
										{-1, -1},	{0, -1},	{1, -1},
							{-2, 0},	{-1, 0},				{1, 0},	{2, 0},
										{-1, 1},	{0, 1},		{1, 1},
							{-2, 2},				{0, 2},				{2, 2}};
	const int coef[16] = {	1,		1,		1,
								2,	2,	2,
							1,	2,		2,	1,
								2,	2,	2,
							1,		1,		1};
	const int num = 24;
	const int yt = blockIdx.y * blockDim.y + threadIdx.y;
	const int xt = blockIdx.x * blockDim.x + threadIdx.x;
	const int curt = wt * yt + xt;

	if (yt < ht && xt < wt){
		float sum[3] = {0}, bsum[3] = {0};
		for (int i=0; i<16; i++){
			int dxt = xt + dir[i][0];
			int dyt = yt + dir[i][1];
			int dcurt = wt * dyt + dxt;
			int dxb = ox + dxt;
			int dyb = oy + dyt;
			int dcurb = wb * dyb + dxb;
			if (dxt >= 0 && dxt < wt && dyt >= 0 && dyt < ht){
				sum[0] += subT[dcurt*3 + 0] * coef[i];
				sum[1] += subT[dcurt*3 + 1] * coef[i];
				sum[2] += subT[dcurt*3 + 2] * coef[i];
			}
			else {
				sum[0] += subT[curt*3 + 0] * coef[i];
				sum[1] += subT[curt*3 + 1] * coef[i];
				sum[2] += subT[curt*3 + 2] * coef[i];
			}

			if ((dxt < 0 || dxt >= wt || dyt < 0 || dyt >= ht ||
				!subM[dcurt])
				&&
				(dxb >= 0 && dxb < wb && dyb >= 0 && dyb < hb)){
				bsum[0] += subBG[dcurb*3 + 0] * coef[i];
				bsum[1] += subBG[dcurb*3 + 1] * coef[i];
				bsum[2] += subBG[dcurb*3 + 2] * coef[i];
			}
		}
		fixed[curt*3+0] = subT[curt*3+0] - sum[0] / num + bsum[0] / num;
		fixed[curt*3+1] = subT[curt*3+1] - sum[1] / num + bsum[1] / num;
		fixed[curt*3+2] = subT[curt*3+2] - sum[2] / num + bsum[2] / num;
	}
}

__global__ void PossionImageCloningIteration(
	const float *fixed,
	const int *mask,
	float *input,
	float *output,
	const int wt, const int ht
){
	const int dir[16][2] = {{-2, -2},				{0, -2},			{2, -2},
										{-1, -1},	{0, -1},	{1, -1},
							{-2, 0},	{-1, 0},				{1, 0},	{2, 0},
										{-1, 1},	{0, 1},		{1, 1},
							{-2, 2},				{0, 2},				{2, 2}};
	const int coef[16] = {	1,		1,		1,
								2,	2,	2,
							1,	2,		2,	1,
								2,	2,	2,
							1,		1,		1};
	const int num = 24;

	const int yt = blockIdx.y * blockDim.y + threadIdx.y;
	const int xt = blockIdx.x * blockDim.x + threadIdx.x;
	const int curt = wt * yt + xt;
	if (yt < ht && xt < wt){
		float sum[3] = {0};
		for (int i=0; i<16; i++){
			int dxt = xt + dir[i][0];
			int dyt = yt + dir[i][1];
			int dcurt = wt * dyt + dxt;
			if (dxt >= 0 && dxt < wt && dyt >= 0 && dyt < ht && mask[dcurt]){
				sum[0] += input[dcurt*3+0] * coef[i];
				sum[1] += input[dcurt*3+1] * coef[i];
				sum[2] += input[dcurt*3+2] * coef[i];
			}
		}
		output[curt*3+0] = fixed[curt*3+0] + sum[0] / num;
		output[curt*3+1] = fixed[curt*3+1] + sum[1] / num;
		output[curt*3+2] = fixed[curt*3+2] + sum[2] / num;
	}
}

__global__ void CalculateSampleT(
	const float *target,
	const float *mask,
	float *subT,
	int *subM,
	const int wt, const int ht,
	const int ws, const int hs,
	const int sRate
){
	const int ys = blockIdx.y * blockDim.y + threadIdx.y;
	const int xs = blockIdx.x * blockDim.x + threadIdx.x;
	const int curst = ws * ys + xs;

	if (ys < hs && xs < ws){
		const int yt = ys * sRate;
		const int xt = xs * sRate;
		int num = 0;
		float sum[3] = {0};
		int _or = 0;

		for (int i=0; i<sRate; i++){
			for (int j=0; j<sRate; j++){
				if (yt + i < ht && xt + j < wt){
					int curt = wt * (yt+i) + (xt+j);
					sum[0] += target[curt*3+0];
					sum[1] += target[curt*3+1];
					sum[2] += target[curt*3+2];
					_or |= (mask[curt] > 127.0f);
					num++;	
				}
			}
		}

		subM[curst] = _or;
		subT[curst*3+0] += sum[0] / num;
		subT[curst*3+1] += sum[1] / num;
		subT[curst*3+2] += sum[2] / num;
	}
}

__global__ void CalculateSampleB(
	const float *background,
	float *subBG,
	const int wb, const int hb,
	const int ws, const int hs,
	const int sRate
){
	const int ys = blockIdx.y * blockDim.y + threadIdx.y;
	const int xs = blockIdx.x * blockDim.x + threadIdx.x;
	const int curst = ws * ys + xs;

	if (ys < hs && xs < ws){
		const int yb = ys * sRate;
		const int xb = xs * sRate;
		int num = 0;
		float sum[3] = {0};

		for (int i=0; i<sRate; i++){
			for (int j=0; j<sRate; j++){
				if (yb + i < hb && xb + j < wb){
					int curb = wb * (yb+i) + (xb+j);
					sum[0] += background[curb*3+0];
					sum[1] += background[curb*3+1];
					sum[2] += background[curb*3+2];
					num++;	
				}
			}
		}
		subBG[curst*3+0] = sum[0] / num;
		subBG[curst*3+1] = sum[1] / num;
		subBG[curst*3+2] = sum[2] / num;
	}
}

__global__ void CalculateDiffSample(
	float *cur,
	float *pre,
	const int wts, const int hts
){
	const int yts = blockIdx.y * blockDim.y + threadIdx.y;
	const int xts = blockIdx.x * blockDim.x + threadIdx.x;
	const int curst = wts * yts + xts;

	if (yts < hts && xts < wts){
		cur[curst*3+0] -= pre[curst*3+0];
		cur[curst*3+1] -= pre[curst*3+1];
		cur[curst*3+2] -= pre[curst*3+2];
		pre[curst*3+0] = 0;
		pre[curst*3+1] = 0;
		pre[curst*3+2] = 0;
	}
}

__global__ void CalculateTransSample(
	const float *input,
	float *output,
	const int wtss, const int htss,
	const int wts, const int hts,
	const int ratio
){
	const int yts = blockIdx.y * blockDim.y + threadIdx.y;
	const int xts = blockIdx.x * blockDim.x + threadIdx.x;
	const int curst = wts * yts + xts;

	const int yt = yts * ratio, xt = xts * ratio;

	if (yts < hts && xts < wts){
		for (int i=0; i<ratio; i++){
			for (int j=0; j<ratio; j++){
				if (yt + i < htss && xt + j < wtss){
					const int curt = wtss * (yt+i) + (xt+j);
					output[curt*3+0] = input[curst*3+0];
					output[curt*3+1] = input[curst*3+1];
					output[curt*3+2] = input[curst*3+2];
				}
			}
		}
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
	cudaMalloc(&buf1,  3*wt*ht*sizeof(float));
	cudaMalloc(&buf2,  3*wt*ht*sizeof(float));

	// subsample
	float *subBG, *subT;
	int *subM;
	cudaMalloc(&subBG, 3*wb*hb*sizeof(float));
	cudaMalloc(&subT,  3*wt*ht*sizeof(float));
	cudaMalloc(&subM,  3*wt*ht*sizeof(int));

	const int sRate[2] = {2, 1};
	const int sIter[2] = {2000, 1000};
	int wts, hts, wbs, hbs, oys, oxs;

	for (int i=0; i<2; i++){
		if (i != 0){
			CalculateDiffSample<<<dim3(CeilDiv(wts, 32), CeilDiv(hts, 16)), dim3(32, 16)>>>(
				buf1, subT,
				wts, hts
			);
			CalculateTransSample<<<dim3(CeilDiv(wts, 32), CeilDiv(hts, 16)), dim3(32, 16)>>>(
				buf1, subT,
				CeilDiv(wt, sRate[i]), CeilDiv(ht, sRate[i]),
				wts, hts,
				sRate[i-1] / sRate[i]
			);
		}
		else
			cudaMemset(subT, 0, 3*wt*ht*sizeof(float));

		wts = CeilDiv(wt, sRate[i]);
		hts = CeilDiv(ht, sRate[i]);
		dim3 tsgdim(CeilDiv(wts, 32), CeilDiv(hts, 16)), tsbdim(32, 16);
		CalculateSampleT<<< tsgdim, tsbdim>>>(
			target, mask, subT, subM,
			wt, ht, wts, hts,
			sRate[i]
		);
	
		wbs = CeilDiv(wb, sRate[i]);
		hbs = CeilDiv(hb, sRate[i]);
		dim3 bsgdim(CeilDiv(wbs, 32), CeilDiv(hbs, 16)), bsbdim(32, 16);
		CalculateSampleB<<< bsgdim, bsbdim>>>(
			background, subBG,
			wb, hb, wbs, hbs,
			sRate[i]
		);
	
		// initialize the iteration
		oys = CeilDiv(oy, sRate[i]);
		oxs = CeilDiv(ox, sRate[i]);
		CalculateFixed<<< tsgdim, tsbdim>>>(
			subBG, subT, subM, fixed,
			wbs, hbs, wts, hts, oys, oxs
		);
		cudaMemcpy(buf1, subT, sizeof(float)*3*wts*hts, cudaMemcpyDeviceToDevice);

		// iterate
		for (int j = 0; j < sIter[i]; j++){
			PossionImageCloningIteration<<<tsgdim, tsbdim>>>(
				fixed, subM, buf1, buf2, wts, hts
			);
			PossionImageCloningIteration<<<tsgdim, tsbdim>>>(
				fixed, subM, buf2, buf1, wts, hts
			);
		}
	}

	// copy the image back
	cudaMemcpy(output, background, wb*hb*sizeof(float)*3, cudaMemcpyDeviceToDevice);
	dim3 gdim(CeilDiv(wt, 32), CeilDiv(ht, 16)), bdim(32, 16);
	SimpleClone<<<gdim, bdim>>>(
		background, buf1, subM, output,
		wb, hb, wt, ht, oy, ox
	);

	cudaFree(fixed);
	cudaFree(buf1);
	cudaFree(buf2);
	cudaFree(subBG);
	cudaFree(subT);
	cudaFree(subM);
}
