/*
 * Copyright (c) 2012 by Jörn Dinkla, www.dinkla.com, All rights reserved.
 */
#include "smooth1.h"

__device__ inline void add(int4& a, const uchar4& b) {
	a.x += b.x;
	a.y += b.y;
	a.z += b.z;
}

__global__
void smooth_kernel1(const uchar4* d_input, uchar4* d_output, const int width, const int height, const int windowSize) {
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x < width && y < height) {
		int4 a = make_int4(0, 0, 0, 0);
		int count = 0;

		for (int dy = -windowSize; dy <= windowSize; dy ++) {
			for (int dx = -windowSize; dx <= windowSize; dx ++) {
				const int nx = x + dx;
				const int ny = y + dy;
				if (0 <= nx && nx < width && 0 <= ny && ny < height) {
					add(a, d_input[ny * width + nx]);
					count++;
				}
			}
		}

		d_output[y * width + x] = make_uchar4(a.x / count, a.y / count, a.z / count, 255);
	}
}


void smooth1(const CExecConfig& config, const uchar4* d_input, uchar4* d_output, const int width, const int height, const int windowSize) {
	smooth_kernel1<<<config.grid,config.threads>>>(d_input, d_output, width, height, windowSize);
}

