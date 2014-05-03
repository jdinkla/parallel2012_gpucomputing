/*
 * Copyright (c) 2012 by Jörn Dinkla, www.dinkla.com, All rights reserved.
 */

#include "smooth2.h"

__device__ inline void add(int4& a, const uchar4& b) {
	a.x += b.x;
	a.y += b.y;
	a.z += b.z;
}

__global__
void smooth_kernel2(const uchar4* d_input, uchar4* d_output, const CExtent extent, const int windowSize) {
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (extent.inBounds(x, y)) {
		int4 a = make_int4(0, 0, 0, 0);
		int count = 0;

		for (int dy = -windowSize; dy <= windowSize; dy ++) {
			for (int dx = -windowSize; dx <= windowSize; dx ++) {
				const int nx = x + dx;
				const int ny = y + dy;
				if (extent.inBoundsStrict(nx, ny)) {
					add(a, d_input[extent.index(nx, ny)]);
					count++;
				}
			}
		}

		d_output[extent.index(x, y)] = make_uchar4(a.x / count, a.y / count, a.z / count, 255);
	}
}


void smooth2(const CExecConfig& config, const uchar4* d_input, uchar4* d_output, const CExtent& extent, const int windowSize) {
	smooth_kernel2<<<config.grid,config.threads>>>(d_input, d_output, extent, windowSize);
}

