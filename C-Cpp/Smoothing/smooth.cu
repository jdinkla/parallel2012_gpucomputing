/*
 * Copyright (c) 2012 by Jörn Dinkla, www.dinkla.com, All rights reserved.
 */

#include "smooth.h"
#include "CAccum.h"

__global__
void smooth_kernel(const CExtent extent, const uchar4* d_input, uchar4* d_output, const int windowSize) {
	
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (extent.inBounds(x, y)) {
		CAccum acc;

		for (int dy = -windowSize; dy <= windowSize; dy ++) {
			for (int dx = -windowSize; dx <= windowSize; dx ++) {
				const int nx = x + dx;
				const int ny = y + dy;
				if (extent.inBoundsStrict(nx, ny)) {
					acc.add(d_input[extent.index(nx, ny)]);
				}
			}
		}

		d_output[extent.index(x, y)] = acc.avg();
	}
}

void smooth(const CExecConfig& config, const CExtent& extent, const uchar4* d_input, uchar4* d_output, const int windowSize) {
	smooth_kernel<<<config.grid,config.threads>>>(extent, d_input, d_output, windowSize);
} 