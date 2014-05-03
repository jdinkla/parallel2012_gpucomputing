/*
 * Copyright (c) 2012 by Jörn Dinkla, www.dinkla.com, All rights reserved.
 */

#include "smooth3_1d.h"
#include "CAccum.h"

__global__
void smooth_kernel3_1d(const uchar4* d_input, uchar4* d_output, const CExtent extent, const int windowSize) {
	
	const int x = blockIdx.x * blockDim.x + threadIdx.x;

	if (extent.inBoundsX(x)) {

		for (int y = 0; y < extent.height; y++) {
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
}

void smooth3_1d(const CExecConfig& config, const uchar4* d_input, uchar4* d_output, const CExtent& extent, const int windowSize) {
	smooth_kernel3_1d<<<config.grid,config.threads>>>(d_input, d_output, extent, windowSize);
}
