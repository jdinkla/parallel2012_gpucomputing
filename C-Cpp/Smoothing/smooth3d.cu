/*
 * Copyright (c) 2012 by Jörn Dinkla, www.dinkla.com, All rights reserved.
 */

#include "smooth3.h"
#include "CAccum.h"

__global__
void smooth3d_kernel(const CExtent extent, const uchar4* d_input, uchar4* d_output, const int windowSize) {
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;
	const int z = blockIdx.z * blockDim.z + threadIdx.z;
	if (extent.inBounds(x, y, z)) {
		CAccum acc;
		for (int dz = -windowSize; dz <= windowSize; dz ++) {
			for (int dy = -windowSize; dy <= windowSize; dy ++) {
				for (int dx = -windowSize; dx <= windowSize; dx ++) {
					const int nx = x + dx;
					const int ny = y + dy;
					const int nz = z + dz;
					if (extent.inBoundsStrict(nx, ny, nz)) {
						acc.add(d_input[extent.index(nx, ny, nz)]);
					}
				}
			}
		}
		d_output[extent.index(x, y, z)] = acc.avg();
	}
}

void smooth3d(const CExecConfig& config, const CExtent& extent, const uchar4* d_input, uchar4* d_output, const int windowSize) {
	smooth3d_kernel<<<config.grid,config.threads>>>(extent, d_input, d_output, windowSize);
}

void smooth3d(const CExecConfig& config, const CDeviceBuffer<uchar4>& input, const CDeviceBuffer<uchar4>& output, const int windowSize) {
	smooth3d_kernel<<<config.grid,config.threads>>>(output, input.getPtr(), output.getPtr(), windowSize);
}
