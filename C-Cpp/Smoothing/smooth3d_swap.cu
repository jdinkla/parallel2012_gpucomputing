/*
 * Copyright (c) 2012 by Jörn Dinkla, www.dinkla.com, All rights reserved.
 */

#include "smooth3.h"
#include "CAccum.h"

/*
__global__
void smooth3d_swap_kernel(
	const uchar4* d_input, const CExtent extentInput,
	uchar4* d_output, const CExtent extentOutput, 
	const int windowSize, const int3 offset, const CExtent realExtent) {
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = ..., const int z = ...
	if (extentOutput.inBounds(x, y, z)) {
		CAccum acc;
		for (int dz = -windowSize; dz <= windowSize; dz ++) {
			... dy, dx ...
			int nx=x+dx; int ny=y+dy; int nz=z+dz;
			if (realExtent.inBoundsStrict(nx, ny, nz)) {
				acc.add(d_input[extentInput.index(nx, ny, nz)]);
			}
		}
		d_output[extentOutput.index(x, y, z)] = acc.avg();
	}
}
*/

__global__
void smooth3d_swap_kernel(const uchar4* d_input, 
						  const CExtent extentInput,
						  uchar4* d_output, 
						  const CExtent extentOutput, 
						  const int windowSize,
	  				      const int3 offset,
						  const CExtent realExtent) {
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;
	const int z = blockIdx.z * blockDim.z + threadIdx.z;
	if (extentOutput.inBounds(x, y, z)) {
		CAccum acc;
		for (int dz = -windowSize; dz <= windowSize; dz ++) {
			for (int dy = -windowSize; dy <= windowSize; dy ++) {
				for (int dx = -windowSize; dx <= windowSize; dx ++) {
					const int nx = x + dx + offset.x;
					const int ny = y + dy + offset.y;
					const int nz = z + dz + offset.z;
					if (realExtent.inBoundsStrict(nx, ny, nz)) {
						acc.add(d_input[extentInput.index(nx, ny, nz)]);
					}
				}
			}
		}
		d_output[extentOutput.index(x, y, z)] = acc.avg();
		//d_output[extentOutput.index(x, y, z)] = make_uchar4(acc.count, offset.z, 0, 99);
	}
}

//d_output[extentOutput.index(x, y, z)] = make_uchar4(acc.count, offset.z, 0, 99);
//d_output[extentOutput.index(x, y, z)] = acc.avg();
//d_output[extentOutput.index(x, y, z)] = make_uchar4(241, 0, 0, 0);

void smooth3d_swap(const CExecConfig& config, 
				   const CDeviceBuffer<uchar4>& input, 
				   const CDeviceBuffer<uchar4>& output, 
				   const int windowSize,
				   const int3& offset,
				   const CExtent& realExtent) {
	smooth3d_swap_kernel<<<config.grid,config.threads,0,config.stream>>>(input.getPtr(), input, output.getPtr(), output, windowSize, offset, realExtent);
}
