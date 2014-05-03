/*
 * Copyright (c) 2012 by Jörn Dinkla, www.dinkla.com, All rights reserved.
 */

#include "CExtent.h"
#include "CAccum.h"
#include "CExecConfig.h"
#include "CPinnedHostBuffer.h"
#include "CDeviceBuffer.h"
#include "CBufferPair.h"

void initialize(uchar4* h_input, const CExtent& extent) {
	for (int y=0; y<extent.height; y++) {
		for (int x=0; x<extent.width; x++) {
			h_input[extent.index(x, y)] = make_uchar4(x, y, 0, x+y);
		}
	}

}
void initialize(uchar4* h_input, const int width, const int height) {
	for (int y=0; y<height; y++) {
		for (int x=0; x<width; x++) {
			h_input[y * width + x] = make_uchar4(x, y, 0, x+y);
		}
	}
}

/*
__device__ inline void add(int4& a, const uchar4& b) {
	a.x += b.x; a.y += b.y; a.z += b.z;
}

__global__ void smooth(const uchar4* d_input, uchar4* d_output, 
			const int width, const int height, const int windowSize) {
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x < width && y < height) {
		...
		d_output[y * width + x] = make_uchar4(a.x/c, a.y/c, a.z/c, 255);
	}
}
*/

__global__ 
void smooth(const CExtent extent, 
			const uchar4* d_input, uchar4* d_output, 
			const int windowSize) {
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;
	const int z = blockIdx.z * blockDim.z + threadIdx.z;
	if (extent.inBounds(x, y, z)) {
		CAccum acc;
		for (int dz = -windowSize; dz <= windowSize; dz ++) {
			for (int dy = -windowSize; dy <= windowSize; dy ++) {
				for (int dx = -windowSize; dx <= windowSize; dx ++) {
					int nx=x+dx; int ny=y+dy; int nz=z+dz;
					if (extent.inBoundsStrict(nx, ny, nz)) {
						acc.add(d_input[extent.index(nx, ny, nz)]);
					}
				}
			}
		}
		d_output[extent.index(x, y, z)] = acc.avg();
	}
}
	
/*
int main3(int argc, char** argv) {

	CExtent extent(1024, 1024, 1); int windowSize = 1;

	CBufferPair<uchar4> input(extent); input.malloc(); 
	CBufferPair<uchar4> output(extent); output.malloc();

	initialize(input.host); 
	input.host->incVersion(); input.updateDevice();

	CExecConfig config(extent);
	smooth<<<config.grid,config.threads>>>(extent, 
		input.device->getPtr(), output.device->getPtr(), windowSize);

	output.device->incVersion(); output.updateHost();

	input.free(); output.free();

	return 0;
}
 */