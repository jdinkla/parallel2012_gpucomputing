/*
 * Copyright (c) 2012 by Jörn Dinkla, www.dinkla.com, All rights reserved.
 */

#include "CExtent.h"
#include "CAccum.h"

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

int main1(int argc, char** argv) {

	CExtent extent(1024, 1024, 1); int windowSize = 1;
	int size = extent.getNumberOfElements() * sizeof(uchar4);

	uchar4 *h_input, *h_output, *d_input, *d_output;
	cudaMallocHost((void**) &h_input, size);
	initialize(h_input, extent);	
	cudaMallocHost((void**) &h_output, size);
	cudaMalloc((void**) &d_input, size);
	cudaMalloc((void**) &d_output, size);

	cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice);	

	dim3 threads(128, 1, 1);
	dim3 grid((extent.width+threads.x-1)/threads.x, 
		      extent.height, extent.depth);
	smooth<<<grid,threads>>>(extent, d_input, d_output, windowSize);

	cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost);

	cudaFree(d_input); cudaFree(d_output); 
	cudaFreeHost(h_input); cudaFreeHost(h_output);

	return 1;
}



int xmain1(int argc, char** argv) {

	CExtent extent(1024, 1024, 1); int windowSize = 1;
	int size = extent.getNumberOfElements() * sizeof(uchar4);

	uchar4 *h_input, *h_output, *d_input, *d_output;
	cudaMallocHost((void**) &h_input, size);
	initialize(h_input, extent);	
	cudaMallocHost((void**) &h_output, size);
	cudaMalloc((void**) &d_input, size);
	cudaMalloc((void**) &d_output, size);

	cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice);	

	dim3 threads(128, 2, 1);
	dim3 grid(8, 512, 1024);
	smooth<<<grid,threads>>>(extent, d_input, d_output, windowSize);

	cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost);

	cudaFree(d_input); cudaFree(d_output); 
	cudaFreeHost(h_input); cudaFreeHost(h_output);

	return 1;
}