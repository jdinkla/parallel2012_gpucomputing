/*
 * Copyright (c) 2012 by Jörn Dinkla, www.dinkla.com, All rights reserved.
 */

#include "copy1D.h"

__global__ void copy1D_kernel(const uchar4* d_input, 
							  uchar4* d_output, 
							  const int length) {
	// Identität bestimmen
	const int x = blockIdx.x * blockDim.x + threadIdx.x; 

	if (x < length) {
		// Daten holen
		const uchar4 value = d_input[x];
		// Und schreiben
		d_output[x] = value;			 
	}
}

void copy1D(const uchar4* d_input, 
			uchar4* d_output, 
			const int length) {
	const dim3 threads(64, 1, 1);
	const dim3 grid((length + threads.x - 1) / threads.x, 1, 1);
	copy1D_kernel<<<grid, threads>>>(d_input, d_output, length);
}

void copy1D(const CExecConfig& config, const uchar4* d_input, uchar4* d_output, const int length) {
	copy1D_kernel<<<config.grid,config.threads>>>(d_input, d_output, length);
}

void copy1D(const CExecConfig& config, const uchar4* d_input, uchar4* d_output, const CExtent& extent) {
	copy1D_kernel<<<config.grid,config.threads>>>(d_input, d_output, extent.width * extent.height);
}

