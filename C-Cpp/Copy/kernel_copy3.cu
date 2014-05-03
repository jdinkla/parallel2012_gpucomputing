/*
 * Copyright (c) 2012 by Jörn Dinkla, www.dinkla.com, All rights reserved.
 */

#include "kernel_copy3.h"

__global__
void kernel_copy_kernel3(const uchar4* d_input, uchar4* d_output, const CExtent extent) {

	// Identität bestimmen
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (extent.inBounds(x, y)) {
		const int idx = extent.index(x, y);
		const uchar4 value = d_input[idx];		// Daten holen
		d_output[idx] = value;					// Und schreiben
	}
}

void kernel_copy3(const CExecConfig& config, const uchar4* d_input, uchar4* d_output, const CExtent& extent) {
	kernel_copy_kernel3<<<config.grid,config.threads>>>(d_input, d_output, extent);
}

