/*
 * Copyright (c) 2012 by Jörn Dinkla, www.dinkla.com, All rights reserved.
 */

#include "kernel_copy2.h"

__global__
void kernel_copy_kernel2(const uchar4* d_input, uchar4* d_output, const CExtent extent) {

	// Identität bestimmen
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x < extent.width && y < extent.height) {
		// Daten holen
		const int idx = y * extent.width + x;
		const uchar4 value = d_input[idx];
		// Und schreiben
		d_output[idx] = value;
	}
}

void kernel_copy2(const CExecConfig& config, const uchar4* d_input, uchar4* d_output, const CExtent& extent) {
	kernel_copy_kernel2<<<config.grid,config.threads>>>(d_input, d_output, extent);
}

