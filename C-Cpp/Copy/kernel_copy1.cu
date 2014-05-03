/*
 * Copyright (c) 2012 by Jörn Dinkla, www.dinkla.com, All rights reserved.
 */

#include "kernel_copy1.h"

__global__ void kernel_copy_kernel1(const uchar4* d_input, uchar4* d_output, const int width, const int height) {
	const int x = blockIdx.x * blockDim.x + threadIdx.x;	// Identität bestimmen
	const int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x < width && y < height) {
		const int idx = y * width + x;
		const uchar4 value = d_input[idx];		// Daten holen
		d_output[idx] = value;					// Und schreiben
	}
}

void kernel_copy1(const uchar4* d_input, uchar4* d_output, const int width, const int height) {
	dim3 tb(8, 8, 1);
	dim3 gb((width + tb.x - 1) / tb.x, (height + tb.y - 1) / tb.y, 1);
	kernel_copy_kernel1<<<gb,tb>>>(d_input, d_output, width, height);
}

void kernel_copy1(const CExecConfig& config, const uchar4* d_input, uchar4* d_output, const int width, const int height) {
	kernel_copy_kernel1<<<config.grid,config.threads>>>(d_input, d_output, width, height);
}

void kernel_copy1(const CExecConfig& config, const uchar4* d_input, uchar4* d_output, const CExtent& extent) {
	kernel_copy_kernel1<<<config.grid,config.threads>>>(d_input, d_output, extent.width, extent.height);
}
