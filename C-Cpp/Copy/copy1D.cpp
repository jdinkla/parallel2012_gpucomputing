/*
 * Copyright (c) 2012 by Jörn Dinkla, www.dinkla.com, All rights reserved.
 */

#include <cuda_runtime_api.h>
#include "copy1D.h"

void copy1D(const int length) {

	// Host-Daten
	uchar4 *h_input, *h_output;
	cudaMallocHost((void**) &h_input, length);
	cudaMallocHost((void**) &h_output, length);

	// Device-Daten
	uchar4 *d_input, *d_output;
	cudaMalloc((void**) &d_input, length);
	cudaMalloc((void**) &d_output, length);

	// Kopiere hin, H2D
	cudaMemcpy(d_input, h_input, length, cudaMemcpyHostToDevice);

	// Rufe Kernel
	copy1D(d_input, d_output, length);

	// Kopiere zurück, D2H
	cudaMemcpy(h_output, d_output, length, cudaMemcpyDeviceToHost);

	// Freigeben
	cudaFree(d_input);
	cudaFree(d_output);
	cudaFreeHost(h_input);
	cudaFreeHost(h_output);

}

void copy1D_h(const uchar4 *h_input,
			  uchar4 *h_output, 
			  const int length) {
	// Device-Daten
	uchar4 *d_input, *d_output;
	cudaMalloc((void**) &d_input, length);
	cudaMalloc((void**) &d_output, length);
	// Kopiere hin, H2D
	cudaMemcpy(d_input, h_input, length, cudaMemcpyHostToDevice);
	// Rufe Kernel
	copy1D(d_input, d_output, length);
	cudaDeviceSynchronize();
	// Kopiere zurück, D2H
	cudaMemcpy(h_output, d_output, length, cudaMemcpyDeviceToHost);
	// Freigeben
	cudaFree(d_input); cudaFree(d_output);
}

