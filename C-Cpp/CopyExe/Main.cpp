/*
 * Copyright (c) 2012 by Jörn Dinkla, www.dinkla.com, All rights reserved.
 */

#include <iostream>
// CUDA
#include <cuda_runtime_api.h>
#include <vector_functions.h>
// Application
#include "CExtent.h"
#include "kernel_copy3.h"

using namespace std;

void beispiel1() {

	cudaError_t err;

	const int width = 10;
	const int height = 10;
	
	CExtent extent(width, height);
	const int size = extent.getNumberOfElements() * sizeof(uchar4);

	// Init
	uchar4* h_input;
	uchar4* h_output;

	err = cudaMallocHost((void**) &h_input, size);
	if (err != cudaSuccess) {
		cerr << "ERROR: " << err << endl;
	}

	err = cudaMallocHost((void**) &h_output, size);
	if (err != cudaSuccess) {
		cerr << "ERROR: " << err << endl;
	}

	// Initialisiere Daten
	for (int y=0; y<height; y++) {
		for (int x=0; x<width; x++) {
			h_input[y * width + x] = make_uchar4(x, y, x, y);
		}
	}

	// Device
	uchar4* d_input;
	uchar4* d_output;

	err = cudaMalloc((void**) &d_input, size);
	if (err != cudaSuccess) {
		cerr << "ERROR: " << err << endl;
	}

	err = cudaMalloc((void**) &d_output, size);
	if (err != cudaSuccess) {
		cerr << "ERROR: " << err << endl;
	}

	// Copy to device
	err = cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice);
	if (err != cudaSuccess) {
		cerr << "ERROR: " << err << endl;
	}

	// Call kernel
	int windowSize = 1;
	CExecConfig config(extent);
	kernel_copy3(config, d_input, d_output, extent);

	// Copy results back
	err = cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost);
	if (err != cudaSuccess) {
		cerr << "ERROR: " << err << endl;
	}

	// Clean up
	err = cudaFree(d_input);
	if (err != cudaSuccess) {
		cerr << "ERROR: " << err << endl;
	}

	err = cudaFree(d_output);
	if (err != cudaSuccess) {
		cerr << "ERROR: " << err << endl;
	}

	err = cudaFreeHost(h_input);
	if (err != cudaSuccess) {
		cerr << "ERROR: " << err << endl;
	}

	err = cudaFreeHost(h_output);
	if (err != cudaSuccess) {
		cerr << "ERROR: " << err << endl;
	}

}


int main(int argc, char** argv) {

	beispiel1();
}
