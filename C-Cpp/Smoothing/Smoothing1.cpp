/*
 * Copyright (c) 2012 by Jörn Dinkla, www.dinkla.com, All rights reserved.
 * 
 * Bemerkung: Reines CUDA Runtime-API (bis auf die Klasse CExtent)
 */
// CUDA
#include <cuda_runtime_api.h>
#include <vector_functions.h>
#include "vector_extras.h"
// Application
#include "CExtent.h"
#include "smooth3.h"
#include "smoothCPU.h"
#include "Utils.h"
// C++
#include <iostream>
using namespace std;

extern bool compareWithCPU;

void smoothing1(const CExtent& extent) {

	cudaError_t err;

	const int size = extent.getNumberOfElements() * sizeof(uchar4);
	cout << "Size=" << (1.0*size) / (1024*1024) << " MB" << endl;

	// Initialisiere Host-Daten
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
	initializeBuffer(h_input, extent);

	// Initialisiere Buffer auf dem Device
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

	// Kopiere zum Device
	err = cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice);
	if (err != cudaSuccess) {
		cerr << "ERROR: " << err << endl;
	}

	const int windowSize = 1;

	// Rufe den Kernel für das Smoothing auf
	CExecConfig config(extent);
	smooth3(config, extent, d_input, d_output, windowSize);

	// Kopiere das Ergebnis zurück zum Host
	err = cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost);
	if (err != cudaSuccess) {
		cerr << "ERROR: " << err << endl;
	}

	// Vergleiche die Ergebnisse zwischen CPU und GPU
	if (compareWithCPU) {
		uchar4* h_outputCPU = (uchar4*) malloc(size);
		smoothCPU(extent, h_input, h_outputCPU, windowSize);
		bool ae = allEqual(h_outputCPU, h_output, extent);
		free(h_outputCPU);
	}

	// Räume auf
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


void initialize(uchar4* h_input, const CExtent& extent) {
	for (int y=0; y<extent.height; y++) {
		for (int x=0; x<extent.width; x++) {
			h_input[extent.index(x, y)] = make_uchar4(x, y, 0, x+y);
		}
	}
}

void x() {

	CExtent extent(1024, 1024, 1); int windowSize = 1;
	int size = extent.getNumberOfElements() * sizeof(uchar4);

	uchar4 *h_input, *h_output;
	h_input = (uchar4*) malloc(size);
	initialize(h_input, extent);
	h_output = (uchar4*) malloc(size);

	smoothCPU(extent, h_input, h_output, windowSize);
	
	free(h_input); free(h_output);
}
