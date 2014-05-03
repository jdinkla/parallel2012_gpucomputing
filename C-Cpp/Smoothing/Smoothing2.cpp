/*
 * Copyright (c) 2012 by Jörn Dinkla, www.dinkla.com, All rights reserved.
 * 
 * Bemerkung: Benutzung der CBuffer-Klassen
 */
// CUDA
#include <cuda_runtime_api.h>
#include <vector_functions.h>
#include "vector_extras.h"
// Application
#include "CExtent.h"
#include "smooth3.h"
#include "smoothCPU.h"
#include "CPinnedHostBuffer.h"
#include "CDeviceBuffer.h"
#include "Utils.h"
// C++
#include <iostream>
using namespace std;

extern bool compareWithCPU;

void smoothing2(const CExtent& extent) {

	cudaError_t err;

	const int size = extent.getNumberOfElements() * sizeof(uchar4);
	cout << "Size=" << (1.0*size) / (1024*1024) << " MB" << endl;

	// Initialisiere Host-Daten
	CPinnedHostBuffer<uchar4> h_input(extent);
	CPinnedHostBuffer<uchar4> h_output(extent);
	
	h_input.malloc();
	h_output.malloc();

	// Initialisiere Daten
	initializeBuffer(h_input.getPtr(), extent);

	// Initialisiere Buffer auf dem Device
	CDeviceBuffer<uchar4> d_input(extent);
	CDeviceBuffer<uchar4> d_output(extent);

	d_input.malloc();
	d_output.malloc();

	// Kopiere zum Device
	err = cudaMemcpy(d_input.getPtr(), h_input.getPtr(), size, cudaMemcpyHostToDevice);
	if (err != cudaSuccess) {
		cerr << "ERROR: " << err << endl;
	}

	// Rufe Host-Smoothing auf
	const int windowSize = 1;

	// Rufe den Kernel für das Smoothing auf
	CExecConfig config(extent);
	smooth3(config, extent, d_input.getPtr(), d_output.getPtr(), windowSize);

	// Kopiere das Ergebnis zurück zum Host
	err = cudaMemcpy(h_output.getPtr(), d_output.getPtr(), size, cudaMemcpyDeviceToHost);
	if (err != cudaSuccess) {
		cerr << "ERROR: " << err << endl;
	}

	// Vergleiche die Ergebnisse zwischen CPU und GPU
	if (compareWithCPU) {
		CHostBuffer<uchar4> h_outputCPU(extent);
		h_outputCPU.malloc();
		smoothCPU(extent, h_input.getPtr(), h_outputCPU.getPtr(), windowSize);
		bool ae = allEqual(h_outputCPU.getPtr(), h_output.getPtr(), extent);
		h_outputCPU.free();
	}

	// Räume auf
	d_input.free();
	d_output.free();
	h_input.free();
	h_output.free();

}
