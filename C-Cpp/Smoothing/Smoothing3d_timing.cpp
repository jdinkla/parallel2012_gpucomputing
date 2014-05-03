/*
 * Copyright (c) 2012 by Jörn Dinkla, www.dinkla.com, All rights reserved.
 * 
 * Bemerkung: Naive Umsetzung des 2D-Codes zu 3D mit Timing.
 */
// CUDA
#include <cuda_runtime_api.h>
#include <vector_functions.h>
#include "vector_extras.h"
// Application
#include "CExtent.h"
#include "smooth3d.h"
#include "smoothCPU.h"
#include "CPinnedHostBuffer.h"
#include "CDeviceBuffer.h"
#include "CBufferPair.h"
#include "Utils.h"
#include "CTimed.h"
// C++
#include <iostream>
using namespace std;

extern bool compareWithCPU;

// Benutzung der CBufferPair-Klasse
void smoothing3d_timing(const CExtent& extent) {

	CTimed timer;

	cout << "Extent: " << extent << endl;
	const int size = extent.getNumberOfElements() * sizeof(uchar4);
	cout << "Size:   " << (1.0*size) / (1024*1024) << " MB" << endl;

	// Buffer-Pair
	CBufferPair<uchar4> input(extent);
	CBufferPair<uchar4> output(extent);

	input.malloc();
	output.malloc();

	// Initialisiere Daten
	initializeBuffer(*input.host);

	const int windowSize = 1;

	// Kopiere zum Device
	timer.start();
	input.updateDevice();
	timer.stop();
	cout << "H2D:    " << timer.getDuration() << endl;

	// Rufe den Kernel für das Smoothing auf
	CExecConfig config(extent);
	cout << "Exec config=" << config << endl;

	timer.start();
	smooth3d(config, *input.device, *output.device, windowSize);
	cudaDeviceSynchronize();
	timer.stop();
	cout << "DEVICE: " << timer.getDuration() << endl;

	// Kopiere das Ergebnis zurück zum Host
	output.device->incVersion();			// Dieses gehört in Funktion smooth3

	timer.start();
	output.updateHost();
	timer.stop();
	cout << "D2H:    " << timer.getDuration() << endl;

	// Vergleiche die Ergebnisse zwischen CPU und GPU
	if (compareWithCPU) {
		CHostBuffer<uchar4> h_outputCPU(extent);
		h_outputCPU.malloc();
		timer.start();
		smoothCPU(extent, input.host->getPtr(), h_outputCPU.getPtr(), windowSize);
		timer.stop();
		cout << "HOST:   " << timer.getDuration() << endl;
		bool ae = allEqual(h_outputCPU.getPtr(), output.host->getPtr(), extent);
		h_outputCPU.free();
	}

	// Räume auf
	input.free();
	output.free();
}
