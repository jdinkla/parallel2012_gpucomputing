/*
 * Copyright (c) 2012 by Jörn Dinkla, www.dinkla.com, All rights reserved.
 * 
 * Bemerkung: Naive Umsetzung des 2D-Codes zu 3D.
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
// C++
#include <iostream>
using namespace std;

extern bool compareWithCPU;

// Benutzung der CBufferPair-Klasse
void smoothing3d(const CExtent& extent) {

	const int size = extent.getNumberOfElements() * sizeof(uchar4);
	cout << "Size=" << (1.0*size) / (1024*1024) << " MB" << endl;

	// Buffer-Pair
	CBufferPair<uchar4> input(extent);
	CBufferPair<uchar4> output(extent);

	input.malloc();
	output.malloc();

	// Initialisiere Daten
	initializeBuffer(*input.host);

	// Kopiere zum Device
	input.updateDevice();

	// Rufe Host-Smoothing auf
	const int windowSize = 1;
	
	// Rufe den Kernel für das Smoothing auf
	CExecConfig config(extent);
	cout << "Exec config=" << config << endl;
	smooth3d(config, *input.device, *output.device, windowSize);

	// Kopiere das Ergebnis zurück zum Host
	output.device->incVersion();			// Dieses gehört in Funktion smooth3
	output.updateHost();

	// Vergleiche die Ergebnisse zwischen CPU und GPU
	if (compareWithCPU) {
		CHostBuffer<uchar4> h_outputCPU(extent);
		h_outputCPU.malloc();
		smoothCPU(extent, input.host->getPtr(), h_outputCPU.getPtr(), windowSize);
		bool ae = allEqual(h_outputCPU.getPtr(), output.host->getPtr(), extent);
		h_outputCPU.free();
	}

	// Räume auf
	input.free();
	output.free();

}
