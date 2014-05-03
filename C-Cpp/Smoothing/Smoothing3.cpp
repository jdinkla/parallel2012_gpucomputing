/*
 * Copyright (c) 2012 by Jörn Dinkla, www.dinkla.com, All rights reserved.
 * 
 * Bemerkung: Benutzung der CBufferPair-Klasse und CDeviceBuffer beim Aufruf des Kernel-Wrappers
 */
// CUDA
#include <cuda_runtime_api.h>
#include <vector_functions.h>
#include "vector_extras.h"
// Application
#include "CExtent.h"
#include "smooth3.h"
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
void smoothing3(const CExtent& extent) {

	// cudaError_t err;		 TODO Gutes Zeichen! Nicht mehr notwendig! Alles in Klassen!

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
	smooth3(config, *input.device, *output.device, windowSize);

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

  
// Benutzung der CBufferPair-Klasse
void smoothing3cu(const CExtent& extent, const int windowSize) {

	CTimed timer, timerH2D, timerD2H;
	uint64 gpu, cpu, h2d, d2h;

	CBufferPair<uchar4> input(extent);
	CBufferPair<uchar4> output(extent);
	input.malloc(); output.malloc();

	initializeBuffer(*input.host);
	timerH2D.start();
	input.host->incVersion(); input.updateDevice();
	timerH2D.stop();
	h2d = timerH2D.getDuration();

	CExecConfig config(extent);
	timer.start();
	smooth3d(config, *input.device, *output.device, windowSize);
	cudaDeviceSynchronize();
	timer.stop();
	gpu = timer.getDuration();

	timerD2H.start();
	output.device->incVersion(); output.updateHost();
	timerD2H.stop();
	d2h = timerD2H.getDuration();

	// Vergleiche die Ergebnisse zwischen CPU und GPU
	if (compareWithCPU) {
		CHostBuffer<uchar4> h_outputCPU(extent);
		h_outputCPU.malloc();
		timer.start();
		smoothCPU(extent, input.host->getPtr(), h_outputCPU.getPtr(), windowSize);
		timer.stop();
		cpu = timer.getDuration();
		bool ae = allEqual(h_outputCPU.getPtr(), output.host->getPtr(), extent);
		h_outputCPU.free();
	}

	input.free(); output.free();

	uint64 sum = h2d + gpu + h2d;
	float speedUp1 = 1.0 * cpu / gpu;
	float speedUp2 = 1.0 * cpu / sum;
	cout << "#;" << extent.width 
		<< ";" << cpu 
		<< ";" << gpu 
		<< ";" << h2d 
		<< ";" << d2h 
		<< ";" << sum
		<< ";" << speedUp1
		<< ";" << speedUp2
		<< endl;
}

