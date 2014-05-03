/*
 * Copyright (c) 2012 by Jörn Dinkla, www.dinkla.com, All rights reserved.
 * 
 * Bemerkung: 3D-Code mit Swapping.
 */
// CUDA
#include <cuda_runtime_api.h>
#include <vector_functions.h>
#include "vector_extras.h"
// Application
#include "CExtent.h"
#include "smooth3d_swap.h"
#include "smoothCPU.h"
#include "CPinnedHostBuffer.h"
#include "CDeviceBuffer.h"
#include "CBufferPair.h"
#include "Utils.h"
#include "CTimed.h"
#include "CBufferIterator.h"
// C++
#include <iostream>
using namespace std;

extern bool compareWithCPU;

void smoothing3d_swap(const CExtent& extentHost, const CExtent& extentDevice, const CExtent& extentPartition, const int windowSize) {

	CTimed timer;

	cout << "Extent Host:      " << extentHost << endl;
	cout << "Extent Device:    " << extentDevice << endl;
	cout << "Extent Partition: " << extentPartition << endl;
	
	const int sizeHost = extentHost.getNumberOfElements() * sizeof(uchar4);
	cout << "Size Host:     " << (1.0*sizeHost) / (1024*1024) << " MB" << endl;

	// Buffer-Pair
	CBufferPair<uchar4> input(extentHost, extentDevice);			// Mit Padding
	CBufferPair<uchar4> output(extentHost, extentPartition);		// Ohne Padding

	input.malloc();
	output.malloc();

	// Initialisiere Daten
	initializeBuffer(*input.host);

	// Konfiguriere Iterationen
	CExecConfig config(extentPartition);
	cout << "Exec config=" << config << endl;

	CBufferIterator it(extentHost, extentPartition);
	it.padBefore = make_int3(0, 0, 1);
	it.padAfter = make_int3(0, 0, 1);

	// Iteriere
	while (it.next()) {
		cout << "------------------------------------------------------------" << endl;
		cout << "Iteration: " << it.getCurrentStart() << " - " << it.getCurrentEnd() << " (" << it.getCurrentSize() << ")" << endl;
		cout << "Padded:    " << it.getCurrentStartPadded() << " - " << it.getCurrentEndPadded() << " (" << it.getCurrentSizePadded() << ")"  << endl;

		const int3 offset = it.getCurrentStart() - it.getCurrentStartPadded(); // Der offset im Device-Buffer wg. Padding
		cout << "Offset:    " << offset << endl;

		// Kopiere zum Device
		timer.start();
		input.updateDevice(it.getCurrentStartPadded(), it.getCurrentSizePadded());
		timer.stop();
		cout << "H2D:    " << timer.getDuration() << endl;

		// Rufe den Kernel für das Smoothing auf
		CExtent realExtent(it.getCurrentSizePadded());
		timer.start();
		smooth3d_swap(config, *input.device, *output.device, windowSize, offset, realExtent);
		cudaDeviceSynchronize();
		timer.stop();
		cout << "DEVICE: " << timer.getDuration() << endl;

		timer.start();
		output.updateHost(it.getCurrentStart(), it.getCurrentSize());
		timer.stop();
		cout << "D2H:    " << timer.getDuration() << endl;
	}

	// Vergleiche die Ergebnisse zwischen CPU und GPU
	if (compareWithCPU) {
		CHostBuffer<uchar4> h_outputCPU(extentHost);
		h_outputCPU.malloc();
		timer.start();
		smoothCPU(extentHost, input.host->getPtr(), h_outputCPU.getPtr(), windowSize);
		timer.stop();
		cout << "HOST:   " << timer.getDuration() << endl;
		bool ae = allEqual(h_outputCPU.getPtr(), output.host->getPtr(), extentHost);
		h_outputCPU.free();
	}

	// Räume auf
	input.free();
	output.free();

}

