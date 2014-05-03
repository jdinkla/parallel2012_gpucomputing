/*
 * Copyright (c) 2012 by Jörn Dinkla, www.dinkla.com, All rights reserved.
 * 
 * Bemerkung: 3D-Code mit Streaming mit kleineren Pinned-Buffern (Page-Locked).
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
#include "CUDAUtilities.h"
#include "CHostBufferPair.h"
// C++
#include <iostream>
using namespace std;

extern bool compareWithCPU;

void smoothing3d_stream2(const CExtent& extentHost, const CExtent& extentDevice, const CExtent& extentPartition, const int windowSize) {

	if (!supportsOverlap()) {
		return;
	}

	CTimed timer;

	cudaError_t err;
	cudaStream_t stream1;
	cudaStream_t stream2;

	err = cudaStreamCreate(&stream1);
	if (err != cudaSuccess) {
		cerr << "ERROR: " << cudaGetErrorString(err) << endl;
	}

	err = cudaStreamCreate(&stream2);
	if (err != cudaSuccess) {
		cerr << "ERROR: " << cudaGetErrorString(err) << endl;
	}

	cout << "Extent Host:      " << extentHost << endl;
	cout << "Extent Device:    " << extentDevice << endl;
	cout << "Extent Partition: " << extentPartition << endl;
	
	const uint64 sizeHost = extentHost.getNumberOfElements() * sizeof(uchar4);
	cout << "Size Host:     " << (1.0*sizeHost) / (1024*1024) << " MB" << endl;

	// Input-Buffer
	CHostBuffer<uchar4> h_input(extentHost);		

	CPinnedHostBuffer<uchar4> h_pinned_input1(extentDevice);	
	CHostBufferPair<uchar4> h_copy_input1(&h_input, &h_pinned_input1);
	CDeviceBuffer<uchar4> d_input1(extentDevice, &stream1);				// Mit Padding
	CBufferPair<uchar4> input1(&h_pinned_input1, &d_input1);

	CPinnedHostBuffer<uchar4> h_pinned_input2(extentDevice);	
	CHostBufferPair<uchar4> h_copy_input2(&h_input, &h_pinned_input2);
	CDeviceBuffer<uchar4> d_input2(extentDevice, &stream2);				// Mit Padding
	CBufferPair<uchar4> input2(&h_pinned_input2, &d_input2);

	// Output-Buffer
	CHostBuffer<uchar4> h_output(extentHost);
	
	CPinnedHostBuffer<uchar4> h_pinned_output1(extentPartition);
	CHostBufferPair<uchar4> h_copy_output1(&h_output, &h_pinned_output1);	
	CDeviceBuffer<uchar4> d_output1(extentPartition, &stream1);			// Ohne Padding
	CBufferPair<uchar4> output1(&h_pinned_output1, &d_output1);

	CPinnedHostBuffer<uchar4> h_pinned_output2(extentPartition);
	CHostBufferPair<uchar4> h_copy_output2(&h_output, &h_pinned_output2);	
	CDeviceBuffer<uchar4> d_output2(extentPartition, &stream2);			// Ohne Padding
	CBufferPair<uchar4> output2(&h_pinned_output2, &d_output2);

	h_input.malloc();
	h_pinned_input1.malloc();
	d_input1.malloc();
	h_pinned_input2.malloc();
	d_input2.malloc();

	h_output.malloc();
	h_pinned_output1.malloc();
	d_output1.malloc();
	h_pinned_output2.malloc();
	d_output2.malloc();

	// Initialisiere Daten
	initializeBuffer(h_input);

	// Konfiguriere Iterationen
	CExecConfig config1(extentPartition, stream1);
	CExecConfig config2(extentPartition, stream2);

	CBufferIterator it(extentHost, extentPartition);
	it.padBefore = make_int3(0, 0, 1);
	it.padAfter = make_int3(0, 0, 1);

	bool even = true;

	// Iteriere
	while (it.next()) {

		CBufferIterator it2 = it.copy();
		const bool hasNext = it2.next();

		// Der offset im Device-Buffer wg. Padding
		const int3 offset1 = it.getCurrentStart() - it.getCurrentStartPadded(); 
		const int3 offset2 = it2.getCurrentStart() - it2.getCurrentStartPadded(); 

		// Kopiere vom großen Host-Buffer in kleinen "Pinned Buffer"
		h_copy_input1.updatePinned(it.getCurrentStartPadded(), it.getCurrentSizePadded());
		if (hasNext) {
			h_copy_input2.updatePinned(it2.getCurrentStartPadded(), it2.getCurrentSizePadded());
		}

		// Kopiere vom Pinned Buffer zum Device (1:1)
		input1.updateDevice(true);
		if (hasNext) {
			input2.updateDevice(true);
		}

		// Rufe den Kernel für das Smoothing auf
		CExtent realExtent1(it.getCurrentSizePadded());
		smooth3d_swap(config1, *input1.device, *output1.device, windowSize, offset1, realExtent1);
		if (hasNext) {
			CExtent realExtent2(it2.getCurrentSizePadded());
			smooth3d_swap(config2, *input2.device, *output2.device, windowSize, offset2, realExtent2);
		}

		// Kopiere das Ergebnis zurück zum Host (in den Pinned Buffer)
		output1.updateHost(true);
		if (hasNext) {
			output2.updateHost(true);
		}

		// Kopiere vom Pinned Buffer in den großen Buffer
		h_copy_output1.updateHost(it.getCurrentStart(), it.getCurrentSize());
		if (hasNext) {
			h_copy_output2.updateHost(it.getCurrentStart(), it.getCurrentSize());
		}

		if (hasNext) {
			it.next();		// Den zweiten Schritt aufholen
		}
	}

	cudaStreamSynchronize(stream1);
	cudaStreamSynchronize(stream2);

	// Vergleiche die Ergebnisse zwischen CPU und GPU
	if (compareWithCPU) {
		CHostBuffer<uchar4> h_outputCPU(extentHost);						// CPU Daten
		h_outputCPU.malloc();
		cout << "Running on CPU" << endl;
		timer.start();
		smoothCPU(extentHost, h_input.getPtr(), h_outputCPU.getPtr(), windowSize);
		timer.stop();
		cout << "HOST:   " << timer.getDuration() << endl;
		bool ae = allEqual(h_outputCPU.getPtr(), h_input.getPtr(), extentHost);
		h_outputCPU.free();
	}

	// Räume auf
	h_input.free();
	h_output.free();
	h_pinned_input1.free();
	d_input1.free();
	h_pinned_input2.free();
	d_input2.free();
	h_pinned_output1.free();
	d_output1.free();
	h_pinned_output2.free();
	d_output2.free();
	
	err = cudaStreamDestroy(stream1);
	if (err != cudaSuccess) {
		cerr << "ERROR: " << cudaGetErrorString(err) << endl;
	}

	err = cudaStreamDestroy(stream2);
	if (err != cudaSuccess) {
		cerr << "ERROR: " << cudaGetErrorString(err) << endl;
	}

}
