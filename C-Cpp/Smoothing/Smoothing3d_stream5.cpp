/*
 * Copyright (c) 2012 by Jörn Dinkla, www.dinkla.com, All rights reserved.
 * 
 * Bemerkung: 3D-Code mit Streaming und zwei Threads (Version ohne Kommentare für Folien).
 */
#include <omp.h>
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

void smoothing3d_stream5(
					  const CExtent& extentHost, 
					  const CExtent& extentDevice, 
					  const CExtent& extentPartition, 
					  const int windowSize) {

	CTimed timer;

	// Stream
	cudaStream_t stream1, stream2;
	cudaStreamCreate(&stream1); cudaStreamCreate(&stream2);

	// In
	CHostBuffer<uchar4> h_input(extentHost);		

	CPinnedHostBuffer<uchar4> h_buf1(extentDevice);	
	CPinnedHostBuffer<uchar4> h_buf2(extentDevice);	

	CHostBufferPair<uchar4> h2h_in1(&h_input, &h_buf1);
	CHostBufferPair<uchar4> h2h_in2(&h_input, &h_buf2);

	CDeviceBuffer<uchar4> d_buf1(extentDevice, &stream1);	
	CDeviceBuffer<uchar4> d_buf2(extentDevice, &stream2);	

	CBufferPair<uchar4> h2d_in1(&h_buf1, &d_buf1);
	CBufferPair<uchar4> h2d_in2(&h_buf2, &d_buf2);

	// Out
	CHostBuffer<uchar4> h_output(extentHost);

	CPinnedHostBuffer<uchar4> h_out1(extentDevice);	
	CPinnedHostBuffer<uchar4> h_out2(extentDevice);	

	CHostBufferPair<uchar4> h2h_out1(&h_output, &h_out1);
	CHostBufferPair<uchar4> h2h_out2(&h_output, &h_out2);

	CDeviceBuffer<uchar4> d_out1(extentDevice, &stream1);	
	CDeviceBuffer<uchar4> d_out2(extentDevice, &stream2);	

	CBufferPair<uchar4> h2d_out1(&h_out1, &d_out1);
	CBufferPair<uchar4> h2d_out2(&h_out2, &d_out2);

	// Malloc
	h_input.malloc(); h_output.malloc();
	h_buf1.malloc(); h_buf2.malloc(); d_buf1.malloc(); d_buf2.malloc();
	h_out1.malloc(); h_out2.malloc(); d_out1.malloc(); d_out2.malloc();
	initializeBuffer(h_input);

	// Config
	CExecConfig config1(extentPartition, stream1);
	CExecConfig config2(extentPartition, stream2);

	CBufferIterator it(extentHost, extentPartition);
	it.padBefore = make_int3(0, 0, 1);
	it.padAfter = make_int3(0, 0, 1);

	CBufferIterator it2 = it.copy();
	it2.next(); 

	timer.start();

	#pragma omp parallel sections
	{
		#pragma omp section
		{
			while (it.next(2)) {
				const int3 offset1 = it.getCurrentStart() - it.getCurrentStartPadded(); 
				h2h_in1.updatePinned(it.getCurrentStartPadded(), it.getCurrentSizePadded());
				h2d_in1.updateDevice(true);
				CExtent realExtent(it.getCurrentSizePadded());
				smooth3d_swap(config1, d_buf1, d_out1, windowSize, offset1, realExtent);
				h2d_out1.updateHost(true);
				h2h_out1.updateHost(it.getCurrentStart(), it.getCurrentSize());
				it.next();  
			}
		}
		#pragma omp section
		{
			while (it2.next(2)) {
				const int3 offset2 = it2.getCurrentStart() - it2.getCurrentStartPadded(); 
				h2h_in2.updatePinned(it2.getCurrentStartPadded(), it2.getCurrentSizePadded());
				h2d_in2.updateDevice(true);
				CExtent realExtent(it2.getCurrentSizePadded());
				smooth3d_swap(config2, d_buf2, d_out2, windowSize, offset2, realExtent);
				h2d_out2.updateHost(true);
				h2h_out2.updateHost(it2.getCurrentStart(), it2.getCurrentSize());
				it2.next();
			}
		}
	}

	cudaStreamSynchronize(stream1); cudaStreamSynchronize(stream2);
	timer.stop();
	cout << "GPU:   " << timer.getDuration() << endl;

	// Vergleiche die Ergebnisse zwischen CPU und GPU
	if (compareWithCPU) {
		CHostBuffer<uchar4> h_outputCPU(extentHost);						// CPU Daten
		h_outputCPU.malloc();
		cout << "Running on CPU" << endl;
		timer.start();
		smoothCPU(extentHost, h_input.getPtr(), h_outputCPU.getPtr(), windowSize);
		timer.stop();
		cout << "HOST:   " << timer.getDuration() << endl;
		bool ae = allEqual(h_outputCPU.getPtr(), h_output.getPtr(), extentHost);
		h_outputCPU.free();
	}

	// Free
	h_input.free(); h_buf1.free(); h_buf2.free(); d_buf1.free(); d_buf2.free();
	h_output.free(); h_out1.free(); h_out2.free(); d_out1.free(); d_out2.free();

}

/*
	Für Folie

	CBufferIterator it(extentHost, extentPartition);
	it.padBefore = make_int3(0, 0, 1);
	it.padAfter = make_int3(0, 0, 1);

	CBufferIterator it2 = it.copy();
	it2.next();

	#pragma omp parallel sections
	{

		#pragma omp section
		{
			while (it.next(2)) ...
		}

		#pragma omp section
		{
			while (it2.next(2)) ...
		}
	}
*/


