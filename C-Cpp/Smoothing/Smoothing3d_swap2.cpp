/*
 * Copyright (c) 2012 by Jörn Dinkla, www.dinkla.com, All rights reserved.
 * 
 * Bemerkung: 3D-Code mit Swapping (Version ohne Kommentare für Folien).
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

void smoothing3d_swap2(const CExtent& extentHost, 
					  const CExtent& extentDevice, 
					  const CExtent& extentPartition, 
					  const int windowSize) {

	CTimed timer;
	CBufferPair<uchar4> input(extentHost, extentDevice);
	CBufferPair<uchar4> output(extentHost, extentPartition);
	input.malloc(); output.malloc();
	initializeBuffer(*input.host);

	CExecConfig config(extentPartition);
	CBufferIterator it(extentHost, extentPartition);
	it.padBefore = make_int3(0, 0, 1);
	it.padAfter = make_int3(0, 0, 1);

	timer.start();
	while (it.next()) {
		input.updateDevice(it.getCurrentStartPadded(), 
						   it.getCurrentSizePadded());
		CExtent realExtent(it.getCurrentSizePadded());
		int3 offset = it.getCurrentStart() - it.getCurrentStartPadded(); 
		smooth3d_swap(config, *input.device, *output.device, 
			windowSize, offset, realExtent);
		output.updateHost(it.getCurrentStart(), it.getCurrentSize());
	}

	cudaDeviceSynchronize();
	timer.stop();

	cout << "#;" << timer.getDuration() << endl;

	input.free(); output.free();

}
 
