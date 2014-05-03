/*
 * Copyright (c) 2012 by Jörn Dinkla, www.dinkla.com, All rights reserved.
 * 
 * Bemerkung: Auf der CPU für Performancemessung
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

void smoothCPU(const CExtent& extentHost, const int windowSize) {
	CTimed timer;
	CHostBuffer<uchar4> h_input(extentHost);
	h_input.malloc();
	initializeBuffer(h_input);

	CHostBuffer<uchar4> h_outputCPU(extentHost);						// CPU Daten
	h_outputCPU.malloc();
	cout << "Running on CPU" << endl;
	timer.start();
	smoothCPU(extentHost, h_input.getPtr(), h_outputCPU.getPtr(), windowSize);
	timer.stop();
	cout << "#;" << extentHost.width << ";" << timer.getDuration() << endl;
	h_outputCPU.free();
} 
