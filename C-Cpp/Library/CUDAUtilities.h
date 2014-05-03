/*
 * Copyright (c) 2012 by Jörn Dinkla, www.dinkla.com, All rights reserved.
 */

#ifndef CUDAUTILITIES_H 
#define CUDAUTILITIES_H

#include <cuda_runtime_api.h>
#include <iostream>
using namespace std;

inline bool supportsOverlap() {
	bool result = true;
	cudaDeviceProp prop;
	int device;
	cudaGetDevice(&device);
	cudaGetDeviceProperties(&prop, device);
	//	cout << "AsyncEngineCount: " << prop.asyncEngineCount << endl;
	if (0 == prop.asyncEngineCount) {
		cerr << "ERROR: Overlapping copies & kernels not possible" << endl;
		result = false;
	}
	return result;
}




#endif
