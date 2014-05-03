/*
 * Copyright (c) 2012 by Jörn Dinkla, www.dinkla.com, All rights reserved.
 */

#ifndef CSTREAM_H
#define CSTREAM_H

#include <cuda_runtime_api.h>
#include <iostream>
using namespace std;

class CStream 
{

public:

	cudaStream_t stream;

	CStream() {
		stream = 0;
	};

	~CStream() {
		if (0 != stream) {
			destroy();
		}
	}

	void create() {
		cudaError_t err = cudaStreamCreate(&stream);
		if (err != cudaSuccess) {
			cerr << "ERROR: " << cudaGetErrorString(err) << endl;
		}
	}

	void sync() {
		cudaStreamSynchronize(stream);
	}

	void destroy() {
		cudaError_t err = cudaStreamDestroy(stream);
		if (err != cudaSuccess) {
			cerr << "ERROR: " << cudaGetErrorString(err) << endl;
		}
	}

};

#endif
