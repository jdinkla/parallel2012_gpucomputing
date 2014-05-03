/*
 * Copyright (c) 2012 by Jörn Dinkla, www.dinkla.com, All rights reserved.
 */

#ifndef CPINNEDHOSTBUFFER_H
#define CPINNEDHOSTBUFFER_H

#include "CHostBuffer.h"
#include <cuda_runtime_api.h>

#define USE_CUDAMEMCPY

template <class T>
class CPinnedHostBuffer : public CHostBuffer<T> {
public:

	CPinnedHostBuffer(const CExtent& extent) 
		: CHostBuffer<T>(extent) {
	}

	virtual void malloc() {
		const size_t size = getNumberOfElements() * sizeof(T);
		cudaError_t err = cudaMallocHost((void**) &ptr, size);
		if (err != cudaSuccess) {
			cerr << "ERROR: " << cudaGetErrorString(err) << ", size=" << size << endl;
			throw 1201;
		}
		incVersion();
	}

	virtual void free()  {
		if (isAllocated()) {
			cudaError_t err = cudaFreeHost(ptr);
			if (err != cudaSuccess) {
				cerr << "ERROR: " << cudaGetErrorString(err) << endl;
				throw 1202;
			}
			resetVersion();
			ptr = 0;
		}
	}

	// Kopiere vom Pinned
	void copyFrom(const T* h_ptr) {
		const size_t size = getNumberOfElements() * sizeof(T);
		copyFrom(h_ptr, size);
	}

	void copyFrom(const T* h_ptr, const size_t size) {
		copyFrom(h_ptr, size, make_int3(0, 0, 0));
	}

	void copyFrom(const T* h_ptr, const size_t size, const int3 offset) {
		const T* ptr = getPtr(offset);
#ifdef USE_CUDAMEMCPY
		cudaError_t err = cudaMemcpy((void*) h_ptr, ptr, size, cudaMemcpyHostToHost);
		if (err != cudaSuccess) {
			cerr << "ERROR: " << cudaGetErrorString(err) << endl;
			throw 1203;
		}
#else
		memcpy((void*) h_ptr, ptr, size);
#endif
	}

	// Kopiere zum Pinned
	void copyTo(T* h_ptr) {
		const size_t size = getNumberOfElements() * sizeof(T);
		copyTo(h_ptr, size);
	}

	void copyTo(T* h_ptr, const size_t size) {
#ifdef USE_CUDAMEMCPY
		cudaError_t err = cudaMemcpy(ptr, h_ptr, size, cudaMemcpyHostToHost);
		if (err != cudaSuccess) {
			cerr << "ERROR: " << cudaGetErrorString(err) << endl;
			throw 1204;
		}
#else
		memcpy(ptr, h_ptr, size);	
#endif
	}

};


#endif

