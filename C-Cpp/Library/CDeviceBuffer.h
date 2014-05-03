/*
 * Copyright (c) 2012 by Jörn Dinkla, www.dinkla.com, All rights reserved.
 */

#ifndef CDEVICEBUFFER_H
#define CDEVICEBUFFER_H

#include "CBaseBuffer.h"
#include <cuda_runtime_api.h>
#include <iostream>
using namespace std;

template <class T>
class CDeviceBuffer : public CBaseBuffer<T> {

protected:

	cudaStream_t* stream;

public:

	CDeviceBuffer(const CExtent& extent, cudaStream_t* _stream = 0) 
		: CBaseBuffer<T>(extent)
		, stream(_stream) {
	}

	void malloc() {
#ifdef _MSC_VER
		cudaError_t err = cudaMalloc((void**) &ptr, getNumberOfElements() * sizeof(T));
#else
		// TODO Hack bei GNU C++
		T* tmp = 0;
		cudaError_t err = cudaMalloc((void**) &tmp, getNumberOfElements() * sizeof(T));
		CBaseBuffer<T>::ptr = tmp;
#endif

		if (err != cudaSuccess) {
			cerr << "ERROR: " << cudaGetErrorString(err) << endl;
			throw 1101;
		}
		incVersion();
	}

	void free() {
		if (isAllocated()) {
			cudaError_t err = cudaFree(CBaseBuffer<T>::ptr);
			if (err != cudaSuccess) {
				cerr << "ERROR: " << cudaGetErrorString(err) << endl;
				throw 1102;
			}
			resetVersion();
			ptr = 0;
		}
	}

	void memset(const T& value) {
		// TODO Kernel schreiben			
	}

	// Kopiere vom Device
	void copyFrom(const T* h_ptr) {
		const size_t size = getNumberOfElements() * sizeof(T);
		copyFrom(h_ptr, size);
	}

	void copyFrom(const T* h_ptr, const size_t size) {
		copyFrom(h_ptr, size, make_int3(0, 0, 0));
	}

	void copyFrom(const T* h_ptr, const size_t size, const int3 offset) {
		cudaError_t err;
		const T* ptr = getPtr(offset);
		if (0 != stream) {
			err = cudaMemcpyAsync((void*) h_ptr, ptr, size, cudaMemcpyDeviceToHost, *stream);
		} else {
			err = cudaMemcpy((void*) h_ptr, ptr, size, cudaMemcpyDeviceToHost);
		}
		if (err != cudaSuccess) {
			cerr << "ERROR: " << cudaGetErrorString(err) << endl;
		}
	}

	// Kopiere zum Device
	void copyTo(T* h_ptr) {
		const size_t size = getNumberOfElements() * sizeof(T);
		copyTo(h_ptr, size);
	}

	void copyTo(T* h_ptr, const size_t size) {
		cudaError_t err;
		if (0 != stream) {
			err = cudaMemcpyAsync(CBaseBuffer<T>::ptr, h_ptr, size, cudaMemcpyHostToDevice, *stream);
		} else {
			err = cudaMemcpy(CBaseBuffer<T>::ptr, h_ptr, size, cudaMemcpyHostToDevice);
		}
		if (err != cudaSuccess) {
			cerr << "ERROR: " << cudaGetErrorString(err) << endl;
		}
	}

};


#endif
