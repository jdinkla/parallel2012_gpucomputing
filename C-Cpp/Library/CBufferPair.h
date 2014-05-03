/*
 * Copyright (c) 2012 by Jörn Dinkla, www.dinkla.com, All rights reserved.
 */

#ifndef CBUFFERPAIR_H
#define CBUFFERPAIR_H

#include "CHostBuffer.h"
#include "CDeviceBuffer.h"

template <class T>
class CBufferPair
{

protected:
	
	bool sameSize;

public:

	CHostBuffer<T>* host;
	CDeviceBuffer<T>* device;

	CBufferPair(const CExtent& extent, cudaStream_t* stream = 0) {
		host = new CHostBuffer<T>(extent);
		device = new CDeviceBuffer<T>(extent, stream);
		sameSize = true;
	}

	CBufferPair(const CExtent& hostExtent, const CExtent& deviceExtent, cudaStream_t* stream = 0) {
		host = new CHostBuffer<T>(hostExtent);
		device = new CDeviceBuffer<T>(deviceExtent, stream);
		sameSize = hostExtent == deviceExtent;
	}

	CBufferPair(CHostBuffer<T>* hostBuffer, CDeviceBuffer<T>* deviceBuffer) {
		host = hostBuffer;
		device = deviceBuffer;
		CExtent hostExtent = (CExtent) *host;
		CExtent deviceExtent = (CExtent) *device;
		sameSize = hostExtent == deviceExtent;
	}

	virtual ~CBufferPair() {
		free();
	}

	virtual void malloc() {
		host->malloc();
		device->malloc();
	}

	virtual void free() {
		if (host->isAllocated()) {
			host->free();
		}
		if (device->isAllocated()) {
			device->free();
		}
	}

	void updateDevice(const bool force = false) {
		cout << "CBufferPair::updateDevice" << endl;
		if (force || host->getVersion() > device->getVersion()) {
			device->copyTo(host->getPtr());
		}
	}

	void updateDevice(const int3& currentPos, const int3& size) {
		cout << "CBufferPair::updateDevice (" << currentPos << "), (" << size << ")" << endl;
		const size_t sizeInBytes = size.x * size.y * size.z * sizeof(T);
		device->copyTo(host->getPtr(currentPos), sizeInBytes);
	}

	void updateHost(const bool force = false) {
		cout << "CBufferPair::updateHost" << endl;
		if (force || host->getVersion() < device->getVersion()) {
			device->copyFrom(host->getPtr());
		}
	}

	void updateHost(const int3& currentPos, const int3& size) {
		const T* ptr = host->getPtr(currentPos);
		const size_t sizeInBytes = size.x * size.y * size.z * sizeof(T);
		cout << "CBufferPair::updateHost (" << currentPos << "), (" << size << "), " 
			<< ptr << ", " << sizeInBytes 
			<< endl;
		device->copyFrom(ptr, sizeInBytes);
	}

	void updateHost(const int3& currentPos, const int3& size, const int3& offset) {
		cout << "CBufferPair::updateHost (" << currentPos << "), (" << size << "), (" << offset << ")" << endl;
		const size_t sizeInBytes = size.x * size.y * size.z * sizeof(T);
		device->copyFrom(host->getPtr(currentPos), sizeInBytes, offset);
	}

};

#endif
