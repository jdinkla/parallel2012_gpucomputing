/*
 * Copyright (c) 2012 by Jörn Dinkla, www.dinkla.com, All rights reserved.
 */

#ifndef CHOSTBUFFERPAIR_H
#define CHOSTBUFFERPAIR_H

#include "CHostBuffer.h"
#include "CDeviceBuffer.h"

template <class T>
class CHostBufferPair
{

protected:
	
	bool sameSize;

public:

	CHostBuffer<T>* host;
	CPinnedHostBuffer<T>* pinned;

	CHostBufferPair(CHostBuffer<T>* hostBuffer, CPinnedHostBuffer<T>* pinnedBuffer) {
		host = hostBuffer;
		pinned = pinnedBuffer;
		CExtent hostExtent = (CExtent) *host;
		CExtent pinnedExtent = (CExtent) *pinned;
		sameSize = hostExtent == pinnedExtent;
	}

	virtual ~CHostBufferPair() {
		free();
	}

	virtual void malloc() {
		if (!host->isAllocated()) {
			host->malloc();
		}
		if (!pinned->isAllocated()) {
			pinned->malloc();
		}
	}

	virtual void free() {
		host->free();
		pinned->free();
	}

	void updatePinned(const bool force = false) {
		if (force || host->getVersion() > pinned->getVersion()) {
			cout << "CHostBufferPair::updatePinned" << endl;
			pinned->copyTo(host->getPtr());
		}
	}

	void updatePinned(const int3& currentPos, const int3& size) {
		cout << "CHostBufferPair::updatePinned (" << currentPos << "), (" << size << ")" << endl;
		const size_t sizeInBytes = size.x * size.y * size.z * sizeof(T);
		pinned->copyTo(host->getPtr(currentPos), sizeInBytes);
	}

	void updateHost(const bool force = false) {
		if (force || host->getVersion() < device->getVersion()) {
			cout << "CHostBufferPair::updateHost" << endl;
			pinned->copyFrom(host->getPtr());
		}
	}

	void updateHost(const int3& currentPos, const int3& size) {
		cout << "CHostBufferPair::updateHost (" << currentPos << "), (" << size << ")" << endl;
		const size_t sizeInBytes = size.x * size.y * size.z * sizeof(T);
		pinned->copyFrom(host->getPtr(currentPos), sizeInBytes);
	}

	void updateHost(const int3& currentPos, const int3& size, const int3& offset) {
		cout << "CHostBufferPair::updateHost (" << currentPos << "), (" << size << "), (" << offset << ")" << endl;
		const size_t sizeInBytes = size.x * size.y * size.z * sizeof(T);
		pinned->copyFrom(host->getPtr(currentPos), sizeInBytes, offset);
	}

};

#endif
 