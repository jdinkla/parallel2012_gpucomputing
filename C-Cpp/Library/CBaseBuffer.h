/*
 * Copyright (c) 2012 by Jörn Dinkla, www.dinkla.com, All rights reserved.
 */

#ifndef CBASEBUFFER_H
#define CBASEBUFFER_H

#include "CVersion.h"
#include "CExtent.h"
#include <cuda_runtime_api.h>

/*

template <class T>
class CBaseBuffer : public CVersion, public CExtent {
protected:
	T* ptr;
public:
	CBaseBuffer(const CExtent& extent); 
	T* getPtr() const;
	virtual void malloc() = 0;
	virtual void free() = 0;
	virtual void memset(const T& value) = 0;


*/

template <class T>
class CBaseBuffer : public CVersion, public CExtent {

protected:

	T* ptr;

public:

	CBaseBuffer(const CExtent& extent) 
		: CVersion()
		, CExtent(extent)
		, ptr(0) {
	}

	virtual ~CBaseBuffer() {
	}

	T* getPtr() const {
		return ptr;
	}

	T* getPtr(const int3& i) const {
		const uint64 n = index(i);
		T* mptr = &ptr[n];
		return mptr;
	}

	virtual void malloc() = 0;

	virtual void free() = 0;

	virtual void memset(const T& value) = 0;

	bool isAllocated() {
		return getVersion() > 0;
	}

};


#endif
