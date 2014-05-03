/*
 * Copyright (c) 2012 by Jörn Dinkla, www.dinkla.com, All rights reserved.
 */

#ifndef CHOSTBUFFER_H
#define CHOSTBUFFER_H

#include "CBaseBuffer.h"
#include <cuda_runtime_api.h>
#include <stdlib.h>

template <class T>
class CHostBuffer : public CBaseBuffer<T> {
public:

	CHostBuffer(const CExtent& extent) 
		: CBaseBuffer<T>(extent) {
	}

	virtual void malloc() {
		ptr = (T*) ::malloc(getNumberOfElements() * sizeof(T));
		if (0 == ptr) {
			cerr << "ERROR: malloc" << endl;
			throw 1001;
		}
		incVersion();
	}

	virtual void free()  {
		if (isAllocated()) {
			::free(ptr);
			resetVersion();
			ptr = 0;
		}
	}

	virtual void memset(const T& value) {
		T* p = ptr;
		for (int z=0; z<depth;z++) {
			for (int y=0; y<height;y++) {
				for (int x=0; x<width;x++) {
					*p = value;
					p++;
				}
			}
		}
	}

	T& operator[](const int index) {
		return ptr[index];
	}

};

#endif
