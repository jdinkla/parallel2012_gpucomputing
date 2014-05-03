/*
 * Copyright (c) 2012 by Jörn Dinkla, www.dinkla.com, All rights reserved.
 */

#ifndef CACCUM_H 
#define CACCUM_H 

#include <cuda_runtime_api.h>
#include <vector_functions.h>

/*
class CAccum {
protected:
	int4 a; int c;
public:
	CAccum() {
		a = make_int4(0, 0, 0, 0);
		c = 0;
	}
	void add(const uchar4& b) {
		c++; a.x += b.x; a.y += b.y; a.z += b.z;
	}
	uchar4 avg() {
		if (0 == c) return make_uchar4(0, 0, 0, 255);
		else return make_uchar4(a.x/c, a.y/c, a.z/c, 255);
	}
};
*/

class CAccum {

public:
	int4 a;

	int count;

	__device__ __host__ CAccum() {
		a = make_int4(0, 0, 0, 0);
		count = 0;
	}

	__device__ __host__ void add(const uchar4& b) {
		count++;
		a.x += b.x;
		a.y += b.y;
		a.z += b.z;
	}

	__device__ __host__ void add(const uchar4& b, const int weight) {
		count++;
		a.x += b.x * weight;
		a.y += b.y * weight;
		a.z += b.z * weight;
	}

	__device__ __host__ uchar4 avg() {
		if (0 == count) {
			return make_uchar4(0, 0, 0, 255);
		} else {
			return make_uchar4(a.x / count, a.y / count, a.z / count, 255);
		}
	}

	__device__ __host__ uchar4 avg(const int div) {
		if (0 == count) {
			return make_uchar4(0, 0, 0, 255);
		} else {
			return make_uchar4(a.x / div, a.y / div, a.z / div, 255);
		}
	}

};

#endif
