/*
 * Copyright (c) 2012 by Jörn Dinkla, www.dinkla.com, All rights reserved.
 */

#ifndef CEXTENT_H
#define CEXTENT_H

#include <cuda_runtime_api.h>
#include <iostream>
#include "Defs.h"

/*

class CExtent {
public:
	int width; int height; int depth;

	CExtent(const int w=1, const int h=1, const int d=1)
		: width(w), height(h), depth(d) {}

	__device__ __host__ 
	uint64 index(const int x, const int y, const int z) const {
		return (d(uint64) z) * (width*height) + y*width + x;
	}


class CExtent {
public:
	int width; int height; int depth;

	CExtent(const int w=1, const int h=1, const int d=1)
		: width(w), height(h), depth(d) {}

	uint64 index(const int x, const int y, const int z) const {
		return (d(uint64) z) * (width*height) + y*width + x;
	}

	bool inBounds(const int x, const int y, const int z) const {
		return x<width && y<height && z<depth;
	}

	bool inBoundsStrict(const int x, 
			const int y, const int z) const {
		return 0<=x && x<width && 0<=y && y<height 
			&& 0<=z && z<depth;
	}
	...

};
*/

class CExtent {
	
public:

	int width;
	int height;
	int depth;

	CExtent(const int _width = 1, const int _height = 1, const int _depth = 1)
		: width(_width)
	    , height(_height)
	    , depth(_depth) 
	{
	}

	CExtent(const CExtent& extent)
	        : width(extent.width)
	        , height(extent.height)
	        , depth(extent.depth) {
	}

	CExtent(const int3& d)
		: width(d.x)
		, height(d.y)
		, depth(d.z) {
	}

	__device__ __host__ 
	int index(const int x, const int y) const {
		return y * width + x;
	}

	__device__ __host__ 
	uint64 index(const int x, const int y, const int z) const {
		return ((uint64) z) * (width * height) + y * width + x;
	}

	__device__ __host__ 
	int index32(const int x, const int y, const int z) const {
		return z * (width * height) + y * width + x;
	}

	__device__ __host__ uint64 index(const int3 i) const {
		return ((uint64) i.z) * (width * height) + i.y * width + i.x;
	}

	__device__ __host__ 
	int index32(const int3 i) const {
		return i.z * (width * height) + i.y * width + i.x;
	}

	__device__ __host__ 
	int checkedIndex(const int x, const int y) const {
		if (0 <= x && x < width && 0 <= y && y < height) {
			return y * width + x;
		} else {
			return -1;
		}
	}

	__device__ __host__ 
	bool inBounds(const int x, const int y) const {
		return x < width && y < height;
	}

	__device__ __host__ 
	bool inBoundsStrict(const int x, const int y) const {
		return 0 <= x && x < width && 0 <= y && y < height;
	}

	__device__ __host__ 
	bool inBounds(const int x, const int y, const int z) const {
		return x < width && y < height && z < depth;
	}

	__device__ __host__ 
	bool inBoundsStrict(const int x, const int y, const int z) const {
		return 0 <= x && x < width && 0 <= y && y < height && 0 <= z && z < depth;
	}

	__device__ __host__ 
	bool inBoundsX(const int x) const {
		return x < width;
	}

	__device__ __host__ 
	int getNumberOfElements() const {
		return width * height * depth;
	}

	CExtent& operator=(const CExtent& a) { 
		width = a.width;
		height = a.height;
		depth = a.depth;
		return *this;
	}

	bool operator==(const CExtent& b) const {
		return this->width == b.width && this->height == b.height && this->depth == b.depth;
	}

};

inline std::ostream &operator<<(std::ostream& ostr, const CExtent& d) 
{
	return ostr << d.width << "," << d.height << "," << d.depth;
}

#endif
