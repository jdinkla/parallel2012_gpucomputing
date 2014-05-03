/*
 * Copyright (c) 2012 by Jörn Dinkla, www.dinkla.com, All rights reserved.
 */

#include <omp.h>
#include "SmoothCPU.h"
#include "CAccum.h"

uchar4 smooth(const int width, const int height, const int depth,
			  const uchar4* h_input, 
			  const int x, const int y, const int z, 
			  const int windowSize) {
	int4 a = make_int4(0, 0, 0, 0); int c = 0;
	for (int dz = -windowSize; dz <= windowSize; dz ++) {
		for (int dy = -windowSize; dy <= windowSize; dy ++) {
			for (int dx = -windowSize; dx <= windowSize; dx ++) {
				int nx=x+dx; int ny=y+dy; int nz=z+dz;
				if (0<=nx && nx<width && 0<=ny && ny<height 
					    && 0<=nz && nz<depth) {
					uint64 idx = nz*(height*width)+ny*width+nx;
					uchar4 b = h_input[idx];
					c++; a.x += b.x; a.y += b.y; a.z += b.z; } } } }
	if (0 == c) return make_uchar4(0, 0, 0, 255);
	else return make_uchar4(a.x/c, a.y/c, a.z/c, 255);
}

void smoothCPU(const int width, const int height, const int depth, 
			   const uchar4* h_input, uchar4* h_output, 
			   const int windowSize) {
	for (int z=0; z<depth; z++) {
		for (int y=0; y<height; y++) {
			for (int x=0; x<width; x++) {
				const uint64 idx = z*(height*width)+y*width+x;
				h_output[idx] = smooth(width, height, depth, 
					h_input, x, y, z, windowSize);
			}
		}
	}
}

uchar4 smooth(const CExtent& extent, const uchar4* h_input, 
			  const int x, const int y, const int z, 
			  const int windowSize) {
	CAccum acc;
	for (int dz = -windowSize; dz <= windowSize; dz ++) {
		for (int dy = -windowSize; dy <= windowSize; dy ++) {
			for (int dx = -windowSize; dx <= windowSize; dx ++) {
				int nx=x+dx; int ny=y+dy; int nz=z+dz;
				if (extent.inBoundsStrict(nx, ny, nz)) {
					acc.add(h_input[extent.index(nx, ny, nz)]);
				} } } }
	return acc.avg();
}

void smoothCPU(const CExtent& extent, 
			   const uchar4* h_input, uchar4* h_output, 
			   const int windowSize)
{
#pragma omp parallel for
	for (int z=0; z<extent.depth; z++) {
		for (int y=0; y<extent.height; y++) {
			for (int x=0; x<extent.width; x++) {
				h_output[extent.index(x, y, z)] 
					= smooth(extent, h_input, x, y, z, windowSize);
			}
		}
	}
}

