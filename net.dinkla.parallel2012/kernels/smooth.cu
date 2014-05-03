#include "CExtent.h"
#include "CAccum.h"

extern "C"
__global__
void smooth(const CExtent extent,
			const uchar4* d_input, uchar4* d_output,
			const int windowSize) {
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;
	const int z = blockIdx.z * blockDim.z + threadIdx.z;
	if (extent.inBounds(x, y, z)) {
		CAccum acc;
		for (int dz = -windowSize; dz <= windowSize; dz ++) {
			for (int dy = -windowSize; dy <= windowSize; dy ++) {
				for (int dx = -windowSize; dx <= windowSize; dx ++) {
					int nx=x+dx; int ny=y+dy; int nz=z+dz;
					if (extent.inBoundsStrict(nx, ny, nz)) {
						acc.add(d_input[extent.index(nx, ny, nz)]);
					}
				}
			}
		}
		d_output[extent.index(x, y, z)] = acc.avg();
	}
}
