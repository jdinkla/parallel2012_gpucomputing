/*
 * Copyright (c) 2012 by Jörn Dinkla, www.dinkla.com, All rights reserved.
 */

#ifndef CEXECCONFIG_H
#define CEXECCONFIG_H

#include <cuda_runtime_api.h>
#include "CExtent.h"
#include "vector_extras.h"

/*
class CExecConfig {
public:
	dim3 grid;
	dim3 threads;
	cudaStream_t stream;

	CExecConfig(const CExtent& extent) {
		threads = dim3(128, 1, 1);		// CC!
		grid = dim3((extent.width + threads.x - 1) / threads.x, 
					(extent.height + threads.y - 1) / threads.y, 
					(extent.depth + threads.z - 1) / threads.z
					);
	}
*/

class CExecConfig {

public:

	dim3 threads;
	dim3 grid;
	cudaStream_t stream;

	CExecConfig() {
		grid = dim3(0, 0, 0);
		threads = dim3(0, 0, 0);
		stream = 0;
	}

	CExecConfig(const CExtent& extent) {
		threads = dim3(128, 1, 1);		// CC!
		//threads = dim3(1024, 1, 1);		// CC!
		grid = dim3((extent.width + threads.x - 1) / threads.x, 
					(extent.height + threads.y - 1) / threads.y, 
					(extent.depth + threads.z - 1) / threads.z
					);
	}

	CExecConfig(const CExtent& extent, const int n) {
		threads = dim3(n, n, 1);
		grid = dim3((extent.width + threads.x - 1) / threads.x, (extent.height + threads.y - 1) / threads.y, 1);
	}

	CExecConfig(const CExtent& extent, cudaStream_t& _stream) {
		stream = _stream;
		threads = dim3(128, 1, 1);
		grid = dim3((extent.width + threads.x - 1) / threads.x, 
					(extent.height + threads.y - 1) / threads.y, 
					(extent.depth + threads.z - 1) / threads.z
					);
	}

};

inline std::ostream &operator<<(std::ostream& ostr, const CExecConfig& ec) 
{
	return ostr << "grid=(" << ec.grid << "), threadblock=(" << ec.threads << ")";
}

#endif
