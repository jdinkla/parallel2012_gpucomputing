__device__ inline void add(int4& a, const uchar4& b) {
	a.x += b.x;
	a.y += b.y;
	a.z += b.z;
}

extern "C" __global__
void smooth1(const uchar4* d_input, uchar4* d_output,
             const int width, const int height, const int depth,
             const int windowSize) {
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int z = blockIdx.z * blockDim.z + threadIdx.z;
	if (x < width && y < height && z < depth) {
		int4 a = make_int4(0, 0, 0, 0);
		int count = 0;
        for (int dz = -windowSize; dz <= windowSize; dz ++) {
            for (int dy = -windowSize; dy <= windowSize; dy ++) {
                for (int dx = -windowSize; dx <= windowSize; dx ++) {
                    const int nx = x + dx;
                    const int ny = y + dy;
                    const int nz = z + dz;
                    if (0 <= nx && nx < width
                        && 0 <= ny && ny < height
                        && 0 <= nz && nz < depth) {
                        const long idx = nz * (width*height) + ny * width + nx;
#ifdef READ
                        add(a, d_input[idx]);
#endif
                        count++;
                    }
                }
            }
        }
        const long idx = z * (width*height) + y * width + x;
#ifdef WRITE
		d_output[idx] = make_uchar4(a.x/count, a.y/count, a.z/count, 255);
#endif
	}
}

extern "C" __global__
void smooth3(void) {
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int z = blockIdx.z * blockDim.z + threadIdx.z;
}

extern "C" __global__
void smooth2(const uchar4* d_input, uchar4* d_output) {
    const int width = 1024;
    const int height= 1024;
    const int depth = 1;
    const int windowSize = 1;

	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int z = blockIdx.z * blockDim.z + threadIdx.z;

    /*
	if (x < width && y < height && z < depth) {
		int4 a = make_int4(0, 0, 0, 0);
		int count = 0;
        for (int dz = -windowSize; dz <= windowSize; dz ++) {
            for (int dy = -windowSize; dy <= windowSize; dy ++) {
                for (int dx = -windowSize; dx <= windowSize; dx ++) {
                    const int nx = x + dx;
                    const int ny = y + dy;
                    const int nz = z + dz;
                    if (0 <= nx && nx < width
                        && 0 <= ny && ny < height
                        && 0 <= nz && nz < depth) {
                        const long idx = nz * (width*height) + ny * width + nx;
#ifdef READ
                        add(a, d_input[idx]);
#endif
                        count++;
                    }
                }
            }
        }
        const long idx = z * (width*height) + y * width + x;
#ifdef WRITE
		d_output[idx] = make_uchar4(a.x/count, a.y/count, a.z/count, 255);
#endif
	}
	*/
}
