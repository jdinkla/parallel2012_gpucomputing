/*
 * Copyright (c) 2012 by Jörn Dinkla, www.dinkla.com, All rights reserved.
 */

#ifndef VECTOR_EXTRAS_H
#define VECTOR_EXTRAS_H

#include <vector_functions.h>
#include <iostream>
#include <sstream>
using namespace std;

inline __host__ __device__ uchar4 operator-(uchar4 a, uchar4 b)
{
    return make_uchar4(a.x - b.x, a.y - b.y, a.z - b.z,  a.w - b.w);
}

inline __host__ __device__ bool operator==(uchar4 a, uchar4 b)
{
    return a.x == b.x && a.y == b.y && a.z == b.z && a.w == b.w;
}

inline __host__ __device__ bool operator!=(uchar4 a, uchar4 b)
{
    return !(a == b);
}

/*
inline __host__ __device__ void operator-=(uchar4 &a, uchar4 b)
{
    a.x -= b.x; a.y -= b.y; a.z -= b.z; a.w -= b.w;
}

inline __host__ __device__ uchar4 operator-(uchar4 a, uchar b)
{
    return make_uchar4(a.x - b, a.y - b, a.z - b,  a.w - b);
}

inline __host__ __device__ uchar4 operator-(uchar b, uchar4 a)
{
    return make_uchar4(b - a.x, b - a.y, b - a.z, b - a.w);
}
*/

inline ostream &operator<<(ostream& ostr, const uchar4& p_u) 
{
	return ostr << "(" << (int) p_u.x << "," << (int) p_u.y << "," << (int) p_u.z << "," << (int) p_u.w << ")";
}

inline ostream &operator<<(ostream& ostr, const dim3& d) 
{
	return ostr << d.x << "," << d.y << "," << d.z;
}

inline ostream &operator<<(ostream& ostr, const int3& d) 
{
	return ostr << d.x << "," << d.y << "," << d.z;
}


#endif

