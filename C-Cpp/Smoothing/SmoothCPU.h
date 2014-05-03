/*
 * Copyright (c) 2012 by Jörn Dinkla, www.dinkla.com, All rights reserved.
 */

#ifndef SMOOTHCPU_H
#define SMOOTHCPU_H

#include "CExtent.h"

void smoothCPU(const CExtent& extent, const uchar4* h_input, uchar4* h_output, const int windowSize);

void smoothCPU(const int width, const int height, const int depth, 
			   const uchar4* h_input, uchar4* h_output, 
			   const int windowSize);

#endif 