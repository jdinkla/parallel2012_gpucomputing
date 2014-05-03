/*
 * Copyright (c) 2012 by Jörn Dinkla, www.dinkla.com, All rights reserved.
 */
#ifndef UTILS_H
#define UTILS_H

#include "CExtent.h"
#include "CHostBuffer.h"

void initializeBuffer(uchar4* h_input, const CExtent& extent);

void initializeBuffer(CHostBuffer<uchar4>& buf);

void initializeBuffer(uchar4* h_input, const int width, const int height);

bool allEqual(uchar4* h_outputCPU, uchar4* h_output, const CExtent& extent);

#endif
