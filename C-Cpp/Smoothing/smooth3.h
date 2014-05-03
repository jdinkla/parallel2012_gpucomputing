/*
 * Copyright (c) 2012 by Jörn Dinkla, www.dinkla.com, All rights reserved.
 */

#ifndef SMOOTH3_H
#define SMOOTH3_H

#include "CExecConfig.h"
#include "CExtent.h"
#include "CDeviceBuffer.h"

void smooth3(const CExecConfig& config, const CExtent& extent, const uchar4* d_input, uchar4* d_output, const int windowSize);

void smooth3(const CExecConfig& config, const CDeviceBuffer<uchar4>& input, const CDeviceBuffer<uchar4>& output, const int windowSize);

#endif

