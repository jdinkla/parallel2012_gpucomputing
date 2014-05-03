/*
 * Copyright (c) 2012 by Jörn Dinkla, www.dinkla.com, All rights reserved.
 */

#ifndef SMOOTH3D_H
#define SMOOTH3D_H

#include "CExecConfig.h"
#include "CExtent.h"
#include "CDeviceBuffer.h"

void smooth3d(const CExecConfig& config, const CDeviceBuffer<uchar4>& input, const CDeviceBuffer<uchar4>& output, const int windowSize);

#endif

