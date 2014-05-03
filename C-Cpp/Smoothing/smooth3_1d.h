/*
 * Copyright (c) 2012 by Jörn Dinkla, www.dinkla.com, All rights reserved.
 */

#ifndef SMOOTH3_1D_H
#define SMOOTH3_1D_H

#include "CExecConfig.h"
#include "CExtent.h"

void smooth3_1d(const CExecConfig& config, const uchar4* d_input, uchar4* d_output, const CExtent& extent, const int windowSize);

#endif

