/*
 * Copyright (c) 2012 by Jörn Dinkla, www.dinkla.com, All rights reserved.
 */

#ifndef SMOOTH_H
#define SMOOTH_H

#include "CExecConfig.h"
#include "CExtent.h"

void smooth(const CExecConfig& config, const CExtent& extent, const uchar4* d_input, uchar4* d_output, const int windowSize);

#endif
