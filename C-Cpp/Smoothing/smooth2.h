/*
 * Copyright (c) 2012 by Jörn Dinkla, www.dinkla.com, All rights reserved.
 */

#ifndef SMOOTH2_H
#define SMOOTH2_H

#include "CExecConfig.h"
#include "CExtent.h"

void smooth2(const CExecConfig& config, const uchar4* d_input, uchar4* d_output, const CExtent& extent, const int windowSize);

#endif

