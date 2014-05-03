/*
 * Copyright (c) 2012 by Jörn Dinkla, www.dinkla.com, All rights reserved.
 */

#ifndef SMOOTH1_H
#define SMOOTH1_H

#include "CExecConfig.h"

void smooth1(const CExecConfig& config, const uchar4* d_input, uchar4* d_output, const int width, const int height, const int windowSize);

#endif

