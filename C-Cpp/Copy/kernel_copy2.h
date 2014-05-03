/*
 * Copyright (c) 2012 by Jörn Dinkla, www.dinkla.com, All rights reserved.
 */

#ifndef KERNEL_COPY2_H
#define KERNEL_COPY2_H

#include "CExecConfig.h"
#include "CExtent.h"

void kernel_copy2(const CExecConfig& config, const uchar4* d_input, uchar4* d_output, const CExtent& extent);

#endif

