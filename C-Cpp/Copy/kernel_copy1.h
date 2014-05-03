/*
 * Copyright (c) 2012 by Jörn Dinkla, www.dinkla.com, All rights reserved.
 */

#ifndef KERNEL_COPY_H
#define KERNEL_COPY_H

#include "CExecConfig.h"
#include "CExtent.h"

void kernel_copy1(const uchar4* d_input, uchar4* d_output, const int width, const int height);

void kernel_copy1(const CExecConfig& config, const uchar4* d_input, uchar4* d_output, const int width, const int height);

void kernel_copy1(const CExecConfig& config, const uchar4* d_input, uchar4* d_output, const CExtent& extent);

#endif

