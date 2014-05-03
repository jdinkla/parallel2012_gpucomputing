/*
 * Copyright (c) 2012 by Jörn Dinkla, www.dinkla.com, All rights reserved.
 */

#ifndef COPY1D_H
#define COPY1D_H

#include "CExecConfig.h"
#include "CExtent.h"

void copy1D(const uchar4* d_input, uchar4* d_output, const int length);

void copy1D(const CExecConfig& config, const uchar4* d_input, uchar4* d_output, const int length);

void copy1D(const CExecConfig& config, const uchar4* d_input, uchar4* d_output, const CExtent& extent);

void copy1D();

void copy1D_h(const uchar4 *h_input,
			  uchar4 *h_output, 
			  const int length);

#endif

