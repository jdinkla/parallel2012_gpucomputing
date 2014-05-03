/*
 * Copyright (c) 2012 by Jörn Dinkla, www.dinkla.com, All rights reserved.
 */

#include <stdio.h>

// extern "C" ist notwendig, damit cuModuleGetFunction die Funktion findet
extern "C"
__global__ void hello()
{
    int i = threadIdx.x;
    int j = blockIdx.x;
	printf("Hello World %i %i\n", i, j);
}

