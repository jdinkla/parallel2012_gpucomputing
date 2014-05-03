/*
 * Copyright (c) 2012 by J�rn Dinkla, www.dinkla.com, All rights reserved.
 */

#include <stdio.h>

__global__ void hello()
{
    int i = threadIdx.x;
	printf("Hello World %i\n", i);
}

int main()
{
	hello<<<1, 3>>>();
	cudaDeviceSynchronize();
}
