/*
 * Copyright (c) 2012 by J�rn Dinkla, www.dinkla.com, All rights reserved.
 */

#include "CL/opencl.h"

char *source = {
	"kernel helloWorld() {\n"
	"  int id = get_global_id(0);\n"
	"}\n"
};

int main()
{
	// Setup
	cl_device_id device;
	clGetDeviceIDs(0, CL_DEVICE_TYPE_DEFAULT, 1, &device, 0);
	cl_context context = clCreateContext(0, 1, &device, 0, 0, 0);
	cl_command_queue queue = clCreateCommandQueue(context, device, 
		(cl_command_queue_properties)0, 0);
	
	// Kernel erstellen
	cl_program program = clCreateProgramWithSource(context, 1, 
		(const char**)&source, 0, 0);
	clBuildProgram(program, 0, 0, 0, 0, 0);
	cl_kernel kernel = clCreateKernel(program, "helloWorld", 0);

	// Kernel ausf�hren
	size_t global_dimensions[] = {3,0,0};
	clEnqueueNDRangeKernel(queue, kernel, 1, 0, global_dimensions, 0, 0, 0, 0);
	
	// Aufr�umen
	clReleaseKernel(kernel);
	clReleaseProgram(program);
	clReleaseCommandQueue(queue);
	clReleaseContext(context);
}


