package net.dinkla.parallel2012.groovy

import com.nativelibs4java.opencl.CLDevice
import com.nativelibs4java.opencl.CLPlatform
import com.nativelibs4java.opencl.JavaCL
import com.nativelibs4java.opencl.CLContext
import com.nativelibs4java.opencl.CLQueue
import com.nativelibs4java.opencl.CLProgram
import com.nativelibs4java.opencl.CLKernel
import com.nativelibs4java.opencl.CLEvent

/*
 * Copyright (c) 2012 by Jörn Dinkla, www.dinkla.com, All rights reserved.
 */

class HelloWorldOpenCL {
    final static String src = """
__kernel void helloWorld() {
    int id = get_global_id(0);
}
"""
    static void main(String[] args) {
        CLContext context = JavaCL.createBestContext()
        CLQueue queue = context.createDefaultQueue()
        CLProgram program = context.createProgram(src)
        CLKernel kernel = program.createKernel("helloWorld")

        int[] globalSize = [3]
        int[] localSize = [3]
        CLEvent event = kernel.enqueueNDRange(queue, globalSize, localSize)
    }
}



