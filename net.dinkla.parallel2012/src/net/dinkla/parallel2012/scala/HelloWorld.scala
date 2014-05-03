/*
* Copyright (c) 2012 by Jörn Dinkla, www.dinkla.com, All rights reserved.
*/

import com.nativelibs4java.opencl.{CLKernel, CLProgram, CLQueue,
                                   CLContext, CLEvent, JavaCL}

val src = """
__kernel void helloWorld() {
    int id = get_global_id(0);
}
                          """
val context : CLContext = JavaCL.createBestContext()
val queue : CLQueue = context.createDefaultQueue()
val program : CLProgram = context.createProgram(src)
val kernel : CLKernel = program.createKernel("helloWorld")

val globalSize = Array(3)
val localSize = Array(3)
val event : CLEvent = kernel.enqueueNDRange(queue, globalSize, localSize)


