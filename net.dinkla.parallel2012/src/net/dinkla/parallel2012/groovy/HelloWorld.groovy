package net.dinkla.parallel2012.groovy

import net.dinkla.gpu.cuda.Kernel
import net.dinkla.utils.Extent
import net.dinkla.gpu.cuda.ExecConfig
import static jcuda.runtime.JCuda.cudaDeviceSynchronize

/*
 * Copyright (c) 2012 by Jörn Dinkla, www.dinkla.com, All rights reserved.
 */
 
class HelloWorld {
    final static String src = "kernels/helloWorld.cu"
    final static String ptx = "build/helloWorld.ptx"
    static void main(String[] args) {
        cudaDeviceSynchronize()
        Kernel kernel = new Kernel(src, ptx, "hello")
        kernel.compile()

        ExecConfig config = new ExecConfig()
        config.grid = new Extent(2, 1, 1)
        config.threads = new Extent(3, 1, 1)
        kernel.run(config)
        cudaDeviceSynchronize()
    }
}

