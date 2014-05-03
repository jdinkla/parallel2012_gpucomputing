package net.dinkla.parallel2012.java;

/*
 * Copyright (c) 2012 by Jörn Dinkla, www.dinkla.com, All rights reserved.
 */

import jcuda.Pointer;
import jcuda.driver.*;
import net.dinkla.gpu.cuda.Nvcc;
import static jcuda.driver.JCudaDriver.*;

public class JavaCUDADriverHelloWorld {

    public static void main(String[] args) {
        JCudaDriver.setExceptionsEnabled(true);

        final String src = "kernels/helloWorld.cu";
        final String ptx = "build/helloWorld.ptx";

        Nvcc nvcc = new Nvcc();
        nvcc.compile(src, ptx);

        cuInit(0);
        CUdevice device = new CUdevice();
        cuDeviceGet(device, 0);
        cuCtxCreate(new CUcontext(), 0, device);

        CUmodule module = new CUmodule();
        cuModuleLoad(module, ptx);

        CUfunction function = new CUfunction();
        cuModuleGetFunction(function, module, "hello");

        Pointer params = new Pointer();
        cuLaunchKernel(function,
                2, 1, 1,      // Grid
                3, 1, 1,      // Thread-Block
                0, null,      // Shared, Stream
                params, null
        );
        cuCtxSynchronize();
    }

}
