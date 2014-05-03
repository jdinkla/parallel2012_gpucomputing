package net.dinkla.gpu.cuda

import jcuda.driver.CUmodule
import jcuda.driver.CUfunction
import net.dinkla.utils.Utilities
import jcuda.Pointer

import static jcuda.driver.JCudaDriver.cuLaunchKernel
import jcuda.driver.CUresult
import net.dinkla.gpu.GpuException

import static jcuda.runtime.JCuda.cudaConfigureCall
import static jcuda.runtime.JCuda.cudaLaunch
import jcuda.runtime.cudaStream_t

/*
* Copyright (c) 2012 by Jörn Dinkla, www.dinkla.com, All rights reserved.
*/
 
class Kernel {

    final String cuFile
    final String ptxFile
    final String name

    CUmodule module
    CUfunction function

    Kernel(final String cuFile, final String ptxFile, final String name) {
        this.cuFile = cuFile
        this.ptxFile = ptxFile
        this.name = name
    }

    void compile() {
        boolean srcIsOk = true
        if (Utilities.isNewer(cuFile, ptxFile)) {
            def nvcc = new Nvcc()
            srcIsOk = nvcc.compile(cuFile, ptxFile)
        }
        if (srcIsOk) {
            module = Cuda.module(ptxFile)
            function = Cuda.function(module, name)
        } else {
            throw new RuntimeException("Error during compilation of kernel")
        }
    }

    void run(final ExecConfig config, final Pointer params = null) {
        final int rc = cuLaunchKernel(function,
                config.grid.width, config.grid.height, config.grid.depth,
                config.threads.width, config.threads.height, config.threads.depth,
                config.sharedMemBytes,
                config.streamHandle,
                params ?: new Pointer(),
                null
        );
        if (rc != CUresult.CUDA_SUCCESS) {
            throw new GpuException(rc, toString() + ": " + CUresult.stringFor(rc))
        }
    }

    void run2(final ExecConfig ec, final Pointer params = null) {
        def grid = CudaUtilities.asDim3(ec.grid)
        def threads = CudaUtilities.asDim3(ec.threads)
        def stream  = new cudaStream_t()
        int rc = cudaConfigureCall(grid, threads, 0, stream)
        if (rc != CUresult.CUDA_SUCCESS) {
            throw new GpuException(rc, "cudaConfigureCall")
        }
        rc = cudaLaunch(name)
        if (rc != CUresult.CUDA_SUCCESS) {
            throw new GpuException(rc, "cudaLaunch")
        }
    }

    @Override
    String toString() {
        "Kernel ${name} ${grid} ${block} ${sharedMemBytes} ${streamHandle}"
    }

}
