package net.dinkla.gpu.cuda

import static jcuda.driver.JCudaDriver.cuInit
import jcuda.driver.CUdevice

import static jcuda.driver.JCudaDriver.cuDeviceGet
import jcuda.driver.CUcontext

import static jcuda.driver.JCudaDriver.cuCtxCreate
import jcuda.driver.CUmodule

import static jcuda.driver.JCudaDriver.cuModuleLoad
import jcuda.driver.CUfunction

import static jcuda.driver.JCudaDriver.cuModuleGetFunction
import jcuda.driver.CUdeviceptr

import static jcuda.driver.JCudaDriver.cuMemAlloc

/*
 * Copyright (c) 2012 by Jörn Dinkla, www.dinkla.com, All rights reserved.
 */
 
class Cuda {

    static {
        cuInit(0);
    }

    static CUdevice device(final int num) {
        CUdevice device = new CUdevice()
        cuDeviceGet(device, num)
        return device
    }

    static CUcontext context(CUdevice device, final int num) {
        CUcontext context = new CUcontext()
        cuCtxCreate(context, 0, device)
        return context
    }

    static CUmodule module(final String ptx) {
        CUmodule module = new CUmodule();
        cuModuleLoad(module, ptx)
        return module
    }

    static CUfunction function(CUmodule module, final String name) {
        CUfunction function = new CUfunction();
        cuModuleGetFunction(function, module, name);
        return function
    }

    static CUdeviceptr alloc(final long numBytes) {
        CUdeviceptr ptr = new CUdeviceptr();
        cuMemAlloc(ptr, numBytes);
        return ptr
    }

}
