package net.dinkla.gpu.cuda.buffer;

/*
 * Copyright (c) 2012 by Jörn Dinkla, www.dinkla.com, All rights reserved.
 */

import jcuda.Pointer;
import net.dinkla.utils.Extent;
import net.dinkla.gpu.GpuException;
import net.dinkla.gpu.buffer.IHostBuffer;

import static jcuda.runtime.JCuda.cudaFreeHost;
import static jcuda.runtime.JCuda.cudaMallocHost;
import static jcuda.runtime.cudaError.cudaSuccess;

public class HostBuffer<T> extends BaseBuffer<T> implements IHostBuffer<T> {

    public HostBuffer(final Extent extent) {
        super(extent);
    }

    @Override
    public void malloc() {
        ptr = new Pointer();
        int err = cudaMallocHost(ptr, extent.getSize());
        if (err != cudaSuccess) {
            throw new GpuException(err, "HostBuffer.malloc()");
        }
        incVersion();
    }

    @Override
    public void free() {
        if (isAllocated()) {
            int err = cudaFreeHost(ptr);
            if (err != cudaSuccess) {
                throw new GpuException(err, "HostBuffer.free()");
            }
            resetVersion();
            ptr = null;
        }
    }

    /*
    public void putAt(long idx, final T value) {
        Pointer p = new Pointer(ptr, idx);
        cudaMemset(p, (int) value, (long) 1);
    }
    */

}
