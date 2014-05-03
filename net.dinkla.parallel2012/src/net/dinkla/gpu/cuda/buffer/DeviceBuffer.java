package net.dinkla.gpu.cuda.buffer;

/*
 * Copyright (c) 2012 by Jörn Dinkla, www.dinkla.com, All rights reserved.
 */

import jcuda.Pointer;
import jcuda.runtime.cudaStream_t;
import net.dinkla.utils.Extent;
import net.dinkla.gpu.GpuException;
import net.dinkla.gpu.buffer.IDeviceBuffer;
import net.dinkla.gpu.buffer.IHostBuffer;
import net.dinkla.utils.int3;

import static jcuda.runtime.JCuda.*;
import static jcuda.runtime.JCuda.cudaMemcpy;
import static jcuda.runtime.JCuda.cudaMemcpyAsync;
import static jcuda.runtime.cudaError.cudaSuccess;
import static jcuda.runtime.cudaMemcpyKind.cudaMemcpyDeviceToHost;
import static jcuda.runtime.cudaMemcpyKind.cudaMemcpyHostToDevice;

public class DeviceBuffer<T> extends BaseBuffer<T> implements IDeviceBuffer<T> {

    cudaStream_t stream;

    public DeviceBuffer(final Extent extent) {
        super(extent);
        this.stream = null;
}

    public DeviceBuffer(final Extent extent, cudaStream_t stream) {
        super(extent);
        this.stream = stream;
    }

    @Override
    public void malloc() {
        ptr = new Pointer();
        int err = cudaMalloc(ptr, extent.getSize());
        if (err != cudaSuccess) {
            throw new GpuException(err, "DeviceBuffer.malloc()");
        }
        incVersion();
    }

    @Override
    public void free() {
        if (isAllocated()) {
            int err = cudaFree(ptr);
            if (err != cudaSuccess) {
                throw new GpuException(err, "DeviceBuffer.free()");
            }
            resetVersion();
            ptr = null;
        }
    }

    // Kopiere vom Device

    public void copyFrom(IHostBuffer<T> host) {
        HostBuffer<T> h = (HostBuffer<T>) host;
        copyFrom(h.getPtr());
    }

    public void copyFrom(IHostBuffer<T> host, long size, int3 offset) {
        HostBuffer<T> h = (HostBuffer<T>) host;
        copyFrom(h.getPtr(), size, offset);
    }

    public void copyFrom(final Pointer hPtr) {
        copyFrom(hPtr, extent.getSize());
    }

    public void copyFrom(final Pointer hPtr, final long size) {
        copyFrom(hPtr, size, new int3(0, 0, 0));
    }

    public void copyFrom(final Pointer hPtr, long size, final int3 offset) {
        int err;
        Pointer ptr = this.ptr.withByteOffset(extent.index(offset));
        if (stream != null) {
            err = cudaMemcpyAsync(hPtr, ptr, size, cudaMemcpyDeviceToHost, stream);
        } else {
            err = cudaMemcpy(hPtr, ptr, size, cudaMemcpyDeviceToHost);
        }
        if (err != cudaSuccess) {
            throw new GpuException(err, "DeviceBuffer.copyFrom()");
        }
    }

    // Kopiere zum Device

    public void copyTo(IHostBuffer<T> host) {
        HostBuffer<T> h = (HostBuffer<T>) host;
        copyTo(h.getPtr());
    }

    public void copyTo(IHostBuffer<T> host, long size) {
        HostBuffer<T> h = (HostBuffer<T>) host;
        copyTo(h.getPtr(), size);
    }

    public void copyTo(final Pointer hPtr) {
        copyTo(hPtr, extent.getSize());
    }

    public void copyTo(final Pointer hPtr, final long size) {
        int err;
        if (stream != null) {
            err = cudaMemcpyAsync(ptr, hPtr, size, cudaMemcpyHostToDevice, stream);
        } else {
            err = cudaMemcpy(ptr, hPtr, size, cudaMemcpyHostToDevice);
        }
        if (err != cudaSuccess) {
            throw new GpuException(err, "DeviceBuffer.copyTo()");
        }
    }

}
