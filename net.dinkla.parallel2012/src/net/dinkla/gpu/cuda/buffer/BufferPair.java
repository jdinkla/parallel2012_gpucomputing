package net.dinkla.gpu.cuda.buffer;

/*
 * Copyright (c) 2012 by Jörn Dinkla, www.dinkla.com, All rights reserved.
 */

import jcuda.runtime.cudaStream_t;
import net.dinkla.utils.Extent;
import net.dinkla.gpu.buffer.IDeviceBuffer;
import net.dinkla.gpu.buffer.IHostBuffer;

public class BufferPair<T> extends net.dinkla.gpu.buffer.impl.BufferPair<T> {

    public BufferPair(final Extent extent) {
        super(new HostBuffer<T>(extent), new DeviceBuffer<T>(extent));
    }

    public BufferPair(final Extent extent, cudaStream_t stream) {
        super(new HostBuffer<T>(extent), new DeviceBuffer<T>(extent, stream));
    }

    public BufferPair(final Extent hostExtent, final Extent deviceExtent, cudaStream_t stream) {
        super(new HostBuffer<T>(hostExtent), new DeviceBuffer<T>(deviceExtent, stream));
    }

    public BufferPair(final IHostBuffer<T> host, final IDeviceBuffer<T> device) {
        super(host, device);
    }

}
