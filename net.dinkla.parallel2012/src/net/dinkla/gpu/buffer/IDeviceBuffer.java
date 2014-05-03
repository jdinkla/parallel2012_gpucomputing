package net.dinkla.gpu.buffer;/*
 * Copyright (c) 2012 by Jörn Dinkla, www.dinkla.com, All rights reserved.
 */

import jcuda.Pointer;
import net.dinkla.utils.int3;

public interface IDeviceBuffer<T> extends IBuffer<T> {

    public void copyFrom(final IHostBuffer<T> host);

    public void copyFrom(final IHostBuffer<T> host, long size, final int3 offset);

    public void copyTo(final IHostBuffer<T> host);

    public void copyTo(final IHostBuffer<T> host, final long size);

}
