package net.dinkla.gpu.buffer;/*
 * Copyright (c) 2012 by Jörn Dinkla, www.dinkla.com, All rights reserved.
 */

public interface IBufferPair<T> extends IAllocated {

    public IHostBuffer<T> getHost();

    public IDeviceBuffer<T> getDevice();

    public void updateDevice();

    public void updateDevice(final boolean force);

    public void updateHost();

    public void updateHost(final boolean force);

    public boolean isHostNewer();

    public boolean isDeviceNewer();

}
