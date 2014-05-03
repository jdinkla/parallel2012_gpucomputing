package net.dinkla.gpu.buffer.impl;

/*
 * Copyright (c) 2012 by Jörn Dinkla, www.dinkla.com, All rights reserved.
 */

import net.dinkla.gpu.buffer.IBufferPair;
import net.dinkla.gpu.buffer.IDeviceBuffer;
import net.dinkla.gpu.buffer.IHostBuffer;

public class BufferPair<T> implements IBufferPair<T> {

    public final boolean sameSize;
    public final IHostBuffer<T> host;
    public final IDeviceBuffer<T> device;

    public BufferPair(final IHostBuffer<T> host, final IDeviceBuffer<T> device) {
        this.host = host;
        this.device = device;
        sameSize = host.getExtent().equals(device.getExtent());
    }

    public void malloc() {
        host.malloc();
        device.malloc();
    }

    public void free() {
        if (host.isAllocated()) {
            host.free();
        }
        if (device.isAllocated()) {
            device.free();
        }
    }

    public boolean isDeviceNewer() {
        return device.getVersion() > host.getVersion();
    }

    public boolean isHostNewer() {
        return host.getVersion() > device.getVersion();
    }

    public void updateDevice() {
        updateDevice(false);
    }

    public void updateDevice(final boolean force) {
        if (force || isHostNewer()) {
            device.copyTo(host);
        }
    }

    public void updateHost() {
        updateHost(false);
    }

    public void updateHost(final boolean force) {
        if (force || isDeviceNewer()) {
            device.copyFrom(host);
        }
    }

    public boolean isAllocated() {
        return host.isAllocated();
    }

    public IDeviceBuffer<T> getDevice() {
        return device;
    }

    public IHostBuffer<T> getHost() {
        return host;
    }

    /*
        void updateDevice(final int3 currentPos, final int3 size) {
            long sizeInBytes = size.x * size.y * size.z * sizeof(T)
            device.copyTo(host.getPtr(currentPos), sizeInBytes)
        }


        void updateHost(const int3& currentPos, const int3& size) {
            const T* ptr = host->getPtr(currentPos);
            const size_t sizeInBytes = size.x * size.y * size.z * sizeof(T);
            device->copyFrom(ptr, sizeInBytes);
        }

        void updateHost(const int3& currentPos, const int3& size, const int3& offset) {
            const size_t sizeInBytes = size.x * size.y * size.z * sizeof(T);
            device->copyFrom(host->getPtr(currentPos), sizeInBytes, offset);
        }

    */


}
