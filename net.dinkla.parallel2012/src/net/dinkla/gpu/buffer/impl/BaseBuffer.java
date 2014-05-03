package net.dinkla.gpu.buffer.impl;

/*
 * Copyright (c) 2012 by Jörn Dinkla, www.dinkla.com, All rights reserved.
 */

import net.dinkla.utils.Extent;
import net.dinkla.gpu.buffer.IBuffer;
import net.dinkla.gpu.buffer.IVersion;
import net.dinkla.utils.Version;

abstract public class BaseBuffer<T> implements IBuffer<T>, IVersion {

    protected Version version;
    final public Extent extent;

    public BaseBuffer(final Extent extent) {
        this.extent = extent;
        this.version = new Version();
    }

    abstract public void free();

    abstract public void malloc();

    public boolean isAllocated() {
        return version.getVersion() > 0;
    }

    public int getVersion() {
        return version.getVersion();
    }

    public void incVersion() {
        version.incVersion();
    }

    public void resetVersion() {
        version.resetVersion();
    }

    public Extent getExtent() {
        return extent;
    }

}
