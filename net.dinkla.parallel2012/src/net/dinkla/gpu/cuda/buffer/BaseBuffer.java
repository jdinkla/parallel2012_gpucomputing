package net.dinkla.gpu.cuda.buffer;

/*
 * Copyright (c) 2012 by Jörn Dinkla, www.dinkla.com, All rights reserved.
 */

import jcuda.Pointer;
import net.dinkla.utils.Extent;

abstract public class BaseBuffer<T> extends net.dinkla.gpu.buffer.impl.BaseBuffer<T> {

    Pointer ptr;

    public BaseBuffer(final Extent extent) {
        super(extent);
        this.ptr = null;
    }

    public Pointer getPtr() {
        return ptr;
    }

}

