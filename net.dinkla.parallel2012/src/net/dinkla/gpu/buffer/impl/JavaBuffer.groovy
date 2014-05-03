package net.dinkla.gpu.buffer.impl

import net.dinkla.utils.Extent

import jcuda.Pointer

/*
 * Copyright (c) 2012 by Jörn Dinkla, www.dinkla.com, All rights reserved.
 */
 
class JavaBuffer<T> extends BaseBuffer<T> {

    T[] elems

    JavaBuffer(Extent extent) {
        super(extent)
    }

    @Override
    void malloc() {
        elems = new T[extent.numberOfElements]
        ptr = Pointer.to(elems[0])
    }

    @Override
    void free() {
        version.resetVersion()
        ptr = null
    }

}
