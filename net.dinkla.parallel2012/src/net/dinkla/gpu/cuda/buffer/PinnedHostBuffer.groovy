package net.dinkla.gpu.cuda.buffer

import net.dinkla.utils.Extent

/*
 * Copyright (c) 2012 by Jörn Dinkla, www.dinkla.com, All rights reserved.
 */
 
class PinnedHostBuffer<T> extends BaseBuffer<T> {

    PinnedHostBuffer(final Extent extent) {
        super(extent)
    }

    @Override
    void malloc() {

    }

    @Override
    void free() {
        ptr = null
    }

}
