package net.dinkla.gpu

/*
 * Copyright (c) 2012 by Jörn Dinkla, www.dinkla.com, All rights reserved.
 */
 
class GpuException extends RuntimeException {

    final int rc
    final String msg

    GpuException(final int rc, final String msg) {
        super()
        this.rc = rc
        this.msg = msg
    }

    @Override
    String toString() {
        "GpuException ${rc} ${msg}"
    }

}