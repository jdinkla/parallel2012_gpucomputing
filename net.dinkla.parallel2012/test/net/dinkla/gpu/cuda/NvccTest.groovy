package net.dinkla.gpu.cuda

import net.dinkla.gpu.cuda.Nvcc

/*
 * Copyright (c) 2012 by Jörn Dinkla, www.dinkla.com, All rights reserved.
 */

class NvccTest extends GroovyTestCase {

    void testCompile() {
        final String src = "kernels/simple_kernel.cu"
        final String ptx = "build/simple_kernel.ptx"
        def f = new File(ptx)
        f.delete()
        assert(!f.exists())         // PTX does not exist
        def nvcc = new Nvcc()
        nvcc.compile(src, ptx)
        assert(f.exists())          // PTX exists
    }

}
