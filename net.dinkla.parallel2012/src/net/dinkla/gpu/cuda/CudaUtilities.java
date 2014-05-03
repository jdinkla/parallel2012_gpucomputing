package net.dinkla.gpu.cuda;

/*
 * Copyright (c) 2012 by Jörn Dinkla, www.dinkla.com, All rights reserved.
 */

import jcuda.runtime.dim3;
import net.dinkla.utils.Extent;

public class CudaUtilities {

    static public dim3 asDim3(final Extent extent) {
        return new dim3(extent.width, extent.depth, extent.height);
    }

}
