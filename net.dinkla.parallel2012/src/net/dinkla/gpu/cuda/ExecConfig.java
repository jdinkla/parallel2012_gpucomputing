package net.dinkla.gpu.cuda;/*
 * Copyright (c) 2012 by Jörn Dinkla, www.dinkla.com, All rights reserved.
 */

import net.dinkla.utils.Extent;

public class ExecConfig {

    final Extent threads;
    final Extent grid;
    final int sharedMemBytes;
    //final jcuda.driver.CUstream stream;

    public ExecConfig() {
        threads = null;
        grid = null;
        sharedMemBytes = 0;
        //streamHandle = null;
    }

    public ExecConfig(final Extent size) {
        threads = new Extent(128, 1, 1);
        grid = new Extent((int) (size.width + threads.x - 1) / threads.x,
                (int) (size.height + threads.y - 1) / threads.y,
                (int) (size.depth + threads.z - 1) / threads.z);
        sharedMemBytes = 0;
        //streamHandle = null
    }

    public Extent getGrid() {
        return grid;
    }

    public Extent getThreads() {
        return grid;
    }

}
