package net.dinkla.parallel2012.utils;/*
 * Copyright (c) 2012 by Jörn Dinkla, www.dinkla.com, All rights reserved.
 */

import net.dinkla.gpu.cuda.buffer.HostBuffer;
import net.dinkla.utils.uchar4;

public class Utilities {

    static public void initialize(HostBuffer<uchar4> buf) {
        for (int z=0; z<buf.extent.depth; z++) {
            for (int y=0; y<buf.extent.height; y++) {
                for (int x=0; x<buf.extent.width; x++) {
                    // buf[buf.extent.index(x, y, z)] = new uchar4(x, y, z, x+y+z)
                }
            }
        }
    }

}
