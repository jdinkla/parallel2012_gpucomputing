package net.dinkla.utils;/*
 * Copyright (c) 2012 by Jörn Dinkla, www.dinkla.com, All rights reserved.
 */

public class uchar4 {

    byte x, y, z, w;

    uchar4(int x, int y, int z, int w) {
        this.x = (byte) x;
        this.y = (byte) y;
        this.z = (byte) z;
        this.w = (byte) w;
    }

    uchar4(byte x, byte y, byte z, byte w) {
        this.x = x;
        this.y = y;
        this.z = z;
        this.w = w;
    }

    @Override
    public String toString() {
        return "(" + x + "," + y + "," + z + "," + w + ")";
    }

}
