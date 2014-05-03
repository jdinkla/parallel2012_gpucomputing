/*
 * Copyright (c) 2012 by Jörn Dinkla, www.dinkla.com, All rights reserved.
 */

package net.dinkla.utils

class Extent {

    final public int width
    final public int height
    final public int depth
    final int sizeOfElem      // size in bytes of one element

    final public int x
    final public int y
    final public int z
    final public int w

    Extent(int width, int height, int depth, final int sizeOfElem = 4) {
        this.width = width
        this.height = height
        this.depth = depth
        this.sizeOfElem = sizeOfElem
        this.x = width
        this.y = height
        this.z = depth
        this.w = sizeOfElem
    }

    int getNumberOfElements() {
        width * height * depth;
    }

    @Override
    boolean equals(Object obj) {
        boolean result
        if (obj instanceof Extent) {
            Extent other = (Extent) obj
            result = width == other.width && height == other.height && depth == other.depth
        } else {
            result = false
        }
        result
    }

    long index(final int x, final int y, final int z) {
        return ((long) z) * (width * height) + y * width + x;
    }

    long index(final int3 p) {
        return ((long) p.z) * (width * height) + p.y * width + p.x;
    }

    long getSize() {
        getNumberOfElements() * sizeOfElem
    }

}
