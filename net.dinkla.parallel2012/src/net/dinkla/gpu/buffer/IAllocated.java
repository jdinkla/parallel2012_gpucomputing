package net.dinkla.gpu.buffer;

/*
 * Copyright (c) 2012 by Jörn Dinkla, www.dinkla.com, All rights reserved.
 */

public interface IAllocated {

    public void malloc();

    public void free();

    public boolean isAllocated();

}
