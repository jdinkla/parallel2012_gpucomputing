package net.dinkla.utils

/*
 * Copyright (c) 2012 by Jörn Dinkla, www.dinkla.com, All rights reserved.
 */
 
class Timed {

    long startTime
    long endTime

    Timed() {
        startTime = endTime = 0
    }

    void start() {
        startTime = System.currentTimeMillis()
    }

    void stop() {
        endTime = System.currentTimeMillis()
    }

    long getDuration() {
        endTime - startTime
    }

}
