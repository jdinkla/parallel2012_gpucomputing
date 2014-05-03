package net.dinkla.utils

/*
 * Copyright (c) 2012 by Jörn Dinkla, www.dinkla.com, All rights reserved.
 */
 
class Version implements Comparator<Version> {

    private int version

    Version() {
        version = -1
    }

    void incVersion() {
        version++;
    }

    int getVersion() {
        version
    }

    void resetVersion() {
        version = -1;
    }

    @Override
    boolean equals(Object obj) {
        boolean result = false
        if (obj instanceof Version) {
            Version other = (Version) obj
            result = version == other.version
        }
        return super.equals(obj)
    }

    int compare(Version o1, Version o2) {
        o1.version.compare(o2.version)
    }

}
