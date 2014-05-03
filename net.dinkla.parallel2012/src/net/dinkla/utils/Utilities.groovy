package net.dinkla.utils

import static java.lang.StrictMath.min

/*
 * Copyright (c) 2012 by Jörn Dinkla, www.dinkla.com, All rights reserved.
 */
 
class Utilities {

    // Returns at most n characters of str
    static String prefix(final String str, final int n) {
        final int m = min(n, str.size())
        str.substring(0, m)
    }

    // Returns true, if fileA is newer than fileB of if fileB does not exist
    static boolean isNewer(final String fileA, final String fileB) {
        File a = new File(fileA)
        assert(a.exists())
        File b = new File(fileB)
        if (!b.exists()) {
            return true
        }
        return a.lastModified() > b.lastModified()
    }

    // Compares a and b
    static int compare(final int a, final int b) {
        a < b ? -1 : a == b ? 0 : 1
    }

}
