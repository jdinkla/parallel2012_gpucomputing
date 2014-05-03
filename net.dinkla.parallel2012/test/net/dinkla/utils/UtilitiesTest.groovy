package net.dinkla.utils

/*
 * Copyright (c) 2012 by Jörn Dinkla, www.dinkla.com, All rights reserved.
 */

class UtilitiesTest extends GroovyTestCase {

    static final String TEST1 = "abcdef"

    void testPrefix() {
        assertEquals("abc", Utilities.prefix(TEST1, 3))
        assertEquals(TEST1, Utilities.prefix(TEST1, 6))
        assertEquals(TEST1, Utilities.prefix(TEST1, 7))
        assertEquals(TEST1, Utilities.prefix(TEST1, 8))
        assertEquals("", Utilities.prefix(TEST1, 0))
    }

}
