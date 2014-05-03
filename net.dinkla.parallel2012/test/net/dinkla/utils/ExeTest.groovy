package net.dinkla.utils

/*
 * Copyright (c) 2012 by Jörn Dinkla, www.dinkla.com, All rights reserved.
 */

class ExeTest extends GroovyTestCase {

    void testProcessOk() {
        def e = new Exe("nvcc.exe --help")
        e.process()
        assert(e.isOk())
    }

    void testProcessBroken() {
        shouldFail(IOException) {
            def e = new Exe("MyBrokennvcc.exe --help")
            e.process()
        }
    }

}
