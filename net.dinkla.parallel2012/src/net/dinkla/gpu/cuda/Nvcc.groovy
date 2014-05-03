package net.dinkla.gpu.cuda

import net.dinkla.utils.Exe

/*
 * Copyright (c) 2012 by Jörn Dinkla, www.dinkla.com, All rights reserved.
 */
 
class Nvcc {

    static final String CL_BIN = "C:\\Program Files (x86)\\Microsoft Visual Studio 9.0\\VC\\bin\\cl.exe";
    static final String NVCC_PATH = "D:\\ACC\\CUDA\\v4.2\\bin";

    boolean debug

    Nvcc() {
        debug = false
    }

    boolean compile(final String kernelFile, final String objFile) {
        String machine = "-m" + System.getProperty("sun.arch.data.model");
        String cc = """-gencode=arch=compute_20,code=\\"sm_20,compute_20\\" """
        String xcompiler = """-Xcompiler "/O2" """
        String command = """${NVCC_PATH}/nvcc -ccbin "${CL_BIN}" ${machine} ${cc} ${xcompiler} -ptx ${kernelFile} -o ${objFile}"""
        def e = new Exe(command)
        e.process()
        e.check()
        return e.isOk()
    }

}
