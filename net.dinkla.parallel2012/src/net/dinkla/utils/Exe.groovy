package net.dinkla.utils

/*
 * Copyright (c) 2012 by Jörn Dinkla, www.dinkla.com, All rights reserved.
 */
 
class Exe {

    protected String command
    protected String errorMessage
    protected String outputMessage
    int exitValue

    Exe(String command) {
        this.command = command
        reset()
    }

    protected void reset() {
        errorMessage = null
        outputMessage = null
        exitValue = 0
    }

    void process() {
        reset()
        System.out.println("Executing\n" + command)
        Process process = Runtime.getRuntime().exec(command)
        outputMessage = new String(process.getInputStream().text)
        errorMessage = new String(process.getErrorStream().text)
        try {
            exitValue = process.waitFor()
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new IOException("Interrupted while waiting for output", e);
        }
    }

    boolean isOk() {
        exitValue == 0
    }

    void check() {
        if (!isOk()) {
            println("process exitValue " + exitValue);
            println("errorMessage:\n" + errorMessage);
            println("outputMessage:\n" + outputMessage);
            throw new IOException("Could not create .ptx file: " + errorMessage);
        }
    }

    @Override
    String toString() {
        "Exe ${command} ${exitValue} out='${Utilities.prefix(outputMessage, 20)}' err='${Utilities.prefix(errorMessage, 20)}'"
    }

}
