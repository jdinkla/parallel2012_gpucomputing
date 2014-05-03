/*
 * Copyright (c) 2012 by Jörn Dinkla, www.dinkla.com, All rights reserved.
 */

#include <omp.h>
#include "Utils.h"
#include <vector_functions.h>
#include "vector_extras.h"
#include <iostream>
using namespace std;

void initializeBuffer(CHostBuffer<uchar4>& buf) {
	cout << "Initialising test data" << endl;
	initializeBuffer(buf.getPtr(), buf);
	buf.incVersion();
}

void initializeBuffer(uchar4* h_input, const CExtent& extent) {
	// Initialisiere Daten
#pragma omp parallel for
	for (int z=0; z<extent.depth; z++) {
		for (int y=0; y<extent.height; y++) {
			for (int x=0; x<extent.width; x++) {
				h_input[extent.index(x, y, z)] = make_uchar4(x, y, z, x+y+z);
			}
		}
	}
}

void initializeBuffer(uchar4* h_input, const int width, const int height) {
	// Initialisiere Daten
#pragma omp parallel for
	for (int y=0; y<height; y++) {
		for (int x=0; x<width; x++) {
			h_input[y * width + x] = make_uchar4(x, y, 0, x+y);
		}
	}
}


bool allEqual(uchar4* h_outputCPU, uchar4* h_output, const CExtent& extent) {
	int errors = 0;
	for (int z=0; z<extent.depth; z++) {
		for (int y=0; y<extent.height; y++) {
			for (int x=0; x<extent.width; x++) {
				const int idx = extent.index(x, y, z);
				const uchar4 cpu = h_outputCPU[idx];
				const uchar4 gpu = h_output[idx];
				const uchar4 diff = cpu - gpu;
				//cout << "CPU=" << h_outputCPU[idx] << endl;
				//cout << "GPU=" << h_output[idx] << endl;
				//cout << "Diff=" << diff << endl;
				if (diff != make_uchar4(0,0,0,0)) {
					cerr << "ERROR: difference " << diff << " at " << x << ", " << y << ", " << z 
						<< " CPU: " << cpu
					    << " GPU: " << gpu
						<< endl;
					errors++;
				}
				/*
				cout << "difference " << diff << " at " << x << ", " << y << ", " << z 
					<< " CPU: " << cpu
				    << " GPU: " << gpu
					<< endl;
					*/
			}
		}
	}
	cout << "Errors: " << errors << endl;
	return errors == 0;
}

